#!/usr/bin/env python3
"""
Unified framework for short-horizon temperature forecasting.

Changes vs. original script
---------------------------
1. Adds scale-independent metrics (MASE, SkillScore) next to MAE/RMSE.
2. Uses a pluggable `model_fn` interface (default = ARIMAX via pmdarima.auto_arima).
3. Computes naïve-persistence errors inside every train/test split so metrics are
   comparable across stations and windows.
4. Consolidates metric aggregation at the end.

Date: 2025-08-05
"""

import os, glob, random, argparse, warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from tqdm import tqdm
import pmdarima as pm
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")         # silence pmdarima


# --------------------------------------------------------------------------- #
#                         Data preparation utilities                          #
# --------------------------------------------------------------------------- #

def prepare_station_data(station_id: str, base_dir: str = "NOAA_GSOD") -> pd.DataFrame | None:
    """Load & clean one GSOD station; return tidy DataFrame with all years."""
    pattern = os.path.join(base_dir, "*", f"{station_id}.csv")
    files = glob.glob(pattern)
    if not files:
        print(f"[WARN] No files for station {station_id}")
        return None

    df = pd.concat((pd.read_csv(fp, parse_dates=["DATE"]) for fp in files)).sort_values("DATE")
    num_cols = ['TEMP', 'DEWP', 'SLP', 'STP', 'VISIB', 'WDSP', 'PRCP']
    for col in num_cols:
        df[col] = (
            df[col].astype(str)
                  .str.strip()
                  .replace({'9999.9': np.nan, '999.9': np.nan, '99.99': np.nan, '': np.nan})
                  .astype(float)
        )

    # Calendar & lag features
    df["YEAR"]         = df["DATE"].dt.year
    df["MONTH"]        = df["DATE"].dt.month
    df["DAY_OF_YEAR"]  = df["DATE"].dt.dayofyear
    df["WEEK_OF_YEAR"] = df["DATE"].dt.isocalendar().week
    df["TEMP_LAG1"]    = df["TEMP"].shift(1)

    df = df.dropna(subset=["TEMP"])  # target must exist
    features = ['DEWP', 'SLP', 'STP', 'VISIB', 'WDSP', 'PRCP',
                'YEAR', 'MONTH', 'DAY_OF_YEAR', 'WEEK_OF_YEAR', 'TEMP_LAG1']

    # keep cols with ≤20 % missing
    keep = [c for c in features if df[c].isna().mean() <= 0.20]
    return df[keep + ["TEMP"]].dropna()


def load_station_list(txt_path: str) -> list[str]:
    with open(txt_path) as f:
        return [l.strip() for l in f if l.strip()]


# --------------------------------------------------------------------------- #
#                             Metric functions                                #
# --------------------------------------------------------------------------- #

def mase(y_true: np.ndarray, y_pred: np.ndarray, insample: np.ndarray) -> float:
    """
    Mean Absolute Scaled Error (Hyndman & Koehler 2006).
    Scale = mean(|y_t - y_{t-1}|) over in-sample training series.
    """
    scale = np.mean(np.abs(np.diff(insample)))
    return np.inf if scale == 0 else np.mean(np.abs(y_true - y_pred)) / scale


def skill_score(rmse_model: float, rmse_naive: float) -> float:
    """Skill score >0 indicates improvement over naïve forecast."""
    return 1.0 - (rmse_model / rmse_naive)


# --------------------------------------------------------------------------- #
#                           Model wrapper(s)                                  #
# --------------------------------------------------------------------------- #

def arimax_forecaster(y_train: np.ndarray,
                      X_train: np.ndarray,
                      X_test: np.ndarray,
                      horizon: int = 3) -> np.ndarray:
    """
    Default model: non-seasonal ARIMAX via pmdarima.auto_arima.
    Returns array of length `horizon`.
    """
    model = pm.auto_arima(
        y_train, exogenous=X_train,
        seasonal=False, stepwise=True,
        suppress_warnings=True, error_action="ignore"
    )
    return model.predict(n_periods=horizon, exogenous=X_test)


# --------------------------------------------------------------------------- #
#                        Single split evaluation                              #
# --------------------------------------------------------------------------- #

def evaluate_once(station_df: pd.DataFrame,
                  valid_years: list[int],
                  model_fn=arimax_forecaster,
                  horizon: int = 3) -> dict[str, float] | None:
    """
    • Randomly chooses a 90-day training + 3-day test window within one valid year.
    • Fits `model_fn`, predicts 3 steps, and computes MAE, RMSE, MASE & Skill.
    """
    year = random.choice(valid_years)
    df_year = station_df[station_df['YEAR'] == year].reset_index(drop=True)
    if len(df_year) < (90 + horizon):
        return None

    start = random.randint(0, len(df_year) - (90 + horizon))
    train = df_year.iloc[start:start + 90]
    test  = df_year.iloc[start + 90: start + 90 + horizon]

    y_train, y_test = train["TEMP"].values, test["TEMP"].values
    exog_cols = [c for c in station_df.columns if c != "TEMP"]
    X_train, X_test = train[exog_cols].values, test[exog_cols].values

    # Model forecast
    y_pred = model_fn(y_train, X_train, X_test, horizon=horizon)

    # Naïve persistence forecast (last observed temperature)
    y_naive = np.repeat(y_train[-1], horizon)

    # Metrics
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    rmse_naive = np.sqrt(mean_squared_error(y_test, y_naive))
    return {
        "MAE"  : mae,
        "RMSE" : rmse,
        "MASE" : mase(y_test, y_pred, y_train),
        "SKILL": skill_score(rmse, rmse_naive)
    }


# --------------------------------------------------------------------------- #
#                                    Main                                     #
# --------------------------------------------------------------------------- #

def main(args: argparse.Namespace) -> None:
    stations = load_station_list(args.station_list)
    print("→ Pre-processing station data (2000-2024)…")

    station_data, station_years = {}, {}
    for sid in stations:
        df = prepare_station_data(sid, args.base_dir)
        if df is None or df.empty:
            continue
        # valid yrs must have ≥90+3 days
        counts = df.groupby("YEAR").size()
        valid  = [y for y, n in counts.items() if 2000 <= y <= 2024 and n >= 93]
        if not valid:
            continue
        station_data[sid]  = df
        station_years[sid] = valid

    if not station_data:
        print("No usable stations found—exiting.")
        return
    print(f"✓ Prepared {len(station_data)} stations.")

    # --------------------------------------------------------------------- #
    #                         Monte-Carlo evaluation                        #
    # --------------------------------------------------------------------- #
    metrics = {"MAE": [], "RMSE": [], "MASE": [], "SKILL": []}
    rng = random.Random(args.seed)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = []
        for _ in range(args.iterations):
            sid = rng.choice(list(station_data.keys()))
            futures.append(
                pool.submit(
                    evaluate_once,
                    station_data[sid],
                    station_years[sid]
                )
            )

        for fut in tqdm(as_completed(futures), total=len(futures), desc="Evaluations"):
            res = fut.result()
            if res is None:
                continue
            for k in metrics:
                metrics[k].append(res[k])

    if metrics["MAE"]:
        print(f"\nSuccessful iterations: {len(metrics['MAE'])}/{args.iterations}")
        for k, v in metrics.items():
            print(f"Average {k}: {np.mean(v):.4f}")
    else:
        print("No successful evaluations completed.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Unified ARIMAX benchmark")
    p.add_argument("--base-dir",     default="NOAA_GSOD", help="Root folder with GSOD CSVs")
    p.add_argument("--station-list", default="listToTrain.txt", help="File of station IDs")
    p.add_argument("--iterations",   type=int, default=100, help="Number of random Monte-Carlo splits")
    p.add_argument("--workers",      type=int, default=16,   help="ThreadPool size")
    p.add_argument("--seed",         type=int, default=42,   help="Random seed for reproducibility")
    args = p.parse_args()
    main(args)



'''#!/usr/bin/env python3
"""
ARIMA-based baseline for station temperature forecasting
Uses auto_arima on 25 years of daily data, then evaluates on random 30-day samples from years 2000 to 2024
"""
import os
import glob
import random
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from tqdm import tqdm
import pmdarima as pm
from sklearn.metrics import mean_absolute_error, mean_squared_error

import warnings
warnings.filterwarnings('ignore')  # suppress pmdarima warnings


def prepare_station_data(station_id, base_dir='NOAA_GSOD'):
    """
    Load and preprocess GSOD station data for a given station_id.
    Drops features with >20% missing, returns DataFrame of features + TEMP.
    """
    pattern = os.path.join(base_dir, '*', f'{station_id}.csv')
    files = glob.glob(pattern)
    if not files:
        print(f"Warning: no files for station {station_id}")
        return None

    dfs = []
    for fp in files:
        df = pd.read_csv(fp, parse_dates=['DATE'])
        dfs.append(df)
    station_df = pd.concat(dfs).sort_values('DATE').reset_index(drop=True)

    # clean numeric
    num_cols = ['TEMP', 'DEWP', 'SLP', 'STP', 'VISIB', 'WDSP', 'PRCP']
    for col in num_cols:
        station_df[col] = (
            station_df[col].astype(str)
            .str.strip()
            .replace({'9999.9': np.nan, '999.9': np.nan, '99.99': np.nan, '': np.nan})
            .astype(float)
        )

    # calendar features & lag
    station_df['YEAR'] = station_df['DATE'].dt.year
    station_df['MONTH'] = station_df['DATE'].dt.month
    station_df['DAY_OF_YEAR'] = station_df['DATE'].dt.dayofyear
    station_df['WEEK_OF_YEAR'] = station_df['DATE'].dt.isocalendar().week
    station_df['TEMP_LAG1'] = station_df['TEMP'].shift(1)

    # drop rows without target
    station_df = station_df.dropna(subset=['TEMP'])

    # initial feature list
    features = ['DEWP', 'SLP', 'STP', 'VISIB', 'WDSP', 'PRCP',
                'YEAR', 'MONTH', 'DAY_OF_YEAR', 'WEEK_OF_YEAR', 'TEMP_LAG1']

    # drop columns with >20% missing
    na_frac = station_df[features].isna().mean()
    keep = na_frac[na_frac <= 0.2].index.tolist()
    df_final = station_df[keep + ['TEMP']].dropna()
    return df_final


def load_station_list(txt_path):
    with open(txt_path) as f:
        return [l.strip() for l in f if l.strip()]


def evaluate_single(station_id, df_all_years, valid_years):
    """
    Train-auto_arima evaluation for a random 90-day block from a random valid year.
    """
    # pick a year that has at least 93 days of data
    year = random.choice(valid_years)
    df_year = df_all_years[df_all_years['YEAR'] == year].reset_index(drop=True)

    # pick contiguous 90-day training window + 3-day test
    start = random.randint(0, len(df_year) - 93)
    train = df_year.iloc[start:start+90]
    test = df_year.iloc[start+90:start+93]

    y_train = train['TEMP'].values
    y_test = test['TEMP'].values
    exog_cols = [c for c in df_year.columns if c != 'TEMP']
    X_train = train[exog_cols].values
    X_test = test[exog_cols].values

    model = pm.auto_arima(
        y_train,
        exogenous=X_train,
        seasonal=False,
        stepwise=True,
        suppress_warnings=True,
        error_action='ignore'
    )

    y_pred = model.predict(n_periods=3, exogenous=X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    return mae, rmse


def main(args):
    stations = load_station_list(args.station_list)

    print("Preprocessing station data for years 2000-2024...")
    station_data = {}
    station_years = {}
    for station in stations:
        df = prepare_station_data(station, args.base_dir)
        if df is None or df.empty:
            continue
        # identify years with sufficient data
        grouped = df.groupby('YEAR')
        valid = [y for y, grp in grouped if 2000 <= y <= 2024 and len(grp) >= 93]
        if not valid:
            continue
        station_data[station] = df
        station_years[station] = valid

    if not station_data:
        print("No stations with valid data between 2000 and 2024. Exiting.")
        return

    print(f"Prepared data for {len(station_data)} stations.")

    # multithreaded evaluation
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as exe:
        futures = []
        for _ in range(args.iterations):
            station = random.choice(list(station_data.keys()))
            futures.append(
                exe.submit(
                    evaluate_single,
                    station,
                    station_data[station],
                    station_years[station]
                )
            )

        for f in tqdm(as_completed(futures), total=len(futures), desc="Evaluations"):
            try:
                results.append(f.result())
            except Exception:
                continue

    if results:
        maes, rmses = zip(*results)
        print(f"Successful iterations: {len(results)}")
        print(f"Average MAE: {np.mean(maes):.3f}")
        print(f"Average RMSE: {np.mean(rmses):.3f}")
    else:
        print("No successful iterations completed.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="ARIMA-based baseline forecasting")
    parser.add_argument('--base-dir', default='NOAA_GSOD', help='Base directory for GSOD data')
    parser.add_argument('--station-list', default='listToTrain.txt', help='Path to station list')
    parser.add_argument('--iterations', type=int, default=1000, help='Number of random trials')
    parser.add_argument('--workers', type=int, default=16, help='Number of threads to use')
    args = parser.parse_args()
    main(args)



'''