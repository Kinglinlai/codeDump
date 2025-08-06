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
                      horizon: int = 15) -> np.ndarray:
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
                  horizon: int = 15) -> dict[str, float] | None:
    """
    • Randomly chooses a 180-day training + 15-day test window within one valid year.
    • Fits `model_fn`, predicts 15 steps, and computes MAE, RMSE, MASE & Skill.
    """
    year = random.choice(valid_years)
    df_year = station_df[station_df['YEAR'] == year].reset_index(drop=True)
    if len(df_year) < (180 + horizon):
        return None

    start = random.randint(0, len(df_year) - (180 + horizon))
    train = df_year.iloc[start:start + 180]
    test  = df_year.iloc[start + 180: start + 180 + horizon]

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
        # valid yrs must have 180+15 days
        counts = df.groupby("YEAR").size()
        valid  = [y for y, n in counts.items() if 2000 <= y <= 2024 and n >= 180 +15]
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

