# XGB_trainNtest.py
import argparse
import os
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from tqdm import tqdm
from xgboost import XGBRegressor
from xgboost.callback import EarlyStopping


# ---------------- Config ----------------
FEATURE_COLUMNS = ["TEMP", "DEWP", "SLP", "STP", "VISIB", "WDSP", "PRCP"]
STATIC_COLUMNS = ["MonthOfFirstForcast", "LATITUDE", "LONGITUDE", "ELEVATION"]
YEARS = list(range(2000, 2025))
PLACEHOLDER_STRINGS = {"9999.9", "999.9", "99.9"}

# ---------------- Utils ----------------
def _coerce_date(s):
    try:
        if isinstance(s, (int, float)) and not np.isnan(s):
            s = str(int(s))
        s = str(s)
        if len(s) == 8 and s.isdigit():
            return datetime.strptime(s, "%Y%m%d").date()
        return pd.to_datetime(s).date()
    except Exception:
        return pd.NaT

def _read_station_dataframe(root_dir, station_id):
    dfs = []
    for y in YEARS:
        path = os.path.join(root_dir, f"{y}", f"{station_id}.csv")
        if os.path.exists(path):
            df = pd.read_csv(path, dtype=str)
            df["__YEAR__"] = y
            df["__STATION__"] = station_id
            dfs.append(df)
    if not dfs:
        return None
    df = pd.concat(dfs, ignore_index=True)
    df.columns = [c.strip().upper() for c in df.columns]

    if "DATE" not in df.columns:
        raise ValueError(f"{station_id}: missing DATE column.")
    df["DATE"] = df["DATE"].apply(_coerce_date)
    df = df.dropna(subset=["DATE"]).sort_values("DATE").reset_index(drop=True)

    # Ensure required columns exist
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    for sc in ["LATITUDE", "LONGITUDE", "ELEVATION"]:
        if sc not in df.columns:
            df[sc] = np.nan

    # Clean numerics (placeholders -> NaN -> numeric)
    def _clean_numeric(col):
        s = df[col].astype(str).str.strip()
        s = s.replace(list(PLACEHOLDER_STRINGS), np.nan)
        s = pd.to_numeric(s, errors="coerce")
        return s

    for col in FEATURE_COLUMNS + ["LATITUDE", "LONGITUDE", "ELEVATION"]:
        df[col] = _clean_numeric(col)

    # Impute per-feature mean (after removal)
    for col in FEATURE_COLUMNS:
        mean_val = df[col].mean(skipna=True)
        if pd.isna(mean_val):
            mean_val = 0.0
        df[col] = df[col].fillna(mean_val)

    # Station statics (first non-NaN, else 0)
    for sc in ["LATITUDE", "LONGITUDE", "ELEVATION"]:
        val = df[sc].dropna().iloc[0] if df[sc].notna().any() else 0.0
        df[sc] = float(val)

    return df

def _read_ids(path):
    with open(path, "r") as f:
        return [ln.strip() for ln in f if ln.strip()]

def _scale(train_vals_by_feat):
    concat = pd.concat(train_vals_by_feat, axis=1)
    mins = concat.min(axis=0).values.astype(np.float32)
    maxs = concat.max(axis=0).values.astype(np.float32)
    ranges = (maxs - mins).astype(np.float32)
    ranges[ranges == 0] = 1.0
    return mins, maxs, ranges

def mae(a, b):
    return float(np.mean(np.abs(a - b)))

def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))

def skill_score(y_true, y_pred):
    mse = np.mean((y_true - y_pred) ** 2)
    var = np.var(y_true)
    if var < 1e-12:
        return 0.0
    return float(1.0 - (mse / var))

# ------------- Dataset builder (10-day lookback) -------------
def build_samples(station_dfs, mins, ranges, lookback=10, horizon=15):
    X_list, Y_scaled_list, Y_real_list = [], [], []
    for df in station_dfs:
        feats = df[FEATURE_COLUMNS].astype(float).values  # (N,7)
        dates = pd.to_datetime(df["DATE"]).dt.date.values
        lat = float(df["LATITUDE"].iloc[0])
        lon = float(df["LONGITUDE"].iloc[0])
        elev = float(df["ELEVATION"].iloc[0])

        N = len(df)
        if N < lookback + horizon:
            continue

        # pre-scale features once
        feats_s = (feats - mins.reshape(1, -1)) / ranges.reshape(1, -1)

        for i in range(lookback, N - horizon + 1):
            hist_s = feats_s[i - lookback:i, :]            # (10,7) scaled
            fut_s  = feats_s[i:i + horizon, :]             # (15,7) scaled
            fut_r  = feats[i:i + horizon, :]               # (15,7) real (for metrics)
            month  = pd.Timestamp(dates[i]).month

            x_flat = np.concatenate([hist_s.reshape(-1), np.array([month, lat, lon, elev], dtype=np.float32)])
            y_s    = fut_s.reshape(-1)     # (15*7,)
            y_r    = fut_r.reshape(-1)     # (15*7,)

            X_list.append(x_flat.astype(np.float32))
            Y_scaled_list.append(y_s.astype(np.float32))
            Y_real_list.append(y_r.astype(np.float32))

    if not X_list:
        return None, None, None

    X = np.stack(X_list, axis=0)                    # (M, 10*7+4)
    Y_scaled = np.stack(Y_scaled_list, axis=0)      # (M, 15*7)
    Y_real = np.stack(Y_real_list, axis=0)          # (M, 15*7)
    return X, Y_scaled, Y_real

# ---------------- Training per target (parallel) ----------------
def fit_one_target(i, X_tr, y_tr, X_va, y_va, args, tree_method, predictor):
    est = XGBRegressor(
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        reg_alpha=args.reg_alpha,
        reg_lambda=args.reg_lambda,
        objective="reg:squarederror",
        n_jobs=args.inner_jobs,               # inner threads per target
        random_state=args.seed + i,
        tree_method=tree_method,
        predictor=predictor,
    )

    # Try early_stopping_rounds kwarg (some versions support it)
    try:
        est.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            verbose=False,
            early_stopping_rounds=args.early_stopping_rounds,
        )
        return i, est
    except TypeError:
        pass  # fall through to callbacks-based ES

    # Try callbacks-based early stopping (version-agnostic)
    try:
        es_cb = EarlyStopping(
            rounds=args.early_stopping_rounds,
            save_best=True,        # keep best iteration
            maximize=False         # RMSE is minimized
            # metric_name left unset -> uses first eval metric (RMSE for reg:squarederror)
        )
        est.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            verbose=False,
            callbacks=[es_cb],
        )
        return i, est
    except TypeError:
        pass  # fall through to no early stopping

    # Fallback: train without early stopping
    est.fit(X_tr, y_tr, verbose=False)
    return i, est



# ---------------- Main ----------------
def main():
    parser = argparse.ArgumentParser(description="Train + Test XGBoost (10-day lookback) for GSOD (parallel, with progress bar)")
    parser.add_argument("--dataRoot", type=str, default=".\\NOAA_GSOD")
    parser.add_argument("--trainList", type=str, default=".\\listToTrain.txt")
    parser.add_argument("--testList", type=str, default=".\\listToTest.txt")
    parser.add_argument("--evalDir", type=str, default=".\\eval")
    parser.add_argument("--lookback", type=int, default=10)
    parser.add_argument("--horizon", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)

    # XGB hyperparams
    parser.add_argument("--n_estimators", type=int, default=2000)
    parser.add_argument("--learning_rate", type=float, default=0.05)
    parser.add_argument("--max_depth", type=int, default=6)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--colsample_bytree", type=float, default=0.8)
    parser.add_argument("--reg_alpha", type=float, default=0.0)
    parser.add_argument("--reg_lambda", type=float, default=1.0)
    parser.add_argument("--early_stopping_rounds", type=int, default=50)

    # Parallelism
    parser.add_argument("--outer_jobs", type=int, default=8, help="parallel targets (thread pool workers)")
    parser.add_argument("--inner_jobs", type=int, default=1, help="threads per XGBRegressor (avoid oversubscription)")

    # GPU
    parser.add_argument("--use_gpu", action="store_true", help="use GPU via tree_method=gpu_hist if available")

    args = parser.parse_args()

    np.random.seed(args.seed)
    os.makedirs(args.evalDir, exist_ok=True)

    # Station lists
    train_ids = _read_ids(args.trainList)
    test_ids  = _read_ids(args.testList)

    print("Stations for TRAIN:")
    for s in train_ids: print("  ", s)
    print("Stations for TEST/VAL:")
    for s in test_ids: print("  ", s)

    # Load dataframes
    def _load_many(ids):
        out = []
        for sid in ids:
            df = _read_station_dataframe(args.dataRoot, sid)
            if df is None or len(df) < (args.lookback + args.horizon + 5):
                print(f"Skipping {sid} (no data / too short).")
                continue
            out.append(df)
        return out

    train_dfs = _load_many(train_ids)
    test_dfs  = _load_many(test_ids)

    if not train_dfs or not test_dfs:
        print("Insufficient train/test data.")
        return

    # Train min/max from TRAIN only
    train_feature_series = []
    for col in FEATURE_COLUMNS:
        series_list = [df[col].astype(float) for df in train_dfs]
        train_feature_series.append(pd.concat(series_list, ignore_index=True).rename(col))
    mins, maxs, ranges = _scale(train_feature_series)

    print("\nFeature scaling (train set): min/max")
    for i, col in enumerate(FEATURE_COLUMNS):
        print(f"  {col:>6s}  min={mins[i]:10.4f}  max={maxs[i]:10.4f}")

    # Build samples (10-day lookback)
    X_tr, Y_tr_s, _ = build_samples(train_dfs, mins, ranges, lookback=args.lookback, horizon=args.horizon)
    X_te, Y_te_s, Y_te_r = build_samples(test_dfs,  mins, ranges, lookback=args.lookback, horizon=args.horizon)

    if X_tr is None or X_te is None:
        print("No samples could be built; check data length.")
        return

    print(f"\nSamples: train={len(X_tr)}  val/test={len(X_te)}")
    out_dim = Y_tr_s.shape[1]  # 15*7 = 105

    # Decide GPU/CPU
    tree_method = "gpu_hist" if args.use_gpu else "hist"
    predictor   = "gpu_predictor" if args.use_gpu else "auto"
    print(f"\nTraining XGBoost per-target with early stopping "
          f"({'GPU' if args.use_gpu else 'CPU'}; outer_jobs={args.outer_jobs}, inner_jobs={args.inner_jobs})")

    # ---- Parallel training over targets with progress bar
    estimators = [None] * out_dim
    with ThreadPoolExecutor(max_workers=max(1, args.outer_jobs)) as ex, \
         tqdm(total=out_dim, desc="Training targets", unit="tgt") as pbar:

        futures = []
        for i in range(out_dim):
            fut = ex.submit(
                fit_one_target,
                i,
                X_tr, Y_tr_s[:, i],
                X_te, Y_te_s[:, i],
                args, tree_method, predictor
            )
            futures.append(fut)

        for fut in as_completed(futures):
            i, est = fut.result()
            estimators[i] = est
            pbar.update(1)

    # ---- Predict (scaled), then inverse-scale to real units per feature/day
    # Stack predictions column-wise
    Y_pred_s = np.column_stack([est.predict(X_te) for est in estimators]).astype(np.float32)  # (N, 15*7)
    Y_pred_s = Y_pred_s.reshape(-1, args.horizon, len(FEATURE_COLUMNS))
    Y_true_r = Y_te_r.reshape(-1, args.horizon, len(FEATURE_COLUMNS))

    # inverse scaling: y_real = s*(max-min) + min
    mins_b = mins.reshape(1, 1, -1)
    ranges_b = ranges.reshape(1, 1, -1)
    Y_pred_r = Y_pred_s * ranges_b + mins_b

    # ---- Metrics for Day 1,7,15
    horizon_map = {1: 0, 7: 6, 15: 14}
    rows = []
    for day, idx in horizon_map.items():
        if idx >= args.horizon:
            continue
        for fi, feat in enumerate(FEATURE_COLUMNS):
            y_t = Y_true_r[:, idx, fi]
            y_p = Y_pred_r[:, idx, fi]
            m_mae = mae(y_t, y_p)
            m_rmse = rmse(y_t, y_p)
            m_skill = skill_score(y_t, y_p)
            rows.append({"Horizon": f"Day {day}", "Feature": feat, "MAE": m_mae, "RMSE": m_rmse, "Skill": m_skill})

    df_out = pd.DataFrame(rows, columns=["Horizon", "Feature", "MAE", "RMSE", "Skill"])

    # ---- Save CSV
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(args.evalDir, f"XGB_{timestamp}.csv")
    df_out.to_csv(out_path, index=False)
    print(f"\nSaved evaluation to: {out_path}")

    # Quick preview
    print("\nPreview:")
    print(df_out.head(9).to_string(index=False))

if __name__ == "__main__":
    main()
