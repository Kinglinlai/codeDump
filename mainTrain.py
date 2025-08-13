# mainTrain.py
import argparse
import importlib
import json
import os
import sys
import time
from datetime import datetime
from glob import glob

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

FEATURE_COLUMNS = [
    "TEMP", "DEWP", "SLP", "STP", "VISIB", "WDSP", "PRCP",
]
STATIC_COLUMNS = ["MonthOfFirstForcast", "LATITUDE", "LONGITUDE", "ELEVATION"]
YEARS = list(range(2000, 2025))
PLACEHOLDER_STRINGS = {"9999.9", "999.9", "99.9"}

# ----------------------- Utilities -----------------------

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
    # concat years; ignore missing files
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

    # normalize column names
    df.columns = [c.strip().upper() for c in df.columns]

    # parse date
    date_col = "DATE" if "DATE" in df.columns else None
    if date_col is None:
        raise ValueError(f"{station_id}: missing DATE column.")
    df["DATE"] = df["DATE"].apply(_coerce_date)
    df = df.dropna(subset=["DATE"]).sort_values("DATE").reset_index(drop=True)

    # keep only used columns
    needed = set(["DATE"] + FEATURE_COLUMNS + [c for c in STATIC_COLUMNS if c != "MonthOfFirstForcast"])
    present = [c for c in df.columns if c in set([c.upper() for c in needed])]
    # ensure required feature columns exist
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan  # will be imputed
    for sc in ["LATITUDE", "LONGITUDE", "ELEVATION"]:
        if sc not in df.columns:
            df[sc] = np.nan

    # Cast numeric-like strings; turn placeholders -> NaN
    def _clean_numeric(col):
        s = df[col].astype(str).str.strip()
        s = s.replace(list(PLACEHOLDER_STRINGS), np.nan)
        s = pd.to_numeric(s, errors="coerce")
        return s

    for col in FEATURE_COLUMNS + ["LATITUDE", "LONGITUDE", "ELEVATION"]:
        df[col] = _clean_numeric(col)

    # Simple per-column mean imputation AFTER removing placeholders
    for col in FEATURE_COLUMNS:
        mean_val = df[col].mean(skipna=True)
        if pd.isna(mean_val):
            mean_val = 0.0
        df[col] = df[col].fillna(mean_val)

    # Static fields: use the first non-NaN as the station's static
    statics = {}
    for sc in ["LATITUDE", "LONGITUDE", "ELEVATION"]:
        val = df[sc].dropna().iloc[0] if df[sc].notna().any() else 0.0
        statics[sc] = float(val)

    # Attach static columns to every row (handy at sample creation time)
    for sc in ["LATITUDE", "LONGITUDE", "ELEVATION"]:
        df[sc] = statics[sc]

    return df

def _build_minmax(train_series_list):
    # train_series_list: list of pd.Series (one per feature concatenated over all stations)
    concat = pd.concat(train_series_list, axis=1)
    mins = concat.min(axis=0)
    maxs = concat.max(axis=0)
    # Avoid zero range
    ranges = (maxs - mins).replace(0, 1.0)
    return mins.values.astype(np.float32), maxs.values.astype(np.float32), ranges.values.astype(np.float32)

def _scale_array(arr, mins, ranges):
    # arr: (..., 7)
    return (arr - mins) / ranges

# ----------------------- Dataset -----------------------

class GSODForecastDataset(Dataset):
    """
    Creates rolling windows of 90 days history -> 15 days future.
    Each item returns:
        x_hist: (90, 7) scaled features
        x_static: (4,) [MonthOfFirstForcast, LAT, LON, ELEV]
        y_future: (15, 7) scaled features (same scaling as inputs)
    """
    def __init__(self, station_dfs, mins, ranges, window_in=90, window_out=15):
        self.samples = []
        self.window_in = window_in
        self.window_out = window_out
        self.mins = mins
        self.ranges = ranges

        for df in station_dfs:
            feats = df[FEATURE_COLUMNS].astype(float).values  # (N,7)
            dates = pd.to_datetime(df["DATE"]).dt.date.values
            lat = float(df["LATITUDE"].iloc[0])
            lon = float(df["LONGITUDE"].iloc[0])
            elev = float(df["ELEVATION"].iloc[0])

            N = len(df)
            for i in range(window_in, N - window_out + 1):
                hist = feats[i - window_in:i, :]  # (90,7)
                fut = feats[i:i + window_out, :]  # (15,7)
                first_forecast_day = dates[i]     # the day immediately after the history window
                month = pd.Timestamp(first_forecast_day).month

                # scale
                hist_s = _scale_array(hist, self.mins, self.ranges)
                fut_s = _scale_array(fut, self.mins, self.ranges)
                x_static = np.array([month, lat, lon, elev], dtype=np.float32)

                self.samples.append((hist_s.astype(np.float32), x_static, fut_s.astype(np.float32)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        h, s, y = self.samples[idx]
        # also provide flattened version for simple models expecting R^{90*7+4}
        flat = np.concatenate([h.reshape(-1), s], axis=0).astype(np.float32)
        return {
            "hist": torch.from_numpy(h),          # (90,7)
            "static": torch.from_numpy(s),        # (4,)
            "flat": torch.from_numpy(flat),       # (90*7+4,)
            "target": torch.from_numpy(y),        # (15,7)
        }

# ----------------------- Training Orchestrator -----------------------

def main():
    parser = argparse.ArgumentParser(description="Weather Forecast Trainer")
    parser.add_argument("--dataRoot", type=str, default=".\\NOAA_GSOD", help="Root folder of GSOD data")
    parser.add_argument("--trainList", type=str, default=".\\listToTrain.txt")
    parser.add_argument("--testList", type=str, default=".\\listToTest.txt")
    parser.add_argument("--modelName", type=str, default="LSTM", help="Model name prefix, e.g., lstm, gru, mlp")
    parser.add_argument("--batchSize", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=7, help="Early stopping patience (epochs)")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--numWorkers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--saveDir", type=str, default=".\\model")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.saveDir, exist_ok=True)

    # read station lists
    def _read_ids(path):
        with open(path, "r") as f:
            return [ln.strip() for ln in f if ln.strip()]
    train_ids = _read_ids(args.trainList)
    test_ids  = _read_ids(args.testList)

    print("Stations for TRAIN:")
    for s in train_ids: print("  ", s)
    print("Stations for TEST/VAL:")
    for s in test_ids: print("  ", s)

    # load dataframes
    def _load_many(ids):
        dfs = []
        for sid in ids:
            df = _read_station_dataframe(args.dataRoot, sid)
            if df is None or len(df) < 120:  # need enough length
                print(f"Skipping {sid} (no data / too short).")
                continue
            dfs.append(df)
        return dfs

    train_dfs = _load_many(train_ids)
    test_dfs  = _load_many(test_ids)

    # Compute min-max on TRAIN ONLY, by concatenating features across all train stations
    train_feature_series = []
    for col in FEATURE_COLUMNS:
        series_list = [df[col].astype(float) for df in train_dfs]
        if series_list:
            train_feature_series.append(pd.concat(series_list, ignore_index=True).rename(col))
        else:
            train_feature_series.append(pd.Series([], dtype=float, name=col))

    mins, maxs, ranges = _build_minmax(train_feature_series)

    # print min/max per feature
    print("\nFeature scaling (train set): min/max")
    for i, col in enumerate(FEATURE_COLUMNS):
        print(f"  {col:>6s}  min={mins[i]:8.4f}  max={maxs[i]:8.4f}")

    # datasets
    train_ds = GSODForecastDataset(train_dfs, mins, ranges)
    val_ds   = GSODForecastDataset(test_dfs, mins, ranges)

    print(f"\nPrepared samples: train={len(train_ds)}  val={len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batchSize, shuffle=True, num_workers=args.numWorkers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batchSize, shuffle=False, num_workers=args.numWorkers, pin_memory=True)
    # dynamic import: {modelName}_train.py must sit alongside this file
    module_name = f"{str(args.modelName).upper()}_train"
    try:
        trainer_mod = importlib.import_module(module_name)
    except ModuleNotFoundError:
        print(f"ERROR: Could not find {module_name}.py next to mainTrain.py")
        sys.exit(1)

    # Build a config to pass to model trainers (so they don't re-derive shapes)
    sample0 = train_ds[0]
    config = {
        "feature_names": FEATURE_COLUMNS,
        "static_names": STATIC_COLUMNS,
        "x_hist_shape": tuple(sample0["hist"].shape),   # (90,7)
        "x_flat_dim": int(sample0["flat"].numel()),     # 90*7+4
        "y_shape": tuple(sample0["target"].shape),      # (15,7)
        "scaling": {"mins": mins.tolist(), "maxs": maxs.tolist()},
        "optim": {"lr": args.lr},
        "device": args.device,
        "epochs": args.epochs,
        "patience": args.patience,
        "modelName": args.modelName,
    }

    # ---- Train via module API ----
    best_state, history = trainer_mod.train(
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        progress_bar=True,
        early_stopping=True,
    )

    # ---- Save best model ----
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(args.saveDir, f"{args.modelName}_{timestamp}.pt")
    torch.save({
        "model_state_dict": best_state,
        "config": config,
        "train_history": history,
    }, save_path)
    print(f"\nSaved best model to: {save_path}")

    # Print final val score if provided in history
    if history and "best_val_loss" in history:
        print(f"Best validation loss (MSE): {history['best_val_loss']:.6f}")

if __name__ == "__main__":
    main()
