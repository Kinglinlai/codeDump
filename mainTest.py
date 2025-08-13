# mainTest.py
import argparse
import os
import sys
import time
import importlib
from glob import glob
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

FEATURE_COLUMNS = ["TEMP", "DEWP", "SLP", "STP", "VISIB", "WDSP", "PRCP"]
STATIC_COLUMNS = ["MonthOfFirstForcast", "LATITUDE", "LONGITUDE", "ELEVATION"]
YEARS = list(range(2000, 2025))
PLACEHOLDER_STRINGS = {"9999.9", "999.9", "99.9"}

# ============== I/O & Cleaning (mirrors mainTrain.py) ==============

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

class GSODForecastDataset(Dataset):
    """
    90-day history -> 15-day future.
    Uses provided mins/maxs for scaling.
    """
    def __init__(self, station_dfs, mins, maxs, window_in=90, window_out=15):
        self.samples = []
        self.mins = np.asarray(mins, dtype=np.float32)
        self.maxs = np.asarray(maxs, dtype=np.float32)
        self.ranges = (self.maxs - self.mins).astype(np.float32)
        self.ranges[self.ranges == 0] = 1.0
        self.window_in = window_in
        self.window_out = window_out

        for df in station_dfs:
            feats = df[FEATURE_COLUMNS].astype(float).values  # (N,7)
            dates = pd.to_datetime(df["DATE"]).dt.date.values
            lat = float(df["LATITUDE"].iloc[0])
            lon = float(df["LONGITUDE"].iloc[0])
            elev = float(df["ELEVATION"].iloc[0])

            N = len(df)
            for i in range(window_in, N - window_out + 1):
                hist = feats[i - window_in:i, :]           # (90,7)
                fut = feats[i:i + window_out, :]           # (15,7)
                month = pd.Timestamp(dates[i]).month

                hist_s = (hist - self.mins) / self.ranges
                fut_s  = (fut  - self.mins) / self.ranges
                x_static = np.array([month, lat, lon, elev], dtype=np.float32)

                flat = np.concatenate([hist_s.reshape(-1), x_static], axis=0).astype(np.float32)

                self.samples.append((
                    hist_s.astype(np.float32),
                    x_static,
                    flat,
                    fut_s.astype(np.float32),
                    fut.astype(np.float32)   # keep real units for convenience (optional)
                ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        h, s, f, y_s, y_real = self.samples[idx]
        return {
            "hist": torch.from_numpy(h),          # (90,7) scaled
            "static": torch.from_numpy(s),        # (4,)
            "flat": torch.from_numpy(f),          # (90*7+4,)
            "target": torch.from_numpy(y_s),      # (15,7) scaled
            "target_real": torch.from_numpy(y_real), # (15,7) real units
        }

# ============== Metrics ==============

def mae(a, b):
    return np.mean(np.abs(a - b))

def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))

def skill_score(y_true, y_pred):
    """
    Nash–Sutcliffe / R^2-like skill: 1 - MSE/Var(y_true)
    Returns 0 if variance is ~0 to avoid div-by-zero.
    """
    mse = np.mean((y_true - y_pred) ** 2)
    var = np.var(y_true)
    if var < 1e-12:
        return 0.0
    return 1.0 - (mse / var)

# ============== Main ==============

def _read_ids(path):
    with open(path, "r") as f:
        return [ln.strip() for ln in f if ln.strip()]

def _find_latest_model(model_dir, abbrev):
    files = glob(os.path.join(model_dir, "*.pt"))
    cand = []
    prefix = abbrev.lower()
    for f in files:
        base = os.path.basename(f)
        if base.lower().startswith(prefix + "_"):
            cand.append((os.path.getmtime(f), f))
    if not cand:
        return None
    cand.sort(key=lambda x: x[0], reverse=True)
    return cand[0][1]

def _import_trainer_module(abbrev):
    # Try exact, lower, and upper variants: {abbrev}_train
    names = [
        f"{abbrev}_train",
        f"{abbrev.lower()}_train",
        f"{abbrev.upper()}_train",
    ]
    last_err = None
    for n in names:
        try:
            return importlib.import_module(n)
        except ModuleNotFoundError as e:
            last_err = e
    raise last_err

def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained weather forecast model")
    parser.add_argument("--modelAbbrev", type=str, required=True, help='e.g., "LSTM" (expects LSTM_train.py)')
    parser.add_argument("--dataRoot", type=str, default=".\\NOAA_GSOD")
    parser.add_argument("--testList", type=str, default=".\\listToTest.txt")
    parser.add_argument("--modelDir", type=str, default=".\\model")
    parser.add_argument("--evalDir", type=str, default=".\\eval")
    parser.add_argument("--batchSize", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(args.evalDir, exist_ok=True)

    # ---- Locate checkpoint
    ckpt_path = _find_latest_model(args.modelDir, args.modelAbbrev)
    if ckpt_path is None:
        print(f"No checkpoint found in {args.modelDir} for prefix '{args.modelAbbrev}_*'.")
        sys.exit(1)

    print(f"Using checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    config = ckpt.get("config", {})
    mins = np.array(config["scaling"]["mins"], dtype=np.float32)
    maxs = np.array(config["scaling"]["maxs"], dtype=np.float32)
    ranges = (maxs - mins).astype(np.float32)
    ranges[ranges == 0] = 1.0

    # ---- Import trainer module and build model
    trainer = _import_trainer_module(args.modelAbbrev)
    if not hasattr(trainer, "create_model"):
        print(f"ERROR: {trainer.__name__} must expose create_model(config) -> nn.Module")
        sys.exit(1)

    model = trainer.create_model(config)
    model.load_state_dict(ckpt["model_state_dict"])
    device = torch.device(args.device)
    model.to(device)
    model.eval()

    # ---- Load test stations
    test_ids = _read_ids(args.testList)
    print("Stations for TEST/VAL:")
    for s in test_ids:
        print("  ", s)

    # ---- Load dataframes
    def _load_many(ids):
        dfs = []
        for sid in ids:
            df = _read_station_dataframe(args.dataRoot, sid)
            if df is None or len(df) < 120:
                print(f"Skipping {sid} (no data / too short).")
                continue
            dfs.append(df)
        return dfs

    test_dfs = _load_many(test_ids)
    if not test_dfs:
        print("No valid test data found.")
        sys.exit(1)

    # ---- Dataset / Loader (scale with train mins/maxs from checkpoint)
    test_ds = GSODForecastDataset(test_dfs, mins, maxs)
    test_loader = DataLoader(test_ds, batch_size=args.batchSize, shuffle=False, num_workers=0, pin_memory=True)

    # ---- Inference (collect predictions & truths in REAL units)
    preds_real = []
    trues_real = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating", leave=False):
            xh = batch["hist"].to(device)        # scaled
            xs = batch["static"].to(device)
            y_s = batch["target"].to(device)     # scaled
            y_real = batch["target_real"].numpy()  # already real units from dataset

            yhat_s = model(xh, xs).cpu().numpy()  # scaled predictions
            # inverse transform to real units: y = s*(max-min)+min (per-feature)
            yhat_real = yhat_s * ranges.reshape(1, 1, -1) + mins.reshape(1, 1, -1)

            preds_real.append(yhat_real)
            trues_real.append(y_real)

    y_pred = np.concatenate(preds_real, axis=0)  # (N, 15, 7)
    y_true = np.concatenate(trues_real, axis=0)  # (N, 15, 7)

    # ---- Metrics per requested horizons: Day 1, 7, 15
    horizon_map = {1: 0, 7: 6, 15: 14}
    rows = []
    for day, idx in horizon_map.items():
        for fi, feat in enumerate(FEATURE_COLUMNS):
            y_t = y_true[:, idx, fi]
            y_p = y_pred[:, idx, fi]
            m_mae = float(mae(y_t, y_p))
            m_rmse = rmse(y_t, y_p)
            m_skill = float(skill_score(y_t, y_p))
            rows.append({"Horizon": f"Day {day}", "Feature": feat, "MAE": m_mae, "RMSE": m_rmse, "Skill": m_skill})

    df_out = pd.DataFrame(rows, columns=["Horizon", "Feature", "MAE", "RMSE", "Skill"])

    # ---- Save CSV
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(args.evalDir, f"{args.modelAbbrev}_{timestamp}.csv")
    df_out.to_csv(out_path, index=False)
    print(f"\nSaved evaluation to: {out_path}")

if __name__ == "__main__":
    main()
