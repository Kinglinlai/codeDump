#!/usr/bin/env python3
"""
Evaluate a trained weather-forecast model on unseen stations.

Assumes:
• checkpoints are in ./models and were saved by train_*.py
• listToTest.txt contains one station ID per line
• NOAA_GSOD directory layout is identical to training

Outputs a per-feature table with MAE, RMSE and skill score.
"""
from __future__ import annotations

import argparse
import math
import sys
from importlib import import_module
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

# --------------------------------------------------------------------------- #
# ------------------------------ CLI ---------------------------------------- #
# --------------------------------------------------------------------------- #
DEF_MODEL = "weather_LSTM.pt"
THIS_DIR = Path(__file__).resolve().parent
MODELS_DIR = THIS_DIR / "models"
DATA_DIR = THIS_DIR / "NOAA_GSOD"
TEST_LIST_FILE = THIS_DIR / "listToTest.txt"

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a GSOD weather model.")
    p.add_argument("--model", default=DEF_MODEL,
                   help="Checkpoint filename inside ./models/ (default: weather_LSTM.pt)")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--device", choices=["cuda", "cpu"], default=None,
                   help="Force device (default: auto-select)")
    p.add_argument("--list_file", type=Path, default=TEST_LIST_FILE,
                   help="Path to list of station IDs for evaluation")
    return p.parse_args()

# --------------------------------------------------------------------------- #
# ------------------------------ Data utils --------------------------------- #
# --------------------------------------------------------------------------- #
FEATURE_COLUMNS = [
    "TEMP", "DEWP", "SLP", "STP", "VISIB",
    "WDSP", "MXSPD", "MAX", "MIN", "PRCP",
]

INPUT_SEQ_LEN  = 90     # must match training
OUTPUT_SEQ_LEN = 15

def _load_single_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    keep = ["DATE", *FEATURE_COLUMNS]
    df = df[keep].copy()
    df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].apply(
        lambda col: pd.to_numeric(col, errors="coerce")
    )
    df["DATE"] = pd.to_datetime(df["DATE"])
    return df.sort_values("DATE").reset_index(drop=True)

def _concat_station_years(data_dir: Path, sid: str) -> pd.DataFrame:
    yearly_files = sorted(data_dir.glob(f"*/{sid}.csv"))
    if not yearly_files:
        print(f"⚠️  No data found for test station {sid}", file=sys.stderr)
        return pd.DataFrame()
    frames = [_load_single_csv(f) for f in yearly_files]
    return pd.concat(frames, ignore_index=True)

def _load_test_stations(data_dir: Path, ids: List[str]) -> pd.DataFrame:
    frames = []
    for sid in ids:
        df = _concat_station_years(data_dir, sid)
        if not df.empty:
            df.insert(0, "STATION", sid)
            frames.append(df)
    if not frames:
        raise SystemExit("❌  No test data available!")
    return pd.concat(frames, ignore_index=True)

class WeatherDataset(Dataset):
    """Raw (un-normalised) windows for evaluation."""
    def __init__(self, df: pd.DataFrame):
        df = df.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)
        feats = torch.tensor(df[FEATURE_COLUMNS].values, dtype=torch.float32)  # (N,F)

        Xs, ys = [], []
        for start in range(len(feats) - INPUT_SEQ_LEN - OUTPUT_SEQ_LEN):
            end_x = start + INPUT_SEQ_LEN
            end_y = end_x + OUTPUT_SEQ_LEN
            Xs.append(feats[start:end_x].T)   # (F, 90)
            ys.append(feats[end_x:end_y].T)   # (F, 15)
        self.X = torch.stack(Xs)  # (N, F, 90)
        self.y = torch.stack(ys)  # (N, F, 15)

    def __len__(self) -> int:        return self.X.shape[0]
    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        return self.X[idx], self.y[idx]

# --------------------------------------------------------------------------- #
# --------------------- Metric aggregation helpers -------------------------- #
# --------------------------------------------------------------------------- #
class MetricTracker:
    """
    Aggregates MAE, RMSE, and Skill for specific forecast horizons.
    horizons = [0, 6, 14]  ⇒ day-1, day-7, day-15
    """
    def __init__(self, num_feat: int, horizons=(0, 6, 14)) -> None:
        self.horizons = list(horizons)
        n_h = len(self.horizons)
        self.abs_err   = torch.zeros((n_h, num_feat))
        self.sq_err    = torch.zeros((n_h, num_feat))
        self.sq_err_p  = torch.zeros((n_h, num_feat))  # persistence baseline
        self.count     = torch.zeros(n_h, dtype=torch.long)

    def update(self, y: Tensor, pred: Tensor, base: Tensor) -> None:
        """
        y, pred, base: (B, F, 15) tensors in *raw* units
        """
        B, F, _ = y.shape
        for i_h, h in enumerate(self.horizons):
            self.abs_err[i_h]  += (pred[:, :, h] - y[:, :, h]).abs().sum(dim=0).cpu()
            self.sq_err[i_h]   += (pred[:, :, h] - y[:, :, h]).pow(2).sum(dim=0).cpu()
            self.sq_err_p[i_h] += (base[:, :, h]  - y[:, :, h]).pow(2).sum(dim=0).cpu()
            self.count[i_h]    += B

    # ---------- tabular summary ---------------------------------------- #
    def summary(self) -> pd.DataFrame:
        mae   = self.abs_err  / self.count[:, None]
        rmse  = torch.sqrt(self.sq_err / self.count[:, None])
        skill = 1 - (self.sq_err / self.count[:, None]) / (self.sq_err_p / self.count[:, None])

        # build tidy table
        rows = []
        for i_h, h in enumerate(self.horizons):
            for feat_idx, feat in enumerate(FEATURE_COLUMNS):
                rows.append(
                    {
                        "Horizon": f"Day {h+1}",
                        "Feature": feat,
                        "MAE":   mae[i_h, feat_idx].item(),
                        "RMSE":  rmse[i_h, feat_idx].item(),
                        "Skill": skill[i_h, feat_idx].item(),
                    }
                )
        df = pd.DataFrame(rows).set_index(["Horizon", "Feature"])
        return df

# --------------------------------------------------------------------------- #
# ------------------------------ Arch lookup -------------------------------- #
# --------------------------------------------------------------------------- #
def _get_arch_class(arch_name: str):
    """Import correct class from train_{arch}.py."""
    mod_name = f"train_{arch_name}"
    try:
        mod = import_module(mod_name)
    except ModuleNotFoundError:
        raise SystemExit(f"❌  Cannot import architecture '{arch_name}'. "
                         "Make sure train_{arch}.py exists.")
    cls = getattr(mod, f"{arch_name}Forecast", None)
    if cls is None:
        raise SystemExit(f"❌  train_{arch_name}.py lacks {arch_name}Forecast class.")
    return cls

# --------------------------------------------------------------------------- #
# ------------------------------ Main eval ---------------------------------- #
# --------------------------------------------------------------------------- #
def main() -> None:
    args = _parse_args()

    ckpt_path = MODELS_DIR / args.model
    if not ckpt_path.exists():
        raise SystemExit(f"❌  {ckpt_path} not found.")

    device = (
        torch.device("cuda")
        if (args.device != "cpu" and torch.cuda.is_available())
        else torch.device("cpu")
    )

    ckpt: Dict = torch.load(ckpt_path, map_location="cpu")

    # --------- infer architecture from filename or ckpt -------------
    if "LSTM" in ckpt_path.stem.upper():
        arch = "LSTM"
    elif "TCN" in ckpt_path.stem.upper():
        arch = "TCN"
    elif "CNN" in ckpt_path.stem.upper():
        arch = "CNN"
    elif "GRU" in ckpt_path.stem.upper():
        arch = "GRU"
    else:
        arch = ckpt.get("arch", "LSTM")   # fallback

    ArchClass = _get_arch_class(arch)
    model = ArchClass(num_features=len(FEATURE_COLUMNS))
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    mu  = ckpt["mu"].to(device)          # (F,)
    std = ckpt["std"].to(device)

    # -------------------- build test dataset ------------------------
    station_ids = [ln.strip() for ln in args.list_file.read_text().splitlines() if ln]
    df_test = _load_test_stations(DATA_DIR, station_ids)
    dataset = WeatherDataset(df_test)
    loader  = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    tracker = MetricTracker(num_feat=len(FEATURE_COLUMNS), horizons=(0, 6, 14))

    with torch.no_grad():
        for X_raw, y_raw in tqdm(loader, desc="Evaluating"):
            X_raw = X_raw.to(device)    # (B,F,90)
            y_raw = y_raw.to(device)    # (B,F,15)

            # --- standardise inputs ---
            X_std = (X_raw - mu[:, None]) / (std[:, None] + 1e-8)
            pred_std = model(X_std)          # (B,F,15) in z-space
            pred_raw = pred_std * (std[:, None] + 1e-8) + mu[:, None]

            # baseline = persistence (repeat last obs)
            baseline = X_raw[:, :, -1:].repeat(1, 1, OUTPUT_SEQ_LEN)

            tracker.update(y_raw, pred_raw, baseline)

    df = tracker.summary()
    print("\nPer-feature metrics at key horizons\n")
    print(df.to_string(float_format=lambda x: f"{x:8.4f}"))

if __name__ == "__main__":
    main()
