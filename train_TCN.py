#!/usr/bin/env python3
"""
TCN trainer.
Maps sequences of shape (features=F, 90) to targets (F, 15).

The script is *imported* by mainTrain.py; do not run directly.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

# --------------------------------------------------------------------------- #
# Hyper-parameters that are *architecture-specific* (not dataset-specific)
# --------------------------------------------------------------------------- #
INPUT_SEQ_LEN = 90
OUTPUT_SEQ_LEN = 15


# --------------------------------------------------------------------------- #
# 1. Data utilities
# --------------------------------------------------------------------------- #
FEATURE_COLUMNS = [
    "TEMP",
    "DEWP",
    "SLP",
    "STP",
    "VISIB",
    "WDSP",
    "MXSPD",
    "MAX",
    "MIN",
    "PRCP",
]  # ← 10 features


def _load_single_station_csv(path: Path) -> pd.DataFrame:
    """Load one CSV, keep only DATE + numeric feature cols, coerce to float,
    replace placeholders with na, and fill na per year for station."""
    df = pd.read_csv(path, low_memory=False)
    df = df[["DATE", *FEATURE_COLUMNS]].copy()
    # Coerce to numeric (invalid parsing -> NaN)
    df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].apply(
        lambda col: pd.to_numeric(col, errors="coerce")
    )

    # Replace placeholder values with NaN
    df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].replace([9999.9, 999.9, 99.9], np.nan)

    # Parse dates and extract year
    df["DATE"] = pd.to_datetime(df["DATE"])
    df["year"] = df["DATE"].dt.year

    # Fill NaNs with mean for that year (station-year)
    for col in FEATURE_COLUMNS:
        df[col] = df.groupby("year")[col].transform(lambda grp: grp.fillna(grp.mean()))

    # Drop helper year column
    df = df.drop(columns=["year"])  # type: ignore

    return df.sort_values("DATE").reset_index(drop=True)


def _load_all_stations(
    data_dir: Path, station_ids: List[str]
) -> pd.DataFrame:  # (rows, 1+F)
    """Concatenate many years for each station → one big DataFrame per station."""
    frames: List[pd.DataFrame] = []
    for sid in station_ids:
        # GSOD files are nested: NOAA_GSOD/2000/01001099999.csv etc.
        yearly_files = sorted(data_dir.glob(f"*/{sid}.csv"))
        if not yearly_files:
            print(f"⚠️   No data found for station {sid}. Skipping.")
            continue
        station_frames = [_load_single_station_csv(p) for p in yearly_files]
        df_station = pd.concat(station_frames, ignore_index=True)
        df_station.insert(0, "STATION", sid)
        frames.append(df_station)
    return pd.concat(frames, ignore_index=True)


class WeatherDataset(Dataset):
    """Sliding-window dataset producing (X, y)."""

    def __init__(self, df: pd.DataFrame):
        # Drop rows with any NaNs in feature columns
        df = df.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)

        # Normalisation stats (global)
        self.mu = torch.tensor(df[FEATURE_COLUMNS].mean().values, dtype=torch.float32)
        self.std = torch.tensor(df[FEATURE_COLUMNS].std().values, dtype=torch.float32)

        feats: Tensor = torch.tensor(
            df[FEATURE_COLUMNS].values, dtype=torch.float32
        )  # (N, F)

        # Standardise
        feats = (feats - self.mu) / (self.std + 1e-8)

        # Build sliding windows
        windows = []
        targets = []
        for start in range(len(feats) - INPUT_SEQ_LEN - OUTPUT_SEQ_LEN):
            end_x = start + INPUT_SEQ_LEN
            end_y = end_x + OUTPUT_SEQ_LEN
            X = feats[start:end_x].T  # (F, 90)
            y = feats[end_x:end_y].T  # (F, 15)
            windows.append(X)
            targets.append(y)

        self.X = torch.stack(windows)  # (N_w, F, 90)
        self.y = torch.stack(targets)  # (N_w, F, 15)

    # -- PyTorch Dataset protocol --
    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        return self.X[idx], self.y[idx]

# --------------------------------------------------------------------------- #
# 2. TCN model
# --------------------------------------------------------------------------- #
class Chomp1d(nn.Module):
    """Removes the last `chomp` elements so causal padding doesn’t leak
    future information."""
    def __init__(self, chomp: int):
        super().__init__()
        self.chomp = chomp

    def forward(self, x: Tensor) -> Tensor:
        return x[..., :-self.chomp] if self.chomp else x


class TemporalBlock(nn.Module):
    """Dilated causal Conv1D → ReLU → Dropout, with residual connection."""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        padding: int,
        dropout: float,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                padding=padding,
                dilation=dilation,
            ),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(
                out_channels,
                out_channels,
                kernel_size,
                padding=padding,
                dilation=dilation,
            ),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.downsample = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        out = self.net(x)
        return nn.functional.relu(out + self.downsample(x))  # residual


class TCN(nn.Module):
    """Stack of TemporalBlocks with exponentially increasing dilation."""
    def __init__(
        self,
        num_inputs: int,
        num_channels: list[int],
        kernel_size: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            in_ch = num_inputs if i == 0 else num_channels[i - 1]
            out_ch = num_channels[i]
            dilation = 2**i
            padding = (kernel_size - 1) * dilation
            layers.append(
                TemporalBlock(
                    in_ch, out_ch, kernel_size, dilation, padding, dropout
                )
            )
        self.network = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, F, L)
        return self.network(x)


class TCNForecast(nn.Module):
    """Forecast the next 15 steps given the previous 90 using a TCN."""
    def __init__(
        self,
        num_features: int,
        hidden_channels: int = 64,
        num_levels: int = 5,  # receptive field ≈ (k−1)·(2^n−1) + 1
        kernel_size: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.tcn = TCN(
            num_inputs=num_features,
            num_channels=[hidden_channels] * num_levels + [num_features],
            kernel_size=kernel_size,
            dropout=dropout,
        )

    def forward(self, x: Tensor) -> Tensor:
        # x is already (B, F, 90) from the dataset.
        y_hat_full = self.tcn(x)            # (B, F, 90)
        return y_hat_full[:, :, -OUTPUT_SEQ_LEN:]  # (B, F, 15)


# --------------------------------------------------------------------------- #
# 3. Training routine
# --------------------------------------------------------------------------- #
def train(
    *,
    data_dir: Path,
    station_list_file: Path,
    epochs: int,
    batch_size: int,
    lr: float,
    save_dir: Path,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Read station IDs
    station_ids = [ln.strip() for ln in station_list_file.read_text().splitlines() if ln]
    df_all = _load_all_stations(data_dir, station_ids)

    dataset = WeatherDataset(df_all)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    model = TCNForecast(num_features=len(FEATURE_COLUMNS)).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        prog = tqdm(loader, desc=f"Epoch {epoch}/{epochs}", leave=False)
        for X, y in prog:
            X, y = X.to(device), y.to(device)

            optimiser.zero_grad()
            y_hat = model(X)
            loss: Tensor = criterion(y_hat, y)
            loss.backward()
            optimiser.step()

            running_loss += loss.item() * X.size(0)
            prog.set_postfix(loss=loss.item())

        epoch_loss = running_loss / len(dataset)
        print(f"Epoch {epoch:02d} | mean MSE: {epoch_loss:.5f}")

    # --------------------------------------------------------------------- #
    # Save
    out_path = save_dir / "weather_TCN.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "mu": dataset.mu,
            "std": dataset.std,
            "input_len": INPUT_SEQ_LEN,
            "output_len": OUTPUT_SEQ_LEN,
            "features": FEATURE_COLUMNS,
        },
        out_path,
    )
    print(f"💾  Model saved to {out_path.resolve()}")