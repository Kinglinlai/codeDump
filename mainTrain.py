#!/usr/bin/env python3
"""
Entry-point script for training weather-forecast models.
Usage:
    python mainTrain.py --model LSTM --epochs 20 --batch_size 128
"""
from importlib import import_module
from pathlib import Path
import argparse
from typing import Callable

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
THIS_DIR = Path(__file__).resolve().parent
MODELS_DIR = THIS_DIR / "models"
DATA_DIR = THIS_DIR / "NOAA_GSOD"
LIST_FILE = THIS_DIR / "listToTrain.txt"

# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a weather-forecast model from NOAA GSOD data."
    )
    parser.add_argument(
        "--model",
        default="LSTM",
        choices=["LSTM", "TCN", "CNN","GRU"],  # add more here as you implement them
        help="Which model architecture to train (default: LSTM)",
    )
    parser.add_argument("--epochs", type=int, default=15, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    return parser.parse_args()


# --------------------------------------------------------------------------- #
# Dynamic dispatch to train_{model}.py
# --------------------------------------------------------------------------- #
def main() -> None:
    args = _parse_args()
    module_name = f"train_{args.model}"
    try:
        mod = import_module(module_name)
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise SystemExit(
            f"❌   Could not find trainer '{module_name}.py' in the current directory."
        ) from exc

    # Every train_{model}.py must expose train(...) with this signature
    train_fn: Callable[..., None] = getattr(mod, "train", None)
    if train_fn is None:
        raise SystemExit(f"❌   '{module_name}.py' lacks a 'train()' function.")

    MODELS_DIR.mkdir(exist_ok=True)
    print(
        f"🚀  Starting training: {args.model} "
        f"(epochs={args.epochs}, batch_size={args.batch_size})"
    )
    train_fn(
        data_dir=DATA_DIR,
        station_list_file=LIST_FILE,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        save_dir=MODELS_DIR,
    )
    print("✅  Training complete!")


if __name__ == "__main__":
    main()
