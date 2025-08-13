# evalVis.py
import argparse
import os
import re
from glob import glob
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


FNAME_RE = re.compile(r"^(?P<model>[A-Za-z0-9\-]+)_(?P<ts>\d{8}_\d{6})\.csv$")

def find_latest_per_model(eval_dir):
    files = glob(os.path.join(eval_dir, "*.csv"))
    latest = {}
    picked = {}
    for fp in files:
        base = os.path.basename(fp)
        m = FNAME_RE.match(base)
        if not m:
            continue
        model = m.group("model")
        ts_str = m.group("ts")
        try:
            ts = datetime.strptime(ts_str, "%Y%m%d_%H%M%S")
        except ValueError:
            continue
        if (model not in latest) or (ts > latest[model]):
            latest[model] = ts
            picked[model] = fp
    return picked  # {model: filepath}

def tidy_eval_df(df):
    # Expect columns: Horizon, Feature, MAE, RMSE, Skill
    # Normalize types
    df = df.copy()
    df["Horizon"] = df["Horizon"].astype(str)
    df["Feature"] = df["Feature"].astype(str)
    for col in ["MAE", "RMSE", "Skill"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    # Sort horizons by numeric day if present (e.g., "Day 1")
    def horizon_key(h):
        m = re.search(r"(\d+)", str(h))
        return int(m.group(1)) if m else 999
    df["HorizonOrder"] = df["Horizon"].map(horizon_key)
    df = df.sort_values(["HorizonOrder", "Feature"]).drop(columns=["HorizonOrder"])
    return df

def aggregate_latest(latest_map):
    # Returns:
    #   long_df: rows for all models (Horizon, Feature, metrics)
    #   per_model_hz: avg across features per horizon
    long_rows = []
    for model, path in latest_map.items():
        df = pd.read_csv(path)
        df = tidy_eval_df(df)
        df["Model"] = model
        long_rows.append(df)
    if not long_rows:
        return None, None
    long_df = pd.concat(long_rows, ignore_index=True)

    # avg across features per horizon
    per_model_hz = (
        long_df
        .groupby(["Model", "Horizon"], as_index=False)[["MAE", "RMSE", "Skill"]]
        .mean()
        .sort_values(["Horizon", "Model"])
    )
    return long_df, per_model_hz

def compute_diffs(per_model_hz, long_df):
    # Diff from model-mean at each horizon (AvgRMSE level)
    hz_mean = (
        per_model_hz.groupby("Horizon", as_index=False)["RMSE"].mean()
        .rename(columns={"RMSE": "AvgRMSE_AllModels"})
    )
    merged = per_model_hz.merge(hz_mean, on="Horizon", how="left")
    merged["DiffFromAvgRMSE"] = merged["RMSE"] - merged["AvgRMSE_AllModels"]

    # Optional: per-feature diffs (RMSE)
    per_feat = (
        long_df.groupby(["Horizon", "Feature", "Model"], as_index=False)[["RMSE"]]
        .mean()
    )
    per_feat_mean = (
        per_feat.groupby(["Horizon", "Feature"], as_index=False)["RMSE"]
        .mean()
        .rename(columns={"RMSE": "MeanRMSE_AllModels"})
    )
    per_feat = per_feat.merge(per_feat_mean, on=["Horizon", "Feature"], how="left")
    per_feat["DiffFromAvgRMSE"] = per_feat["RMSE"] - per_feat["MeanRMSE_AllModels"]

    return merged, per_feat

def save_tables(out_dir, latest_map, per_model_hz, diff_hz, per_feat_diff):
    os.makedirs(out_dir, exist_ok=True)
    # Selected files
    pd.DataFrame(
        [{"Model": m, "File": p} for m, p in sorted(latest_map.items())]
    ).to_csv(os.path.join(out_dir, "latest_files.csv"), index=False)

    per_model_hz.to_csv(os.path.join(out_dir, "summary_per_model_horizon.csv"), index=False)
    diff_hz.to_csv(os.path.join(out_dir, "diff_from_avg_rmse_per_horizon.csv"), index=False)
    per_feat_diff.to_csv(os.path.join(out_dir, "diff_from_avg_rmse_per_feature.csv"), index=False)

def plot_bars(per_model_hz, out_dir):
    # One bar chart per horizon: Avg RMSE per model
    horizons = list(per_model_hz["Horizon"].unique())
    for hz in horizons:
        sub = per_model_hz[per_model_hz["Horizon"] == hz].copy()
        sub = sub.sort_values("RMSE", ascending=True)
        plt.figure(figsize=(8, max(3, 0.4 * len(sub))))
        plt.barh(sub["Model"], sub["RMSE"])
        plt.xlabel("Average RMSE (lower is better)")
        plt.title(f"Avg RMSE by Model — {hz}")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"bar_rmse_{hz.replace(' ', '')}.png"), dpi=150)
        plt.close()

def plot_heatmap(diff_hz, out_dir):
    # Heatmap of DiffFromAvgRMSE (rows models, cols horizons)
    pivot = diff_hz.pivot_table(index="Model", columns="Horizon", values="DiffFromAvgRMSE", aggfunc="mean")
    models = list(pivot.index)
    horizons = list(pivot.columns)

    if pivot.empty:
        return

    data = pivot.values
    plt.figure(figsize=(1.8 + 1.2 * len(horizons), 1.8 + 0.5 * len(models)))
    im = plt.imshow(data, aspect="auto")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(ticks=np.arange(len(horizons)), labels=horizons, rotation=0)
    plt.yticks(ticks=np.arange(len(models)), labels=models)
    plt.title("RMSE − Average(RMSE across models) by Horizon")
    # annotate
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            plt.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "heatmap_diff_from_avg_rmse.png"), dpi=150)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Visualize and compare latest evaluations for all models.")
    parser.add_argument("--evalDir", type=str, default=".\\eval", help="Directory with evaluation CSVs")
    parser.add_argument("--outDir", type=str, default=".\\eval\\vis", help="Output directory for tables/plots")
    args = parser.parse_args()

    latest_map = find_latest_per_model(args.evalDir)
    if not latest_map:
        print(f"No matching CSVs found in {args.evalDir}. Expected pattern: <MODEL>_YYYYmmdd_HHMMSS.csv")
        return

    print("Using latest evaluations:")
    for m, p in sorted(latest_map.items()):
        print(f"  {m}: {p}")

    long_df, per_model_hz = aggregate_latest(latest_map)
    if long_df is None:
        print("No evaluation rows found.")
        return

    diff_hz, per_feat_diff = compute_diffs(per_model_hz, long_df)

    os.makedirs(args.outDir, exist_ok=True)
    save_tables(args.outDir, latest_map, per_model_hz, diff_hz, per_feat_diff)

    # Plots
    plot_bars(per_model_hz, args.outDir)
    plot_heatmap(diff_hz, args.outDir)

    print(f"\nSaved summaries and figures to: {args.outDir}")
    print("Files:")
    print("  - latest_files.csv")
    print("  - summary_per_model_horizon.csv")
    print("  - diff_from_avg_rmse_per_horizon.csv")
    print("  - diff_from_avg_rmse_per_feature.csv")
    print("  - bar_rmse_Day1.png, bar_rmse_Day7.png, bar_rmse_Day15.png (as available)")
    print("  - heatmap_diff_from_avg_rmse.png")

if __name__ == "__main__":
    main()
