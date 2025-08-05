#!/usr/bin/env python3
import argparse
from data_utils import load_station_list, prepare_station_data
from evaluator import run_evaluation
import models.arimax as arimax
# import models.ets as ets  # alternate


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--base-dir', default='NOAA_GSOD')
    p.add_argument('--station-list', default='listToTrain.txt')
    p.add_argument('--iters', type=int, default=1000)
    p.add_argument('--workers', type=int, default=16)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    stations = load_station_list(args.station_list)
    data_map, years_map = {}, {}
    for sid in stations:
        df = prepare_station_data(sid, args.base_dir)
        if df is None: continue
        valid = [y for y,c in df.groupby('YEAR').size().items() if 2000<=y<=2024 and c>=93]
        if valid:
            data_map[sid], years_map[sid] = df, valid

    results = run_evaluation(
        data_map, years_map,
        model_fn=arimax.forecast,
        iterations=args.iters,
        workers=args.workers,
        seed=args.seed
    )

    print("=== Evaluation Results ===")
    for k,v in results.items():
        print(f"{k}: {v:.4f}")

if __name__ == '__main__':
    main()