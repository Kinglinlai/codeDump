import random
import numpy as np
from concurrent.futures import ThreadPoolExecutor

from metrics import mae, rmse, mase, skill_score


def evaluate_split(df, valid_years, model_fn, horizon=3):
    # pick year, random 90-day train + test
    year = random.choice(valid_years)
    dfy = df[df['YEAR']==year].reset_index(drop=True)
    if len(dfy) < 90 + horizon: return None

    start = random.randint(0, len(dfy)-(90+horizon))
    tr, te = dfy.iloc[start:start+90], dfy.iloc[start+90:start+90+horizon]

    y_train, y_test = tr['TEMP'].values, te['TEMP'].values
    ex_cols = [c for c in df.columns if c!='TEMP']
    X_train, X_test = tr[ex_cols].values, te[ex_cols].values

    y_pred = model_fn(y_train, X_train, X_test, horizon)
    y_naive = np.repeat(y_train[-1], horizon)

    return {
        'MAE'  : mae(y_test, y_pred),
        'RMSE' : rmse(y_test, y_pred),
        'MASE' : mase(y_test, y_pred, y_train),
        'SKILL': skill_score(y_test, y_pred, y_naive)
    }


def run_evaluation(data_map, years_map, model_fn, iterations, workers, seed):
    random.seed(seed)
    metrics = {k: [] for k in ['MAE','RMSE','MASE','SKILL']}
    with ThreadPoolExecutor(max_workers=workers) as exe:
        futures = [exe.submit(evaluate_split, data_map[s], years_map[s], model_fn)
                   for _ in range(iterations)
                   for s in [random.choice(list(data_map.keys()))]]
        for f in futures:
            res = f.result()
            if res:
                for k in metrics: metrics[k].append(res[k])
    # aggregate
    return {k: np.mean(v) for k,v in metrics.items() if v}