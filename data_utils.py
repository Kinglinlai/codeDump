import os, glob
import numpy as np
import pandas as pd

def load_station_list(path: str) -> list[str]:
    with open(path) as f:
        return [l.strip() for l in f if l.strip()]


def prepare_station_data(station_id: str, base_dir: str = "NOAA_GSOD") -> pd.DataFrame | None:
    pattern = os.path.join(base_dir, "*", f"{station_id}.csv")
    files = glob.glob(pattern)
    if not files:
        return None

    df = pd.concat((pd.read_csv(fp, parse_dates=["DATE"]) for fp in files))
    df = df.sort_values("DATE").reset_index(drop=True)

    num_cols = ['TEMP','DEWP','SLP','STP','VISIB','WDSP','PRCP']
    for c in num_cols:
        df[c] = (df[c].astype(str).str.strip()
                    .replace({'9999.9':np.nan,'999.9':np.nan,'99.99':np.nan,'':np.nan})
                    .astype(float))

    df['YEAR']        = df['DATE'].dt.year
    df['MONTH']       = df['DATE'].dt.month
    df['DAY_OF_YEAR'] = df['DATE'].dt.dayofyear
    df['WEEK_OF_YEAR']= df['DATE'].dt.isocalendar().week
    df['TEMP_LAG1']   = df['TEMP'].shift(1)

    df = df.dropna(subset=['TEMP'])
    features = ['DEWP','SLP','STP','VISIB','WDSP','PRCP',
                'YEAR','MONTH','DAY_OF_YEAR','WEEK_OF_YEAR','TEMP_LAG1']

    keep = [f for f in features if df[f].isna().mean() <= 0.2]
    return df[keep + ['TEMP']].dropna()