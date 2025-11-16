"""
data_preprocessing.py
Functions to generate synthetic SPE9-like data and prepare features for models.
"""

from typing import Tuple
import os
import numpy as np
import pandas as pd

RNG_SEED = 42

def ensure_dirs():
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)
    os.makedirs("models", exist_ok=True)

def generate_synthetic_spe9(output_path: str = "data/processed/spe9_processed.csv",
                            n_wells: int = 26, time_steps: int = 200, seed: int = RNG_SEED) -> pd.DataFrame:
    """
    Generate a SPE9-like synthetic dataset and save to CSV.
    """
    np.random.seed(seed)
    data_list = []
    for t in range(time_steps):
        field_pressure_trend = 3000 + 150*np.sin(0.04*t) + 0.5*t
        field_flow_trend = 100 + 8*np.sin(0.03*t)
        for w in range(n_wells):
            well_bias = np.random.uniform(-10, 10)
            pressure = field_pressure_trend + 50*np.sin(0.1*w) + np.random.normal(0,20) + well_bias
            flow_rate = max(0.0, field_flow_trend + 3*np.sin(0.2*w) + np.random.normal(0,5) + well_bias*0.1)
            saturation = np.clip(0.2 + 0.05*np.sin(0.07*t + 0.1*w) + np.random.normal(0,0.01), 0, 1)
            permeability = np.random.uniform(50, 800)
            porosity = np.random.uniform(0.05, 0.35)
            data_list.append([t, w, pressure, flow_rate, saturation, permeability, porosity])

    columns = ["Time","Well","Pressure","FlowRate","Saturation","Permeability","Porosity"]
    df = pd.DataFrame(data_list, columns=columns)
    ensure_dirs()
    df.to_csv(output_path, index=False)
    return df

def build_feature_table(df: pd.DataFrame, lags=(1,2,3), rolling_windows=(3,5)) -> pd.DataFrame:
    """
    Given raw df with columns [Time, Well, Pressure, FlowRate, Saturation, Permeability, Porosity],
    return a feature DataFrame with lag and rolling features per (Time, Well).
    """
    records = []
    n_wells = df.Well.nunique()
    times = sorted(df.Time.unique())
    for t in times:
        for w in range(n_wells):
            cur = df[(df.Time==t)&(df.Well==w)]
            if cur.empty:
                continue
            cur = cur.iloc[0]
            row = {"Time": t, "Well": w,
                   "Pressure": cur.Pressure, "FlowRate": cur.FlowRate,
                   "Saturation": cur.Saturation, "Permeability": cur.Permeability, "Porosity": cur.Porosity}
            # lags
            for lag in lags:
                tlag = t - lag
                if tlag >= 0:
                    prev = df[(df.Time==tlag)&(df.Well==w)].iloc[0]
                    row[f"Flow_lag_{lag}"] = prev.FlowRate
                    row[f"Pressure_lag_{lag}"] = prev.Pressure
                else:
                    row[f"Flow_lag_{lag}"] = np.nan
                    row[f"Pressure_lag_{lag}"] = np.nan
            # rolling
            for win in rolling_windows:
                vals = []
                for tt in range(max(0, t-win+1), t+1):
                    r = df[(df.Time==tt)&(df.Well==w)]
                    if not r.empty:
                        vals.append(r.iloc[0].FlowRate)
                row[f"Flow_roll_mean_{win}"] = np.mean(vals) if len(vals)>0 else np.nan
                row[f"Flow_roll_std_{win}"] = np.std(vals) if len(vals)>0 else np.nan
            records.append(row)
    df_feat = pd.DataFrame.from_records(records)
    # forward/backfill then fill remaining with zero
    df_feat.sort_values(["Time","Well"], inplace=True)
    df_feat.fillna(method="bfill", inplace=True)
    df_feat.fillna(0, inplace=True)
    df_feat.reset_index(drop=True, inplace=True)
    df_feat.to_csv("data/processed/spe9_features.csv", index=False)
    return df_feat

if __name__ == "__main__":
    ensure_dirs()
    df = generate_synthetic_spe9()
    df_feat = build_feature_table(df)
    print("Generated synthetic data and features:", df.shape, df_feat.shape)
