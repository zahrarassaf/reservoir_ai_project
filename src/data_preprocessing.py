import pandas as pd
import numpy as np

def load_dataset(path):
    return pd.read_csv(path)

def preprocess(df):
    # Example preprocessing: fill NaN, normalize
    df = df.fillna(0)
    df_norm = (df - df.min()) / (df.max() - df.min())
    return df_norm

if __name__ == "__main__":
    df = load_dataset("../data/processed/spe9_processed.csv")
    df_pre = preprocess(df)
    df_pre.to_csv("../data/processed/spe9_preprocessed.csv", index=False)
