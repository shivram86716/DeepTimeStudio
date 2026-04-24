import pandas as pd
import numpy as np


def detect_anomalies(df: pd.DataFrame, value_col: str, z_threshold: float = 2.5) -> pd.DataFrame:
    values = df[value_col].astype(float)
    mean = values.mean()
    std = values.std(ddof=0)

    if std == 0:
        return pd.DataFrame(columns=df.columns)

    z_scores = (values - mean) / std
    mask = np.abs(z_scores) >= z_threshold

    anomalies_df = df.loc[mask].copy()
    anomalies_df["z_score"] = z_scores[mask]

    return anomalies_df
