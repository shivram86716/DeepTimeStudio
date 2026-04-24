import pandas as pd
from dateutil import parser


def load_time_series(file_obj, date_col: str, value_col: str) -> pd.DataFrame:
    df = pd.read_csv(file_obj)

    if date_col not in df.columns:
        raise ValueError(f"Date column '{date_col}' not found in file.")
    if value_col not in df.columns:
        raise ValueError(f"Value column '{value_col}' not found in file.")

    # Parse dates
    df[date_col] = df[date_col].apply(parser.parse)

    # Sort by date
    df = df.sort_values(by=date_col).reset_index(drop=True)

    # Ensure numeric
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df = df.dropna(subset=[value_col])

    return df[[date_col, value_col]]
