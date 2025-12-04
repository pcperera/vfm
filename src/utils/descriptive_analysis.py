import pandas as pd

def summarize_na(df: pd.DataFrame) -> pd.Series:
    # Count NaNs per column
    nan_counts = df[df.columns.to_list()].isna().sum()
    return nan_counts