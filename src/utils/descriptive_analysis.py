import pandas as pd

def summarize_na(df: pd.DataFrame) -> pd.Series:
    # Count NaNs per column
    nan_counts = df[df.columns.to_list()].isna().sum()
    return nan_counts

def print_values(df: pd.DataFrame, column: str) -> pd.Series:
    non_na_dhp = df.loc[df[column].notna(), column]
    for item in non_na_dhp.items():
        print(item)