import pandas as pd
from sklearn.model_selection import train_test_split

def summarize_null(df: pd.DataFrame) -> pd.Series:
    # Count NaNs per column
    nan_counts = df[df.columns.to_list()].isna().sum()
    return nan_counts

def print_values(df: pd.DataFrame, column: str) -> pd.Series:
    non_na_dhp = df.loc[df[column].notna(), column]
    for item in non_na_dhp.items():
        print(item)

def get_train_test_split(df: pd.DataFrame):
    return train_test_split(
        df,
        test_size=0.2,      # 20% test
        shuffle=False
    )

def get_train_test_split_per_well(
    df: pd.DataFrame,
    well_id_col: str = "well_id",
    test_size: float = 0.2,
):
    """
    Split dataframe into train/test sets with the same proportion
    for each well, assuming df is already sorted by time_idx.

    Returns
    -------
    df_train, df_test
    """

    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be between 0 and 1")

    train_parts = []
    test_parts = []

    for well_id, df_well in df.groupby(well_id_col, sort=False):
        n = len(df_well)
        if n < 2:
            continue  # or raise, depending on your policy

        split_idx = int((1 - test_size) * n)

        train_parts.append(df_well.iloc[:split_idx])
        test_parts.append(df_well.iloc[split_idx:])

    df_train = pd.concat(train_parts)
    df_test = pd.concat(test_parts)

    return df_train, df_test
