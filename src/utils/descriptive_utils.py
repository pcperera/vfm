import pandas as pd
import numpy as np
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

# def get_train_test_split_per_well(
#     df: pd.DataFrame,
#     well_id_col: str = "well_id",
#     test_size: float = 0.2,
#     min_train_points: int = 30,
# ):
#     """
#     Per-well temporal train/test split WITHOUT lagged features.

#     Assumes df is already sorted by time_idx.
#     """

#     if not 0.0 < test_size < 1.0:
#         raise ValueError("test_size must be between 0 and 1")

#     train_parts = []
#     test_parts = []

#     for well_id, df_well in df.groupby(well_id_col, sort=False):
#         n = len(df_well)

#         # Not enough data → skip well
#         if n < min_train_points + 1:
#             continue

#         split_idx = int((1 - test_size) * n)

#         # Enforce minimum train size
#         if split_idx < min_train_points:
#             split_idx = min_train_points

#         df_train = df_well.iloc[:split_idx]
#         df_test = df_well.iloc[split_idx:]

#         if len(df_test) == 0:
#             continue

#         train_parts.append(df_train)
#         test_parts.append(df_test)

#     df_train = pd.concat(train_parts)
#     df_test = pd.concat(test_parts)

#     return df_train, df_test

def get_random_train_test_split_per_well_with_order_preserved(
    df,
    well_id_col="well_id",
    test_size=0.2,
    val_size=0.1,
    min_train=20,
    min_val=5,
    min_test=5,
    random_state=None,
    max_tries=50,
):
    rng = np.random.default_rng(random_state)
    if test_size == 0:
        min_test = 0

    for _ in range(max_tries):
        train, val, test = [], [], []

        for _, d in df.groupby(well_id_col):
            r = rng.random(len(d))
            test_mask = r < test_size
            val_mask = (r >= test_size) & (r < test_size + val_size)
            train_mask = r >= test_size + val_size

            if (
                train_mask.sum() < min_train or
                val_mask.sum() < min_val or
                test_mask.sum() < min_test
            ):
                break

            train.append(d.loc[train_mask])
            val.append(d.loc[val_mask])
            test.append(d.loc[test_mask])
        else:
            return (
                pd.concat(train).sort_index(),
                pd.concat(val).sort_index(),
                pd.concat(test).sort_index(),
            )

    raise RuntimeError("Could not satisfy split constraints")

def get_lowo_train_val_test_split(
    df: pd.DataFrame,
    test_well_id: str,
    well_id_col: str = "well_id",
    random_state: int | None = None,
):
    """
    Leave-One-Well-Out split with RANDOMIZED block-wise validation.
    """

    # --------------------------------------------------
    # 1. LOWO split (cross-well generalization)
    # --------------------------------------------------
    df_test = df[df[well_id_col] == test_well_id].sort_index()
    df_train_val = df[df[well_id_col] != test_well_id].sort_index()

    df_train, df_val, _ = get_random_train_test_split_per_well_with_order_preserved(df=df_train_val, val_size=0.2, test_size=0)
    return df_train, df_val, df_test


# def get_train_val_test_split_per_well(
#     df: pd.DataFrame,
#     well_id_col: str = "well_id",
#     train_frac: float = 0.7,
#     val_frac: float = 0.1,
#     test_frac: float = 0.2,
#     block_size: int = 10,
#     min_train: int = 20,
#     min_val: int = 5,
#     min_test: int = 4,
#     random_state: int | None = None,
# ):
#     """
#     Block-wise train/validation/test split per well with
#     approximate fraction control and future-like test set.
#     """

#     assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6

#     rng = np.random.default_rng(random_state)

#     train_all, val_all, test_all = [], [], []

#     for well_id, d in df.groupby(well_id_col):
#         d = d.sort_index()
#         n = len(d)

#         if n < (min_train + min_val + min_test):
#             raise ValueError("Not enough data points for requested split")

#         # --------------------------------------------------
#         # Split into contiguous blocks
#         # --------------------------------------------------
#         blocks = [
#             d.iloc[i:i + block_size]
#             for i in range(0, n, block_size)
#         ]

#         n_blocks = len(blocks)
#         if n_blocks < 5:
#             raise ValueError("Too few blocks for stable splitting")

#         # --------------------------------------------------
#         # Number of blocks per split
#         # --------------------------------------------------
#         n_test_blocks = max(1, int(test_frac * n_blocks))
#         n_val_blocks = max(1, int(val_frac * n_blocks))
#         n_train_blocks = n_blocks - n_test_blocks - n_val_blocks

#         if n_train_blocks < 1:
#             raise ValueError("Invalid block allocation")

#         # --------------------------------------------------
#         # Assign TEST as last contiguous blocks (future-like)
#         # --------------------------------------------------
#         test_blocks = blocks[-n_test_blocks:]
#         remaining_blocks = blocks[:-n_test_blocks]

#         # --------------------------------------------------
#         # Randomize TRAIN / VAL blocks
#         # --------------------------------------------------
#         rng.shuffle(remaining_blocks)

#         train_blocks = remaining_blocks[:n_train_blocks]
#         val_blocks = remaining_blocks[n_train_blocks:]

#         # --------------------------------------------------
#         # Concatenate and validate sizes
#         # --------------------------------------------------
#         train_df = pd.concat(train_blocks)
#         val_df = pd.concat(val_blocks)
#         test_df = pd.concat(test_blocks)

#         if (
#             len(train_df) < min_train or
#             len(val_df) < min_val or
#             len(test_df) < min_test
#         ):
#             raise ValueError("Split violates minimum size constraints")

#         train_all.append(train_df)
#         val_all.append(val_df)
#         test_all.append(test_df)

#     return (
#         pd.concat(train_all).sort_index(),
#         pd.concat(val_all).sort_index(),
#         pd.concat(test_all).sort_index(),
#     )


# def get_all_wells() -> list[str]:
#     return  ["W06", "W10"]

def get_all_wells() -> list[str]:
    return  ["W06", "W08", "W10", "W11", "W15", "W18", "W19"]

def get_depdendent_vars():
    return ["qo_well_test", "qg_well_test", "qw_well_test"]

def get_mpfm_vars():
    return ["qo_mpfm", "qg_mpfm", "qw_mpfm", "wc_mpfm"]

def get_independent_vars():
    return ["well_code", "dhp", "dht", "whp", "wht", "choke", "dcp"]

def get_independent_tp_vars():
    return ["dhp", "dht", "whp", "wht", "dcp"]

def get_independent_vars_with_no_well_code():
    vars = get_independent_vars()
    vars.remove("well_code")
    return vars

def get_mpfm_ratios():
    return ["gor_mpfm", "wgr_mpfm"]

def get_well_test_ratios():
    return ["gor_well_test", "wgr_well_test"]

def get_all_vars():
    all_vars = []
    all_vars.extend(get_depdendent_vars())
    all_vars.extend(get_independent_vars())
    all_vars.extend(get_mpfm_vars())
    all_vars.extend(get_mpfm_ratios())
    all_vars.extend(get_well_test_ratios())
    all_vars.append("well_id")

    return all_vars