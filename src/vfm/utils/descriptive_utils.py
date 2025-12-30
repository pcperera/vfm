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


import numpy as np
import pandas as pd


def get_blocked_temporal_train_val_test_split(
    df: pd.DataFrame,
    well_id_col: str = "well_id",
    time_col: str | None = None,
    n_blocks: int = 5,
    val_fraction_per_block: float = 0.1,
    test_fraction_per_block: float = 0.2,
    min_block_size: int = 3,
):
    """
    Blocked temporal train-validation-test split with preserved temporal order.

    For each well:
    - data are sorted by time
    - split into contiguous temporal blocks
    - within each block:
        early   → train
        middle → validation (optional)
        late    → test (optional)

    Fractions may be zero; corresponding sets will be empty.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    well_id_col : str
        Well identifier column.
    time_col : str | None
        Time column name. If None, index is assumed temporal.
    n_blocks : int
        Number of temporal blocks per well.
    val_fraction_per_block : float
        Fraction of each block reserved for validation.
    test_fraction_per_block : float
        Fraction of each block reserved for testing.
    min_block_size : int
        Minimum samples required per block.

    Returns
    -------
    df_train : pd.DataFrame
    df_val : pd.DataFrame
    df_test : pd.DataFrame
    """

    if not (0.0 <= val_fraction_per_block < 1.0):
        raise ValueError("val_fraction_per_block must be in [0, 1)")
    if not (0.0 <= test_fraction_per_block < 1.0):
        raise ValueError("test_fraction_per_block must be in [0, 1)")
    if val_fraction_per_block + test_fraction_per_block >= 1.0:
        raise ValueError("val_fraction + test_fraction must be < 1")

    train_parts = []
    val_parts = []
    test_parts = []

    for well_id, df_well in df.groupby(well_id_col):

        # --------------------------------------------------
        # Sort temporally
        # --------------------------------------------------
        if time_col:
            df_well = df_well.sort_values(time_col)
        else:
            df_well = df_well.sort_index()

        n = len(df_well)

        # --------------------------------------------------
        # Fallback: insufficient data for blocks
        # --------------------------------------------------
        if n < n_blocks * min_block_size:
            n_test = int(test_fraction_per_block * n)
            n_val = int(val_fraction_per_block * n)

            n_train = n - n_val - n_test
            if n_train < 0:
                n_train = 0

            train_parts.append(df_well.iloc[:n_train])

            if n_val > 0:
                val_parts.append(df_well.iloc[n_train:n_train + n_val])

            if n_test > 0:
                test_parts.append(df_well.iloc[n_train + n_val:])

            continue

        # --------------------------------------------------
        # Create temporal blocks
        # --------------------------------------------------
        block_edges = np.linspace(0, n, n_blocks + 1, dtype=int)

        for i in range(n_blocks):
            start = block_edges[i]
            end = block_edges[i + 1]
            block = df_well.iloc[start:end]

            if len(block) < min_block_size:
                continue

            nb = len(block)
            n_test = int(test_fraction_per_block * nb)
            n_val = int(val_fraction_per_block * nb)
            n_train = nb - n_val - n_test

            if n_train < 0:
                continue

            train_parts.append(block.iloc[:n_train])

            if n_val > 0:
                val_parts.append(block.iloc[n_train:n_train + n_val])

            if n_test > 0:
                test_parts.append(block.iloc[n_train + n_val:])

    df_train = pd.concat(train_parts).sort_index() if train_parts else pd.DataFrame(columns=df.columns)
    df_val = pd.concat(val_parts).sort_index() if val_parts else pd.DataFrame(columns=df.columns)
    df_test = pd.concat(test_parts).sort_index() if test_parts else pd.DataFrame(columns=df.columns)

    return df_train, df_val, df_test


def get_lowo_train_val_test_split(
    df: pd.DataFrame,
    test_well_id: str,
    well_id_col: str = "well_id",
):
    """
    Leave-One-Well-Out split with RANDOMIZED block-wise validation.
    """

    # --------------------------------------------------
    # 1. LOWO split (cross-well generalization)
    # --------------------------------------------------
    df_test_all = df[df[well_id_col] == test_well_id].sort_index()
    df_train_val = df[df[well_id_col] != test_well_id].sort_index()

    df_train, df_val, _ = get_random_train_test_split_per_well_with_order_preserved(df=df_train_val, val_size=0.2, test_size=0)
    df_calibration, _, df_test = get_blocked_temporal_train_val_test_split(df=df_test_all, test_fraction_per_block=0.9, val_fraction_per_block=0)

    return df_train, df_val, df_calibration, df_test


# def get_all_wells() -> list[str]:
#     return  ["W06", "W10"]

def get_all_wells() -> list[str]:
    return  ["W06", "W08", "W10", "W11", "W15", "W18", "W19"]

def get_depdendent_vars():
    return ["qo_well_test", "qg_well_test", "qw_well_test"]

def get_mpfm_vars():
    return ["qo_mpfm", "qg_mpfm", "qw_mpfm", "wc_mpfm"]

def get_independent_vars():
    return ["well_code", "dhp", "dht", "whp", "wht", "choke", "dcp", "gl_mass_rate", "gl_open_ratio"]

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


import pandas as pd

def scores_to_df(scores: dict, model_name: str) -> pd.DataFrame:
    rows = []

    for well, vars_ in scores.items():
        for var, metrics in vars_.items():
            for metric, value in metrics.items():
                rows.append({
                    "well_id": well,
                    "variable": var,
                    "metric": metric,
                    "model": model_name,
                    "value": value,
                })

    return pd.DataFrame(rows)
