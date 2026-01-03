import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Literal, Tuple

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

def get_random_train_val_test_split_per_well_temporal_order(
    df,
    well_id_col="well_id",
    test_frac=0.2,
    val_frac=0.1,
    min_train=20,
    min_val=5,
    min_test=5,
    random_state=None,
    max_tries=50,
):
    rng = np.random.default_rng(random_state)
    if test_frac == 0:
        min_test = 0

    if val_frac == 0:
        min_val = 0

    for _ in range(max_tries):
        train, val, test = [], [], []

        for _, d in df.groupby(well_id_col):
            r = rng.random(len(d))
            test_mask = r < test_frac
            val_mask = (r >= test_frac) & (r < test_frac + val_frac)
            train_mask = r >= test_frac + val_frac

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


def get_lag_safe_block_split(
    df: pd.DataFrame,
    well_id_col: str = "well_id",
    lags: int = 1,
    test_frac: float = 0.20,
    val_frac: float = 0,
    block_size: int = 10,
    random_state: int | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Lag-safe block-based train/val/test split using DateTimeIndex.

    Supports test_frac = 0 and/or val_frac = 0.
    Assumes df.index is a DateTimeIndex.
    """

    assert isinstance(df.index, pd.DatetimeIndex), \
        "DataFrame index must be a DateTimeIndex"

    assert 0 <= test_frac < 1, "test_frac must be in [0, 1)"
    assert 0 <= val_frac < 1, "val_frac must be in [0, 1)"
    assert test_frac + val_frac < 1, "test_frac + val_frac must be < 1"

    rng = np.random.default_rng(random_state)

    train_parts, val_parts, test_parts = [], [], []

    for wid, d in df.groupby(well_id_col):

        # --------------------------------------------
        # Sort by time (DateTimeIndex)
        # --------------------------------------------
        d = d.sort_index().copy()
        n = len(d)

        if n == 0:
            continue

        # --------------------------------------------
        # Create contiguous temporal blocks
        # --------------------------------------------
        blocks = [
            d.iloc[i:i + block_size]
            for i in range(0, n, block_size)
        ]

        n_blocks = len(blocks)
        if n_blocks == 0:
            continue

        # --------------------------------------------
        # Randomly assign blocks
        # --------------------------------------------
        block_ids = np.arange(n_blocks)
        rng.shuffle(block_ids)

        n_test = int(np.round(test_frac * n_blocks))
        n_val = int(np.round(val_frac * n_blocks))

        test_idx = set(block_ids[:n_test])
        val_idx = set(block_ids[n_test:n_test + n_val])
        train_idx = set(block_ids[n_test + n_val:])

        # --------------------------------------------
        # Collect lag-safe samples
        # --------------------------------------------
        for i, block in enumerate(blocks):

            # Drop first `lags` rows to prevent cross-block leakage
            block_safe = block.iloc[lags:]

            if block_safe.empty:
                continue

            if i in train_idx:
                train_parts.append(block_safe)
            elif i in val_idx:
                val_parts.append(block_safe)
            elif i in test_idx:
                test_parts.append(block_safe)

    # --------------------------------------------
    # Concatenate results (empty-safe)
    # --------------------------------------------
    df_train = (
        pd.concat(train_parts, ignore_index=False)
        if train_parts else df.iloc[0:0]
    )
    df_val = (
        pd.concat(val_parts, ignore_index=False)
        if val_parts else df.iloc[0:0]
    )
    df_test = (
        pd.concat(test_parts, ignore_index=False)
        if test_parts else df.iloc[0:0]
    )

    return df_train, df_val, df_test



def get_temporal_split_per_well(df, test_frac=0.15, val_frac=0.15):
    train, val, test = [], [], []

    assert 0 <= test_frac < 1
    assert 0 <= val_frac < 1
    assert (test_frac + val_frac) < 1, "test_frac + val_frac must be < 1"

    for wid, d in df.groupby("well_id"):
        d = d.sort_index()
        n = len(d)

        n_test = int(n * test_frac)
        n_val = int(n * val_frac)
        n_train = n - n_val - n_test

        train.append(d.iloc[:n_train])
        val.append(d.iloc[n_train:n_train + n_val])
        test.append(d.iloc[n_train + n_val:])

    return (
        pd.concat(train),
        pd.concat(val),
        pd.concat(test),
    )


def get_lowo_train_val_test_split(
    df: pd.DataFrame,
    test_well_id: str,
    well_id_col: str = "well_id",
    split_method: str = Literal["random_temporal", "temporal", "blocked_temporal"]
):
    """
    Leave-One-Well-Out split with RANDOMIZED block-wise validation.
    """

    # --------------------------------------------------
    # 1. LOWO split (cross-well generalization)
    # --------------------------------------------------
    df_test_all = df[df[well_id_col] == test_well_id].sort_index()
    df_train_val = df[df[well_id_col] != test_well_id].sort_index()

    if split_method == "random_temporal":
        df_train, df_val, _ = get_random_train_val_test_split_per_well_temporal_order(df=df_train_val, val_frac=0.2, test_frac=0)
        df_calibration, _, df_test = get_random_train_val_test_split_per_well_temporal_order(df=df_test_all, test_frac=0.9, val_frac=0)
    elif split_method == "temporal":
        df_train, df_val, _ = get_temporal_split_per_well(df=df_train_val, val_frac=0.2, test_frac=0)
        df_calibration, _, df_test = get_temporal_split_per_well(df=df_test_all, test_frac=0.9, val_frac=0)
    elif split_method == "blocked_temporal":
        df_train, df_val, _ = get_lag_safe_block_split(df=df_train_val, val_frac=0.2, test_frac=0)
        df_calibration, _, df_test = get_lag_safe_block_split(df=df_test_all, test_frac=0.9, val_frac=0)
    else:
        raise ValueError("Invalid split_method")

    return df_train, df_val, df_calibration, df_test



# def get_all_wells() -> list[str]:
#     return  ["W06", "W10", "W11", "W15", "W18", "W19"]

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

def get_feature_vars():
    return ["pres_drop", "temp_drop"]

def get_well_test_ratios():
    return ["gor_well_test", "wgr_well_test"]

def get_all_vars():
    all_vars = []
    all_vars.extend(get_depdendent_vars())
    all_vars.extend(get_independent_vars())
    all_vars.extend(get_mpfm_vars())
    all_vars.extend(get_mpfm_ratios())
    all_vars.extend(get_well_test_ratios())
    all_vars.extend(get_feature_vars())
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

