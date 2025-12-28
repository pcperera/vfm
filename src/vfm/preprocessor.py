import pandas as pd
from src.utils.descriptive_utils import *


class Preprocessor:

    def __init__(self):
        # self._time_idx_resolution_min = 1
        self._independent_tp_vars = get_independent_tp_vars()
        self._independent_vars = get_independent_vars_with_no_well_code()

    def _preprocess_well(self, df: pd.DataFrame) -> pd.DataFrame:
        well_ids = df["well_id"].unique()
        assert len(well_ids) == 1, "DataFrame contains multiple well IDs."
        well_id = well_ids[0]

        # Data transformation
        df.sort_index(inplace=True)  # Ensure chronological order

        df = df[df["dhp"] != 0]
        df = df[df["whp"] != 0]
        df = df[df["dcp"] != 0]
        df = df[df["dht"] != 0]
        df = df[df["wht"] != 0]
        df = df[df["choke"] != 0]

        df["well_id"] = well_id
        df["well_code"] = well_id

        # Convert choke to fraction
        df["choke"] /= 100

        # Calculate MPFM flow rates from GOR and WC
        df["gor_mpfm"] = df["qg_mpfm"] / df["qo_mpfm"]
        df["wc_mpfm_ratio"] = df["wc_mpfm"] / 100
        df["qw_mpfm"] = (df["wc_mpfm_ratio"] * df["qo_mpfm"]) / (1 - df["wc_mpfm_ratio"])

        df.dropna(subset=get_depdendent_vars(), inplace=True)
        df.dropna(subset=get_independent_vars_with_no_well_code(), inplace=True)

        df = df[get_all_vars()]
        df = df[(df["qo_mpfm"] >= 0) & (df["qg_mpfm"] >= 0) & (df["qw_mpfm"] >= 0)]

        # Drop rows where oil and gas rates are both zero.
        df = df[~((df["qo_mpfm"] == 0) & (df["qg_mpfm"] == 0))]

        return df    


    def preprocess_wells(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy(deep=True)
        dfs = []
        well_ids = df["well_id"].unique()

        # # Calculate the time steps
        reference_time = df.index.min()

        for well_id in well_ids:
            print(f"Preprocessing well {well_id} with columns {df.columns.tolist()}")
            df_well = df[df["well_id"] == well_id]
            df_well = self._preprocess_well(df=df_well)
            df_well["time_idx"] = (df_well.index - reference_time).total_seconds() // 30 # Unit is 30s
            df_well["time_idx"] = df_well["time_idx"].astype(int)
            dfs.append(df_well)

        df = pd.concat(dfs, ignore_index=False)
        df["well_code"] = (
            df["well_id"]
                .astype("category")
                .cat.codes
                .astype(float)   # ML-friendly
            )

        return df   
    
