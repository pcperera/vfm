import pandas as pd
from src.vfm.utils.descriptive_utils import *


class Preprocessor:

    def __init__(self):
        # self._time_idx_resolution_min = 1
        self._independent_tp_vars = get_independent_tp_vars()
        self._independent_vars = get_independent_vars_with_no_well_code()
        self._well_col = "well_id"

    def _process_zero_water_rates(self, df):
        well_ids = df[self._well_col].unique()
        assert len(well_ids) == 1, "DataFrame contains multiple well IDs."
        well_id = well_ids[0]
        qw_col="qw_well_test"

        pos = df[df[qw_col] > 0]
        if pos.empty:
            return df

        qw_min = pos[qw_col].min()
        t_wb = pos.index.min()

        mask = (df[self._well_col] == well_id) & (df.index >= t_wb) & (df[qw_col] == 0)
        df.loc[mask, qw_col] = qw_min
        assert (df["qw_well_test"] != 0).all(), \
            "Zero values found in qw_well_test after preprocessing"

        return df


    def _preprocess_well(self, df: pd.DataFrame) -> pd.DataFrame:
        well_ids = df["well_id"].unique()
        assert len(well_ids) == 1, "DataFrame contains multiple well IDs."
        well_id = well_ids[0]
        print(f"{well_id} original record count: {len(df)}")

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

        df["gor_mpfm"] = np.where(
            df["qo_mpfm"] > 0,
            df["qg_mpfm"] / df["qo_mpfm"],
            np.nan
        )

        df["wc_mpfm_ratio"] = df["wc_mpfm"] / 100
        df["qw_mpfm"] = (df["wc_mpfm_ratio"] * df["qo_mpfm"]) / (1 - df["wc_mpfm_ratio"])

        df["wgr_mpfm"] = np.where(
            df["qg_mpfm"] > 0,
            df["qw_mpfm"] / df["qg_mpfm"],
            np.nan
        )

        df.dropna(subset=get_depdendent_vars(), inplace=True)
        df.dropna(subset=get_independent_vars_with_no_well_code(), inplace=True)

        print(f"{well_id} Record count before target preprocessing: {len(df)}")

        df = df[(df["qo_well_test"] >= 0) & (df["qg_well_test"] >= 0) & (df["qw_well_test"] >= 0)]

        df = self._process_zero_water_rates(df=df)

        # Drop rows where oil and gas rates are both zero. 0))]
        df = df[~((df["qo_well_test"] == 0) & (df["qg_well_test"] == 0))]

        print(f"{well_id} Record count after target preprocessing: {len(df)}")

        df["gor_well_test"] = np.where(
            df["qo_well_test"] > 0,
            df["qg_well_test"] / df["qo_well_test"],
            np.nan
        )

        df["wgr_well_test"] = np.where(
            df["qg_well_test"] > 0,
            df["qw_well_test"] / df["qg_well_test"],
            np.nan
        )

        df = df[get_all_vars()]
        df.sort_index(inplace=True) 
        print(f"{well_id} Record count after preprocessing: {len(df)}")
        return df    


    def preprocess_wells(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy(deep=True)
        dfs = []
        well_ids = df["well_id"].unique()
        print(f"Total original record count {len(df)}")

        # # Calculate the time steps
        reference_time = df.index.min()

        for well_id in well_ids:
            # print(f"Preprocessing well {well_id} with columns {df.columns.tolist()}")
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
    
