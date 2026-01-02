import pandas as pd
from src.vfm.utils.descriptive_utils import *


class Preprocessor:

    def __init__(self):
        # self._time_idx_resolution_min = 1
        self._independent_tp_vars = get_independent_tp_vars()
        self._independent_vars = get_independent_vars_with_no_well_code()
        self._well_col = "well_id"
        self.qg_col: str = "qg_well_test"
        self.qo_col: str = "qo_well_test"
        self.qw_col: str = "qw_well_test"
        self.choke_col: str = "choke"

    def _process_zero_water_rates(self, df):
        well_ids = df[self._well_col].unique()
        assert len(well_ids) == 1, "DataFrame contains multiple well IDs."
        well_id = well_ids[0]
        qw_col="qw_well_test"

        pos = df[df[qw_col] > 0]
        if pos.empty:
            return df

        # qw_min = pos[qw_col].min()
        # t_wb = pos.index.min()

        # mask = (df[self._well_col] == well_id) & (df.index >= t_wb) & (df[qw_col] == 0)
        # df.loc[mask, qw_col] = qw_min
        # assert (df["qw_well_test"] != 0).all(), \
        #     "Zero values found in qw_well_test after preprocessing"

        nonzero_min = df.loc[df[qw_col] > 0, qw_col].min()
        eps = nonzero_min / 1000
        print(f"eps={eps}")
        df[qw_col] = np.where(df[qw_col] == 0.0, eps, df[qw_col])

        return df
    

    def _robust_zscore(self, x: pd.Series) -> pd.Series:
        """
        Robust z-score using median and MAD.
        """
        median = np.nanmedian(x)
        mad = np.nanmedian(np.abs(x - median))

        if mad == 0:
            return pd.Series(np.zeros(len(x)), index=x.index)

        return 0.6745 * (x - median) / mad

    def _drop_non_physical_well_tests(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Drop non-physical well-test measurements using:
        (1) choke = 0  ⇒ all phase rates must be 0
        (2) choke > 0  ⇒ at least one phase rate must be > 0
        (3) statistical consistency of gas rate under flowing conditions
            (MAD-based robust z-score)

        No fixed rate thresholds are used.
        """

        df = df.copy()
        n_before = len(df)

        # -------------------------------------------------
        # Rule 1: Physically invalid pressure/temperature drops
        # -------------------------------------------------
        df = df[(df["dht"] >= df["wht"])]
        df = df[(df["dhp"] >= df["whp"])]

        # -------------------------------------------------
        # Rule 1: choke = 0 ⇒ no flow
        # -------------------------------------------------
        choke_zero = df[self.choke_col] == 0
        any_rate_nonzero = (
            (df[self.qg_col] != 0) |
            (df[self.qo_col] != 0) |
            (df[self.qw_col] != 0)
        )

        invalid_shutin = choke_zero & any_rate_nonzero
        n_shutin = invalid_shutin.sum()

        # -------------------------------------------------
        # Rule 2: choke > 0 ⇒ at least one phase must flow
        # -------------------------------------------------
        choke_open = df[self.choke_col] > 0
        all_rates_zero = (
            (df[self.qg_col] == 0) &
            (df[self.qo_col] == 0) &
            (df[self.qw_col] == 0)
        )

        invalid_open_no_flow = choke_open & all_rates_zero
        n_open_no_flow = invalid_open_no_flow.sum()

        # ------------------------------------
        # Rule 3: gas-rate collapse under flowing conditions
        #   (a) relative collapse: < median / 1000
        #   (b) absolute safety floor: < 30 Sm3/h
        # ------------------------------------
        flowing_mask = df[self.choke_col] > 0
        invalid_collapse = pd.Series(False, index=df.index)

        n_rel_collapse_meadian = 0
        n_floor = 0
        safety_floor_value = 50.0  # Sm3/h

        if flowing_mask.sum() > 0:
            qg_flowing = df.loc[flowing_mask, self.qg_col]
            median_qg = np.nanmedian(qg_flowing)

            rel_collapse_mask_median = qg_flowing < (median_qg / 1000.0)
            floor_mask = qg_flowing < safety_floor_value  # Sm3/h safety floor


            invalid_collapse.loc[flowing_mask] = (rel_collapse_mask_median | floor_mask)
            n_rel_collapse_meadian = rel_collapse_mask_median.sum()


        # ------------------------------------
        # Combine all invalid rules
        # ------------------------------------
        invalid = invalid_shutin | invalid_open_no_flow | invalid_collapse
        n_total_dropped = invalid.sum()

        df_clean = df.loc[~invalid].sort_index()

        # ------------------------------------
        # Diagnostics / logging
        # ------------------------------------
        print(
            f"[Non-physical filter] Rows before: {n_before}, "
            f"dropped: {n_total_dropped} "
            f"(shut-in violations: {n_shutin}, "
            f"open-no-flow: {n_open_no_flow}, "
            f"relative collapse (< median/1000): {n_rel_collapse_meadian}, "
            f"safety floor (<{safety_floor_value} Sm3/h): {n_floor}), "
            f"remaining: {len(df_clean)}"
        )

        return df_clean


    def _preprocess_well(self, df: pd.DataFrame) -> pd.DataFrame:
        well_ids = df["well_id"].unique()
        assert len(well_ids) == 1, "DataFrame contains multiple well IDs."
        well_id = well_ids[0]
        print(f"{well_id} original record count: {len(df)}")

        # Data transformation
        df.sort_index(inplace=True)  # Ensure chronological order

        # Data quality filters
        df = df[df["dhp"] != 0]
        df = df[df["whp"] != 0]
        df = df[df["dcp"] != 0]
        df = df[df["dht"] != 0]
        df = df[df["wht"] != 0]
        # df = df[df["choke"] != 0]

        df["well_id"] = well_id
        df["well_code"] = well_id
        df["pres_drop"] = df["dhp"] - df["whp"]
        df["temp_drop"] = df["dht"] - df["wht"]

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

        df = df[(df["qo_well_test"] >= 0) & (df["qg_well_test"] >= 0) & (df["qw_well_test"] >= 0)]

        # df = self._process_zero_water_rates(df=df)
        df = self._drop_non_physical_well_tests(df=df)

        # Drop rows where oil and gas rates are both zero. 0))]
        # df = df[~((df["qo_well_test"] == 0) & (df["qg_well_test"] == 0))]

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
    
