import pandas as pd

RESAMPLE_MEAN = "mean"
RESAMPLE_FIRST = "first"

class Preprocessor:

    def __init__(self):
        self._resample_period_min = 120
        self._independent_tp_vars = ["dhp", "dht", "whp", "wht", "dcp"]
        self._independent_vars = self._independent_tp_vars.copy()
        self._independent_vars.append("choke")
        self._dependent_vars = ["qo_well_test", "qg_well_test", "qw_well_test"]

    def preprocess_well(self, df: pd.DataFrame) -> pd.DataFrame:
        well_ids = df["well_id"].unique()
        assert len(well_ids) == 1, "DataFrame contains multiple well IDs."
        well_id = well_ids[0]

        # Data transformation
        df.sort_index(inplace=True)  # Ensure chronological order

        # Ignore records where DHP is zero. These are 
        df = df[df["dhp"] != 0]
        df = df[df["whp"] != 0]
        print(df.index)

        df = df.resample(f"{self._resample_period_min}T").agg({
            "whp": RESAMPLE_MEAN,
            "wht": RESAMPLE_MEAN,
            "dhp": RESAMPLE_MEAN,
            "dht": RESAMPLE_MEAN,
            "choke": RESAMPLE_MEAN,
            "dcp": RESAMPLE_MEAN,
            "qo_well_test": RESAMPLE_MEAN,
            "qg_well_test": RESAMPLE_MEAN,
            "qw_well_test": RESAMPLE_MEAN,
            "qo_mpfm": RESAMPLE_MEAN,
            "qg_mpfm": RESAMPLE_MEAN,
            "wc_mpfm": RESAMPLE_MEAN,
        })

        df["well_id"] = well_id

        # Convert choke to fraction
        df["choke"] /= 100

        df[self._independent_tp_vars] = df[self._independent_tp_vars].interpolate(
            method="time",
            limit_direction="both"      # fill start & end gaps
        )

        # Convert pressures from bara to Pa
        for item in ["dhp", "whp", "dcp"]:
            df[item] *= 1E5

        # Convert C to K
        for item in ["dht", "wht"]:
            df[item] = df[item] + 273.15

        # Calculate MPFM flow rates from GOR and WC
        df["gor_mpfm"] = df["qg_mpfm"] / df["qo_mpfm"]
        df["qw_mpfm"] = (df["wc_mpfm"] * df["qo_mpfm"]) / (1 - df["wc_mpfm"])

        # Training data (based on well test)
        # fields_to_drop_na = ["qo_mpfm"]
        all_vars = self._independent_vars.copy()
        all_vars.extend(["well_id", "qo_well_test", "qg_well_test", "qw_well_test"])
        
        df = df[all_vars]
        df['choke'] = df['choke'].ffill()
        df[self._independent_tp_vars] = df[self._independent_tp_vars].interpolate(method="time", limit_direction="forward")
        df = df.dropna(subset=self._independent_vars)
        df = df[(df["choke"] > 0) & (df["dhp"] > 0) & (df["dht"] > 0) & (df["whp"] > 0) & (df["wht"] > 0) & (df["dcp"] > 0)]
        return df
    
        # df = df[df[['qo_well_test', 'qg_well_test', 'qw_well_test']].notna().any(axis=1)]
        # return df[(df["choke"] > 0) & (df["qo_well_test"] > 0) & (df["qg_well_test"] > 0) & (df["qw_well_test"] > 0)]
        # df = df.dropna(subset=fields_to_drop_na)
        # df = df.fillna(method="ffill")
        # return df[(df["choke"] > 0) & (df["qo_well_test"] > 0)]
        # return df[(df["qo_well_test"] > 0)]
        # return df[(df["choke"] > 0) & (df["qo_mpfm"] > 0) & (df["qg_mpfm"] > 0) & (df["qw_mpfm"] > 0)]

    def preprocess_timeseries(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy(deep=True)
        dfs = []
        well_ids = df["well_id"].unique()

        # Calculate the time steps
        reference_time = df.index.min()

        for well_id in well_ids:
            print(f"Preprocessing well {well_id}... with columns {df.columns.tolist()}")
            df_well = df[df["well_id"] == well_id]
            df_well = self.preprocess_well(df=df_well)
            df_well["time_idx"] = (df_well.index - reference_time).total_seconds() // (60 * self._resample_period_min)
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
