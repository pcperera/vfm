import pandas as pd

RESAMPLE_MEAN = "mean"
RESAMPLE_FIRST = "first"

class Preprocessor:

    def __init__(self, df: pd.DataFrame):
        self._df = df.copy(deep=True)
        self._resample_period_min = 1

    def preprocess(self, well_id: str):
        # Data transformation
        # Ignore records where DHP is zero. These are 
        self._df = self._df[self._df["dhp"] != 0]
        self._df = self._df[self._df["whp"] != 0]

        self._df = self._df.resample(f"{self._resample_period_min}T").agg({
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
            "wc_mpfm": RESAMPLE_MEAN
        })
        
        self._df["well_id"] = well_id
        self._df.sort_index(inplace=True)  # Ensure chronological order
        independent_tp_vars = ["dhp", "dht", "whp", "wht", "dcp"]

        # Convert choke to fraction
        self._df["choke"] /= 100

        self._df[independent_tp_vars] = self._df[independent_tp_vars].interpolate(
            method="time",
            limit_direction="both"      # fill start & end gaps
        )

        # Convert pressures to Pa
        # for item in ["dhp", "whp", "dcp"]:
        #     df[item] *= 1E5

        # Calculate MPFM flow rates from GOR and WC
        self._df["gor_mpfm"] = self._df["qg_mpfm"] / self._df["qo_mpfm"]
        self._df["qw_mpfm"] = (self._df["wc_mpfm"] * self._df["qo_mpfm"]) / (1 - self._df["wc_mpfm"])

        # Training data (based on well test)
        # fields_to_drop_na = ["qo_mpfm"]
        independent_vars = independent_tp_vars.copy()
        independent_vars.append("choke")
        all_vars = independent_vars.copy()
        all_vars.extend(["well_id", "qo_well_test", "qg_well_test", "qw_well_test"])
        
        self._df = self._df[all_vars]
        self._df['choke'] = self._df['choke'].ffill()    
        self._df[independent_tp_vars] = self._df[independent_tp_vars].interpolate(method="time", limit_direction="forward")
        self._df = self._df.dropna(subset=independent_vars)
        # df = df[(df["choke"] > 0) & (df["qo_well_test"] > 0) & (df["qg_well_test"] > 0) & (df["qw_well_test"] > 0)]
        return self._df
    
        # df = df[df[['qo_well_test', 'qg_well_test', 'qw_well_test']].notna().any(axis=1)]
        # return df[(df["choke"] > 0) & (df["qo_well_test"] > 0) & (df["qg_well_test"] > 0) & (df["qw_well_test"] > 0)]
        # df = df.dropna(subset=fields_to_drop_na)
        # df = df.fillna(method="ffill")
        # return df[(df["choke"] > 0) & (df["qo_well_test"] > 0)]
        # return df[(df["qo_well_test"] > 0)]
        # return df[(df["choke"] > 0) & (df["qo_mpfm"] > 0) & (df["qg_mpfm"] > 0) & (df["qw_mpfm"] > 0)]

    def preprocess_timeseries(self, well_id: str):
        df = self.preprocess(well_id=well_id)

        # Calculate the time steps
        reference_time = df.index.min()  # You can choose any reference time you want
        # df["time_step"] = (df.index - reference_time).total_seconds() / 60
        # df["time_step"] = df["time_step"].astype(int)
        df["time_idx"] = (df.index - reference_time).total_seconds() // (60 * self._resample_period_min)
        df["time_idx"] = df["time_idx"].astype(int)
        return df
