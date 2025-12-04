import pandas as pd


class Preprocessor:

    def __init__(self, df: pd.DataFrame):
        self._df = df.copy(deep=True)

    def preprocess(self):
        # Data transformation
        self._df = self._df.sort_index()  # Ensure chronological order
        independent_tp_vars = ["dhp", "dht", "whp", "wht", "dcp"]

        # Convert choke to fraction
        self._df["choke"] /= 100

        # Ignore records where DHP is zero. These are 
        self._df = self._df[self._df["dhp"] != 0]
        self._df = self._df[self._df["whp"] != 0]

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
        
        fields_to_forward_fill = independent_vars
        self._df = self._df[["dhp", "dht", "whp", "wht", "choke", "dcp", "qo_well_test", "qg_well_test", "qw_well_test", "well_id"]]
        self._df[fields_to_forward_fill] = self._df[fields_to_forward_fill].ffill()
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

    def preprocess_timeseries(self):
        df = self.preprocess()

        # Calculate the time steps
        reference_time = df.index.min()  # You can choose any reference time you want
        # df["time_step"] = (df.index - reference_time).total_seconds() / 60
        # df["time_step"] = df["time_step"].astype(int)
        df["time_idx"] = (df.index - reference_time).total_seconds() // 60 
        df["time_idx"] = df["time_idx"].astype(int)
        return df
