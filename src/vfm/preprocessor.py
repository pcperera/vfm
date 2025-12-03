import pandas as pd


class Preprocessor:

    def __init__(self, df: pd.DataFrame):
        self._df_raw = df

    def preprocess(self):
        # Data transformation
        df = self._df_raw.copy(deep=True)
        df = df.sort_index()  # Ensure chronological order

        # Convert pressures to Pa
        for item in ["dhp", "whp", "dcp"]:
            df[item] *= 1E5
    
        # Convert choke to fraction
        df["choke"] /= 100

        # Calculate MPFM flow rates from GOR and WC
        df["gor_mpfm"] = df["qg_mpfm"] / df["qo_mpfm"]
        df["qw_mpfm"] = (df["wc_mpfm"] * df["qo_mpfm"]) / (1 - df["wc_mpfm"])

        # Training data (based on well test)
        # fields_to_drop_na = ["qo_mpfm"]
        fields_to_drop_na = ["qo_well_test"]
        fields_to_forward_fill = ["dhp", "dht", "whp", "wht", "choke", "dcp"]
        df = df[["dhp", "dht", "whp", "wht", "choke", "dcp", "qo_mpfm", "qg_mpfm", "qw_mpfm", "qo_well_test", "qg_well_test", "qw_well_test", "well_id"]]
        df[fields_to_forward_fill].fillna(method="ffill", inplace=True)
        # df = df[df[['qo_well_test', 'qg_well_test', 'qw_well_test']].notna().any(axis=1)]
        # return df[(df["choke"] > 0) & (df["qo_well_test"] > 0) & (df["qg_well_test"] > 0) & (df["qw_well_test"] > 0)]
        return df
        # df = df.dropna(subset=fields_to_drop_na)
        # df = df.fillna(method="ffill")
        # return df[(df["choke"] > 0) & (df["qo_well_test"] > 0)]
        # return df[(df["qo_well_test"] > 0)]
        # return df[(df["choke"] > 0) & (df["qo_mpfm"] > 0) & (df["qg_mpfm"] > 0) & (df["qw_mpfm"] > 0)]
        # return df[(df["choke"] > 0) & (df["qo_well_test"] > 0) & (df["qg_well_test"] > 0) & (df["qw_well_test"] > 0)]

    def preprocess_timeseries(self):
        df = self.preprocess()

        # Calculate the time steps
        reference_time = df.index.min()  # You can choose any reference time you want
        # df["time_step"] = (df.index - reference_time).total_seconds() / 60
        # df["time_step"] = df["time_step"].astype(int)
        df["time_idx"] = (df.index - reference_time).total_seconds() // 60 
        df["time_idx"] = df["time_idx"].astype(int)
        return df
