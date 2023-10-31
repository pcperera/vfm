import pandas as pd


class Preprocessor:

    def __init__(self, df_raw: pd.DataFrame):
        self.__df_raw = df_raw

    def preprocess(self):
        # Data transformation
        df = self.__df_raw

        # Convert pressures to Pa
        for item in ['dhp', 'whp', 'dcp']:
            df[item] *= 1E5

        # Convert choke to fraction
        df['choke'] /= 100

        # Calculate MPFM flow rates from GOR and WC
        df['qg_mpfm'] = df['qo_mpfm'] * df['gor_mpfm']
        df['qw_mpfm'] = (df['wc_mpfm'] * df['qo_mpfm']) / (1 - df['wc_mpfm'])

        # Training data (based on well test)
        fields_to_drop_na = ['qo_mpfm', 'qg_mpfm', 'qw_mpfm']
        df = df[['dhp', 'dht', 'whp', 'wht', 'choke', 'dcp', 'qo_mpfm', 'qg_mpfm', 'qw_mpfm']]
        df = df.dropna(subset=fields_to_drop_na)
        df = df.fillna(method='ffill')
        return df[(df['choke'] > 0) & (df['qo_mpfm'] > 0) & (df['qg_mpfm'] > 0) & (df['qw_mpfm'] > 0)]

    def preprocess_timeseries(self):
        df = self.preprocess()

        # Calculate the time steps
        reference_time = df.index.min()  # You can choose any reference time you want
        df['time_step'] = (df.index - reference_time).total_seconds() / 60
        df['time_step'] = df['time_step'].astype(int)
        return df
