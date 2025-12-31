import numpy as np
import pandas as pd

EPS = 1e-9


class GasDominatedMultiphaseWellPhysicsModel:
    """
    Gas-dominated multiphase well physics model.

    Gas is the controlling phase.
    Oil (condensate) and water are reconstructed using
    physically bounded phase-partitioning relationships
    (CGR and WC).
    """

    # -------------------------------
    # Latent definition (REQUIRED)
    # -------------------------------
    latent_names = ["log_qg", "logit_wc", "log_cgr"]

    # --------------------------------------------------
    # Construction
    # --------------------------------------------------
    def __init__(self):
        self.Cg = None
        self.n = None
        self.alpha_choke = None
        self.y_qo_col:str = "qo_well_test"
        self.y_qg_col:str ="qg_well_test"
        self.y_qw_col:str ="qw_well_test"
        self.wc_coef = None
        self.cgr = None

        self.d_choke_max = None
        self.is_fitted = False

    # --------------------------------------------------
    # Utilities
    # --------------------------------------------------
    @staticmethod
    def _sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    # --------------------------------------------------
    # Fit
    # --------------------------------------------------
    def fit(self, df: pd.DataFrame):

        required = {"dhp", "whp", "choke", self.y_qg_col}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        pr = df["dhp"].values
        pwh = df["whp"].values
        choke = np.maximum(df["choke"].values, EPS)
        qg = np.maximum(df[self.y_qg_col].values, EPS)

        # Gas deliverability
        self.d_choke_max = np.nanmax(choke)
        dp2 = np.maximum(pr**2 - pwh**2, EPS)

        n_est, logCg_est = np.polyfit(np.log(dp2), np.log(qg), 1)
        self.n = float(np.clip(n_est, 0.4, 1.2))
        self.Cg = float(np.exp(logCg_est))
        self.alpha_choke = 0.5

        # CGR
        if self.y_qo_col in df.columns:
            self.cgr = float(np.nanmedian(df[self.y_qo_col].values / (qg + EPS)))
        else:
            self.cgr = EPS

        # WC model
        if self.y_qw_col in df.columns and self.cgr > EPS:
            qo = self.cgr * qg
            wc = df[self.y_qw_col].values / (qo + df[self.y_qw_col].values + EPS)
            wc = np.clip(wc, EPS, 1 - EPS)

            Z = np.column_stack([
                np.ones(len(df)),
                pr - pwh,
                np.log(qg + EPS),
            ])

            self.wc_coef, *_ = np.linalg.lstsq(
                Z, np.log(wc / (1 - wc)), rcond=None
            )
        else:
            self.wc_coef = np.array([-10.0, 0.0, 0.0])

        self.is_fitted = True
        return self

    # --------------------------------------------------
    # Physics prediction (standalone)
    # --------------------------------------------------
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:

        if not self.is_fitted:
            raise RuntimeError("Model must be fitted")

        pr = df["dhp"].values
        pwh = df["whp"].values
        choke = np.maximum(df["choke"].values, EPS)

        dp2 = np.maximum(pr**2 - pwh**2, 0.0)
        choke_factor = (choke / self.d_choke_max) ** self.alpha_choke

        qg = self.Cg * (dp2 ** self.n) * choke_factor
        qo = self.cgr * qg

        Z = np.column_stack([
            np.ones(len(df)),
            pr - pwh,
            np.log(qg + EPS),
        ])
        wc = self._sigmoid(Z @ self.wc_coef)
        wc = np.clip(wc, 0.0, 0.95)

        qw = (wc / (1 - wc + EPS)) * qo

        return pd.DataFrame(
            {
                "qo_pred": np.maximum(qo, 0.0),
                "qw_pred": np.maximum(qw, 0.0),
                "qg_pred": np.maximum(qg, 0.0),
            },
            index=df.index,
        )

    # --------------------------------------------------
    # Latent interface (USED BY HYBRID)
    # --------------------------------------------------
    def forward_latent(self, df: pd.DataFrame, use_measured: bool):

        if use_measured:
            qg = np.maximum(df[self.y_qg_col].values, EPS)
            qo = np.maximum(df.get(self.y_qo_col, 0.0), EPS)
            qw = np.maximum(df.get(self.y_qw_col, 0.0), EPS)
        else:
            pred = self.predict(df)
            qg = np.maximum(pred["qg_pred"].values, EPS)
            qo = np.maximum(pred["qo_pred"].values, EPS)
            qw = np.maximum(pred["qw_pred"].values, EPS)

        wc = np.clip(qw / (qo + qw + EPS), EPS, 1 - EPS)
        cgr = np.maximum(qo / (qg + EPS), EPS)

        return {
            "log_qg": np.log(qg),
            "logit_wc": np.log(wc / (1 - wc)),
            "log_cgr": np.log(cgr),
        }

    def reconstruct(self, latent: dict, df: pd.DataFrame) -> pd.DataFrame:

        qg = np.exp(latent["log_qg"])
        wc = self._sigmoid(latent["logit_wc"])
        cgr = np.exp(latent["log_cgr"])

        qo = cgr * qg
        qw = (wc / (1 - wc + EPS)) * qo

        return pd.DataFrame(
            {
                "qo_pred": np.maximum(qo, 0.0),
                "qw_pred": np.maximum(qw, 0.0),
                "qg_pred": np.maximum(qg, 0.0),
            },
            index=df.index,
        )
