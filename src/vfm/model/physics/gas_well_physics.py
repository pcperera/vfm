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

    # --------------------------------------------------
    # Construction
    # --------------------------------------------------
    def __init__(self):
        self.Cg = None
        self.n = None
        self.alpha_choke = None

        self.wc_coef = None        # coefficients for WC model
        self.cgr = None            # condensate-gas ratio

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
        """
        Fit gas deliverability, WC model, and CGR.

        Required columns:
        - dhp, whp, choke, qg
        Optional:
        - qo, qw
        """

        required = {"dhp", "whp", "choke", "qg"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        pr = df["dhp"].values
        pwh = df["whp"].values
        choke = np.maximum(df["choke"].values, EPS)
        qg = np.maximum(df["qg"].values, EPS)

        # -----------------------------
        # Gas deliverability
        # -----------------------------
        self.d_choke_max = np.nanmax(choke)

        dp2 = np.maximum(pr**2 - pwh**2, EPS)

        X = np.log(dp2)
        y = np.log(qg)

        n_est, logCg_est = np.polyfit(X, y, 1)

        self.n = float(np.clip(n_est, 0.4, 1.2))
        self.Cg = float(np.exp(logCg_est))
        self.alpha_choke = 0.5  # conservative default

        # -----------------------------
        # Condensate–Gas Ratio (CGR)
        # -----------------------------
        if "qo" in df.columns:
            self.cgr = float(
                np.nanmedian(df["qo"].values / (qg + EPS))
            )
        else:
            self.cgr = 0.0

        # -----------------------------
        # Water Cut (WC) model
        # -----------------------------
        if "qw" in df.columns and self.cgr > 0:
            qo = self.cgr * qg
            wc = df["qw"].values / (qo + df["qw"].values + EPS)
            wc = np.clip(wc, EPS, 1 - EPS)

            drawdown = pr - pwh
            Z = np.column_stack([
                np.ones(len(df)),
                drawdown,
                np.log(qg + EPS),
            ])

            logit_wc = np.log(wc / (1 - wc))
            self.wc_coef, *_ = np.linalg.lstsq(Z, logit_wc, rcond=None)
        else:
            self.wc_coef = np.array([ -10.0, 0.0, 0.0 ])  # near-zero WC

        self.is_fitted = True
        return self

    # --------------------------------------------------
    # Predict
    # --------------------------------------------------
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict qo, qw, qg using gas-dominated multiphase physics.
        """

        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")

        pr = df["dhp"].values
        pwh = df["whp"].values
        choke = np.maximum(df["choke"].values, EPS)

        # -----------------------------
        # Gas rate
        # -----------------------------
        dp2 = np.maximum(pr**2 - pwh**2, 0.0)
        choke_factor = (choke / self.d_choke_max) ** self.alpha_choke

        qg = self.Cg * (dp2 ** self.n) * choke_factor

        # -----------------------------
        # Oil (condensate)
        # -----------------------------
        qo = self.cgr * qg

        # -----------------------------
        # Water cut
        # -----------------------------
        drawdown = pr - pwh
        Z = np.column_stack([
            np.ones(len(df)),
            drawdown,
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
