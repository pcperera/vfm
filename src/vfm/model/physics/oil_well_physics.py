import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.optimize import least_squares
from src.vfm.constants import *


# =====================================================
# Helpers
# =====================================================

def logistic(x):
    x = np.clip(x, -50, 50)
    return 1 / (1 + np.exp(-x))

# =====================================================
# Oil Well Physics Model
# =====================================================
class OilDominatedMultiphaseWellPhysicsModel:
    """
    Physics-based well flow model with:
    - Liquid inflow relationship
    - Logistic water-cut closure
    - Pressure-driven gas-rate formulation

    This class supports optional geometry-informed constraints and
    partial pooling of physics parameters. When geometry or global
    priors are not available, the model automatically falls back to
    the original calibration behavior.
    """

    # --------------------------------------------------
    # Scaling constants
    # --------------------------------------------------
    P_SCALE = 100.0       # ~100 bar (numerical scaling only)
    T_SCALE = 100.0       # ~100 °C (numerical scaling only)

    def __init__(
        self,
        estimate_pres_offset: float = 10.0,
        fit_pres: bool = True,
        geometry: dict | None = None,        # optional
        global_params: dict | None = None,   # optional
        n_ref: int = 50,                     # reference data size
    ):
        """
        Parameters
        ----------
        estimate_pres_offset : float
            Fallback pressure offset when reservoir pressure
            cannot be estimated from geometry.
        fit_pres : bool
            Whether to estimate reservoir pressure explicitly.
        geometry : dict, optional
            Well geometry parameters (e.g., BH_TVD).
        global_params : dict, optional
            Global physics parameter priors for partial pooling.
        n_ref : int
            Reference number of samples for full parameter freedom.
        """
        self.fit_pres = fit_pres
        self.estimate_pres_offset = estimate_pres_offset

        # NEW: optional geometry and global priors
        self.geometry = geometry or {}
        self.global_params = global_params
        self.n_ref = n_ref

        self.params_ = None

    # --------------------------------------------------
    # Feature matrix for water-cut closure
    # --------------------------------------------------
    def _feature_matrix_for_wc(self, df):
        choke = df["choke"].values
        dcp = df["dcp"].values / self.P_SCALE
        dht = df["dht"].values / self.T_SCALE
        wht = df["wht"].values / self.T_SCALE

        return np.vstack([
            np.ones(len(df)),
            choke,
            dcp,
            dht,
            wht,
            choke * dcp,
            choke * dht,
            dcp * wht,
        ]).T

    # --------------------------------------------------
    # Parameter unpacking helper
    # --------------------------------------------------
    def _unpack(self, x, n_wc):
        i = 0
        P_res = x[i] if self.fit_pres else None; i += self.fit_pres
        qL_max, a, b = x[i:i+3]; i += 3
        Cg, k_ch, ch0 = x[i:i+3]; i += 3
        C_gl = x[i]; i += 1   # Gas lift efficiency
        A_wc = x[i:i+n_wc]
        return P_res, qL_max, a, b, Cg, k_ch, ch0, C_gl, A_wc


    # --------------------------------------------------
    # Residual function for least squares
    # --------------------------------------------------
    def residuals(self, x, df, y_qo, y_qg, y_qw):
        """
        Residual function for nonlinear least squares calibration.

        Includes:
        - Liquid inflow relationship
        - Logistic water-cut closure
        - Pressure-driven reservoir gas rate
        - Additive gas-lift contribution (when available)

        For non–gas-lifted wells or missing signals, the formulation
        automatically reduces to the original physics model.
        """

        n_wc = 8
        (
            P_res,
            qL_max,
            a,
            b,
            Cg,
            k_ch,
            ch0,
            C_gl,
            A,
        ) = self._unpack(x, n_wc)

        # --------------------------------------------------
        # Reservoir pressure fallback
        # --------------------------------------------------
        if P_res is None:
            P_res = df["dhp"].max() + self.estimate_pres_offset

        # --------------------------------------------------
        # Liquid rate (IPR-like formulation)
        # --------------------------------------------------
        Pwf = df["dhp"].values
        pr = np.clip(Pwf / P_res, 0.0, 1.5)

        qL = np.maximum(
            0.0,
            qL_max * (1.0 - a * pr - b * pr**2),
        )

        # --------------------------------------------------
        # Water-cut closure
        # --------------------------------------------------
        wc = logistic(self._feature_matrix_for_wc(df) @ A)

        qw = wc * qL
        qo = (1.0 - wc) * qL

        # --------------------------------------------------
        # Reservoir gas rate (pressure-driven)
        # --------------------------------------------------
        dp = np.sqrt(np.maximum(0.0, P_res - Pwf)) / self.P_SCALE
        choke_eff = logistic(k_ch * (df["choke"].values - ch0))

        qg_res = Cg * dp * choke_eff

        # --------------------------------------------------
        # Gas lift contribution (additive, optional)
        # --------------------------------------------------
        if "gl_mass_rate" in df.columns:
            gl_mass = np.maximum(df["gl_mass_rate"].values, 0.0)

            if "gl_open_ratio" in df.columns:
                gl_or = np.clip(df["gl_open_ratio"].values, 0.0, 1.0)
            else:
                gl_or = 1.0

            qg_lift = (
                C_gl
                * (gl_mass / GL_MASS_TO_STD_VOL)
                * gl_or
            )
        else:
            qg_lift = 0.0

        qg = qg_res + qg_lift

        # --------------------------------------------------
        # Normalized residual vector
        # --------------------------------------------------
        return np.concatenate([
            (qo - y_qo) / max(np.std(y_qo), EPS),
            (qw - y_qw) / max(np.std(y_qw), EPS),
            (qg - y_qg) / max(np.std(y_qg), EPS),
        ])


    # --------------------------------------------------
    # Fit physics model
    # --------------------------------------------------
    def fit(self, df, y_qo, y_qg, y_qw):
        """
        Calibrate physics model parameters for a single well.

        This method supports:
        - Geometry-informed reservoir pressure bounds (if available)
        - Partial pooling of physics parameters (if global priors exist)

        When geometry or global priors are unavailable, the calibration
        defaults to the original unconstrained formulation.
        """

        # --------------------------------------------------
        # 1. Clean input data
        # --------------------------------------------------
        required_cols = [
            "dhp", "choke", "dcp", "dht", "wht",
            y_qo, y_qg, y_qw
        ]
        df = df.dropna(subset=required_cols)

        if len(df) < 10:
            raise ValueError("Not enough valid data points for physics calibration")

        yqo = df[y_qo].values.astype(float)
        yqg = df[y_qg].values.astype(float)
        yqw = df[y_qw].values.astype(float)

        # --------------------------------------------------
        # 2. Safe statistics
        # --------------------------------------------------
        dhp = df["dhp"].values.astype(float)
        dhp_max = float(np.nanmax(dhp))

        qL_mean = float(np.nanmean(yqo + yqw))
        qL_mean = max(qL_mean, EPS)

        # --------------------------------------------------
        # 3. Geometry-informed reservoir pressure bounds
        # --------------------------------------------------
        BH_TVD = self.geometry.get("BH_TVD")

        RHO_LIQ = 850.0  # kg/m3 (representative liquid density)
        G = 9.81         # m/s2

        if BH_TVD is not None and np.isfinite(BH_TVD):
            # Hydrostatic pressure estimate (Pa → bar)
            hydro = RHO_LIQ * G * BH_TVD / 1e5

            P_res0 = dhp_max + hydro
            P_res_lb = dhp_max + 0.7 * hydro
            P_res_ub = dhp_max + 1.3 * hydro
        else:
            # Fallback to original heuristic bounds
            P_res0 = max(dhp_max + self.estimate_pres_offset, dhp_max * 1.05)
            P_res_lb = dhp_max * 1.01
            P_res_ub = dhp_max * 3.0

        # --------------------------------------------------
        # 4. Initial guess and bounds
        # --------------------------------------------------
        n_wc = 8

        x0 = (
            [P_res0, qL_mean, 0.2, 0.5, 50.0, 5.0, 0.3, 0.01]
            + [0.0] * n_wc
        )

        lb = (
            [P_res_lb, 0.0, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0]
            + [-5.0] * n_wc
        )

        ub = (
            [P_res_ub, 1e5, 1.0, 2.0, 1e5, 50.0, 1.0, 1.0]
            + [5.0] * n_wc
        )

        # --------------------------------------------------
        # 5. Least-squares optimization
        # --------------------------------------------------
        res = least_squares(
            self.residuals,
            x0,
            bounds=(lb, ub),
            args=(df, yqo, yqg, yqw),
            max_nfev=30000,
            xtol=1e-8,
            ftol=1e-8,
            gtol=1e-8,
        )

        P_res, qL_max, a, b, Cg, k_ch, ch0, C_gl, A = self._unpack(res.x, n_wc)

        # --------------------------------------------------
        # 6. Partial pooling of physics parameters
        # --------------------------------------------------
        if self.global_params is not None:
            n_obs = len(df)
            alpha = min(1.0, n_obs / self.n_ref)

            # Shrink sparse-well parameters toward global physics
            qL_max = alpha * qL_max + (1 - alpha) * self.global_params.get("qL_max", qL_max)
            a       = alpha * a       + (1 - alpha) * self.global_params.get("a", a)
            b       = alpha * b       + (1 - alpha) * self.global_params.get("b", b)
            Cg      = alpha * Cg      + (1 - alpha) * self.global_params.get("Cg", Cg)
            k_ch    = alpha * k_ch    + (1 - alpha) * self.global_params.get("k_choke", k_ch)
            ch0     = alpha * ch0     + (1 - alpha) * self.global_params.get("choke0", ch0)

        # --------------------------------------------------
        # 7. Store calibrated parameters
        # --------------------------------------------------
        self.params_ = dict(
            P_res=P_res,
            qL_max=qL_max,
            a=a,
            b=b,
            Cg=Cg,
            C_gl=C_gl,   # Gas lift
            k_choke=k_ch,
            choke0=ch0,
            A_wc=A,
        )

        return self

    def predict(self, df):
        """
        Physics-only prediction.

        Includes:
        - Liquid inflow relationship
        - Logistic water-cut closure
        - Pressure-driven reservoir gas rate
        - Additive gas-lift contribution (when available)

        For non-gas-lifted wells or missing gas-lift signals,
        the formulation automatically reduces to the original
        physics model.
        """

        p = self.params_

        # --------------------------------------------------
        # Liquid rates
        # --------------------------------------------------
        Pwf = df["dhp"].values
        pr = np.clip(Pwf / p["P_res"], 0.0, 1.5)

        qL = np.maximum(
            0.0,
            p["qL_max"] * (1.0 - p["a"] * pr - p["b"] * pr**2),
        )

        wc = logistic(self._feature_matrix_for_wc(df) @ p["A_wc"])

        qw = wc * qL
        qo = (1.0 - wc) * qL

        # --------------------------------------------------
        # Reservoir gas (pressure-driven)
        # --------------------------------------------------
        dp = np.sqrt(np.maximum(0.0, p["P_res"] - Pwf)) / self.P_SCALE
        choke_eff = logistic(
            p["k_choke"] * (df["choke"].values - p["choke0"])
        )

        qg_res = p["Cg"] * dp * choke_eff

        # --------------------------------------------------
        # Gas lift contribution (optional, additive)
        # --------------------------------------------------
        if "gl_mass_rate" in df.columns:
            gl_mass = np.maximum(df["gl_mass_rate"].values, 0.0)

            if "gl_open_ratio" in df.columns:
                gl_or = np.clip(df["gl_open_ratio"].values, 0.0, 1.0)
            else:
                gl_or = 1.0

            qg_lift = (
                p["C_gl"]
                * (gl_mass / GL_MASS_TO_STD_VOL)
                * gl_or
            )
        else:
            qg_lift = 0.0

        qg = qg_res + qg_lift

        # --------------------------------------------------
        # Output
        # --------------------------------------------------
        return pd.DataFrame(
            {
                "qo_pred": qo,
                "qw_pred": qw,
                "qg_pred": qg,
            },
            index=df.index,
        )

