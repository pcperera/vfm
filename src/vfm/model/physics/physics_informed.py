import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.optimize import least_squares
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from src.vfm.constants import *


# =====================================================
# Helpers
# =====================================================

def compute_wgr(qw, qg, min_qg=50.0):
    wgr = np.full_like(qw, np.nan, dtype=float)

    mask = (
        (qg > min_qg) &
        ~np.isnan(qw) &
        ~np.isnan(qg) &
        (qg != 0)
    )

    wgr[mask] = qw[mask] / qg[mask]
    return wgr

def compute_gor(qg, qo, min_qo=5.0):
    gor = np.full_like(qg, np.nan, dtype=float)

    mask = (
        (qo > min_qo) &
        ~np.isnan(qg) &
        ~np.isnan(qo) &
        (qo != 0)
    )

    gor[mask] = qg[mask] / qo[mask]
    return gor


def logistic(x):
    x = np.clip(x, -50, 50)
    return 1 / (1 + np.exp(-x))

def regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mask = (
        np.isfinite(y_true)
        & np.isfinite(y_pred)
        & (np.abs(y_true) > EPS)
    )

    if mask.sum() < 2:
        return {m: np.nan for m in METRICS}

    y_true = y_true[mask]
    y_pred = y_pred[mask]

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    mape = np.mean(np.abs(y_true - y_pred) / np.abs(y_true)) * 100
    mpe  = np.mean((y_pred - y_true) / y_true) * 100

    return r2, mae, rmse, mape, mpe


# =====================================================
# Physics Model
# =====================================================

class PhysicsModel:
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

# =============================
# Physics-Informed Hybrid Model
# =============================
class PhysicsInformedHybridModel:

    def __init__(self, 
                    dependant_vars, 
                    independent_vars,
                    well_id_col:str = "well_id", 
                    y_qo_col:str = "qo_well_test",
                    y_qg_col:str ="qg_well_test",
                    y_qw_col:str ="qw_well_test",
                    mpfm_qo_col: str = "qo_mpfm",
                    mpfm_qg_col: str = "qg_mpfm",
                    mpfm_qw_col: str = "qw_mpfm",
                    mpfm_gor_col: str = "gor_mpfm",
                    mpfm_wgr_col: str = "wgr_mpfm",
                    degree=1, 
                    lags=1,
                    well_geometry: dict | None = None):

        self.dependant_vars = dependant_vars
        self.independent_vars = independent_vars
        self.well_id_col = well_id_col
        self.y_qo_col = y_qo_col
        self.y_qg_col = y_qg_col
        self.y_qw_col = y_qw_col
        self.mpfm_qo_col = mpfm_qo_col
        self.mpfm_qg_col = mpfm_qg_col
        self.mpfm_qw_col = mpfm_qw_col
        self.mpfm_gor_col = mpfm_gor_col
        self.mpfm_wgr_col = mpfm_wgr_col
        self.degree = degree
        self.lags = lags
        self.well_geometry = well_geometry or {}

        self.phys_models = {}

        self.scaler = StandardScaler()
        self.poly = PolynomialFeatures(degree, include_bias=False)

        self.ml_residual = MultiOutputRegressor(
            HistGradientBoostingRegressor(
                max_iter=300,
                learning_rate=0.05,
                max_depth=3,
                min_samples_leaf=20,
                early_stopping=True,
                n_iter_no_change=30,
                validation_fraction=None,  # we supply validation explicitly
                random_state=42,
            )
        )


    # --------------------------------------------------
    # Lag features
    # --------------------------------------------------
    def _create_lagged_features(self, df):
        df_lag = df.copy()

        for lag in range(1, self.lags + 1):
            for col in ["dhp", "whp"]:
                df_lag[f"{col}_lag{lag}"] = (
                    df_lag.groupby(self.well_id_col)[col].shift(lag)
                )

        # IMPORTANT: DO NOT drop NaNs here
        return df_lag


    # --------------------------------------------------
    # Fit
    # --------------------------------------------------
    def fit(
        self,
        df: pd.DataFrame,
        df_val: pd.DataFrame | None = None
    ):
        """
        Fit physics models per well and a global ML residual model.

        Enhancements over the base implementation:
        -------------------------------------------
        1. Geometry-informed hard constraints during physics calibration
        (only when geometry is available).
        2. Partial pooling (hierarchical regularization) of physics parameters
        across wells to improve robustness for sparse wells.

        If geometry or global physics priors are not available, the method
        automatically falls back to the original per-well independent
        physics calibration strategy.
        """

        # ==================================================
        # 1. First pass: fit physics models independently
        #    (used to compute global physics priors)
        # ==================================================
        temp_phys_models: dict[str, PhysicsModel] = {}

        for wid, d in df.groupby(self.well_id_col):
            try:
                temp_phys_models[wid] = PhysicsModel().fit(
                    d, self.y_qo_col, self.y_qg_col, self.y_qw_col
                )
            except Exception as e:
                # Fail-safe: skip wells that cannot be calibrated
                print(f"[WARN] Initial physics fit failed for well {wid}: {e}")

        # ==================================================
        # 2. Compute global physics priors (if possible)
        # ==================================================
        def _compute_global_physics_priors(phys_models: dict):
            """
            Compute global mean physics parameters across wells.
            Used for partial pooling. Returns None if insufficient data.
            """
            params = [m.params_ for m in phys_models.values() if m.params_ is not None]
            if not params:
                return None

            keys = ["qL_max", "a", "b", "Cg", "k_choke", "choke0"]
            priors = {}

            for k in keys:
                values = [p[k] for p in params if np.isfinite(p.get(k, np.nan))]
                if values:
                    priors[k] = float(np.mean(values))

            return priors if priors else None

        global_physics_params = _compute_global_physics_priors(temp_phys_models)

        # ==================================================
        # 3. Second pass: refit physics models with
        #    geometry constraints + partial pooling
        # ==================================================
        self.phys_models = {}

        for wid, d in df.groupby(self.well_id_col):
            # Fetch geometry for this well (if available)
            geom = None
            if hasattr(self, "well_geometry"):
                geom = self.well_geometry.get(wid)

            try:
                self.phys_models[wid] = PhysicsModel(
                    geometry=geom,
                    global_params=global_physics_params,
                ).fit(d, self.y_qo_col, self.y_qg_col, self.y_qw_col)
            except Exception as e:
                # Fallback: original unconstrained physics model
                print(f"[WARN] Geometry-aware fit failed for well {wid}, falling back: {e}")
                self.phys_models[wid] = PhysicsModel().fit(
                    d, self.y_qo_col, self.y_qg_col, self.y_qw_col
                )

        # ==================================================
        # 4. Helper to build ML residual dataset
        # ==================================================
        def _build_residual_dataset(df_in: pd.DataFrame):
            """
            Construct input/output pairs for ML residual learning.

            Inputs:
            - Independent variables
            - Physics model predictions

            Outputs:
            - Log-space residuals between measured and physics-predicted rates
            """

            df_lag = self._create_lagged_features(df_in)

            X_all, y_all = [], []

            model_input_cols = (
                self.independent_vars +
                [self.y_qo_col, self.y_qw_col, self.y_qg_col]
            )

            min_rows = max(5, self.lags + 3)

            for wid, d in df_lag.groupby(self.well_id_col):

                if wid not in self.phys_models:
                    continue

                # IMPORTANT: drop rows with NaNs in any required column
                d = d.dropna(subset=model_input_cols)

                if len(d) < min_rows:
                    print(f"[SKIP] Well {wid}: only {len(d)} usable rows")
                    continue

                phys = self.phys_models[wid].predict(d)

                X = np.column_stack([
                    d[self.independent_vars].values,
                    phys[["qo_pred", "qw_pred", "qg_pred"]].values,
                ])

                y_true = np.maximum(
                    d[[self.y_qo_col, self.y_qw_col, self.y_qg_col]].values, EPS
                )
                y_phys = np.maximum(phys.values, EPS)

                # Residuals in log-space (stabilizes scale)
                y_res = np.log1p(y_true) - np.log1p(y_phys)

                mask = np.isfinite(y_res).all(axis=1)
                if mask.any():
                    X_all.append(X[mask])
                    y_all.append(y_res[mask])

            if not X_all:
                raise ValueError(
                    "No valid residual training data found "
                    "(check lag count, split size, or NaNs)"
                )

            return np.vstack(X_all), np.vstack(y_all)

        # ==================================================
        # 5. Build TRAIN residual dataset
        # ==================================================
        X_train, Y_train = _build_residual_dataset(df)

        Xs_train = self.scaler.fit_transform(X_train)
        Xp_train = self.poly.fit_transform(Xs_train)

        if not np.isfinite(Y_train).all():
            raise ValueError("NaNs or infs in training residual targets")

        # ==================================================
        # 6. Fit ML residual model (GLOBAL)
        # ==================================================
        self.ml_residual.fit(Xp_train, Y_train)

        # ==================================================
        # 7. Optional validation diagnostics (unchanged)
        # ==================================================
        if df_val is not None:
            X_val, Y_val = _build_residual_dataset(df_val)
            Xs_val = self.scaler.transform(X_val)
            Xp_val = self.poly.transform(Xs_val)

            Y_val_pred = self.ml_residual.predict(Xp_val)
            rmse_val = np.sqrt(mean_squared_error(Y_val, Y_val_pred))
            print(f"[Validation] Residual RMSE = {rmse_val:.4f}")

        return self


    def predict_physics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Physics-only prediction (no ML, no lagging).

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe with well_id_col present.

        Returns
        -------
        pd.DataFrame
            Physics-only predictions with columns:
            ['qo_pred', 'qw_pred', 'qg_pred']
            Index is aligned with input df.
        """
        outputs = []

        for wid, d in df.groupby(self.well_id_col, sort=False):
            if wid not in self.phys_models:
                raise KeyError(f"No physics model found for well '{wid}'")

            phys = self.phys_models[wid]
            p = phys.predict(d)

            outputs.append(p)

        return pd.concat(outputs).sort_index()


    # --------------------------------------------------
    # Predict from hybrid model
    # --------------------------------------------------
    def predict_hybrid(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Hybrid prediction with:
        - Oil frozen to physics
        - Water residual learning with regime gating
        - Gas residual learning always on
        """

        # Threshold below which we consider the well "dry"
        WATER_GATE_THRESHOLD = 2.0  # Sm3/h (tune if needed)

        df_lag = self._create_lagged_features(df)
        outputs = []

        for wid, d in df_lag.groupby(self.well_id_col, sort=False):

            if wid not in self.phys_models:
                raise KeyError(f"No physics model found for well '{wid}'")

            # -------------------------------
            # Physics prediction
            # -------------------------------
            phys = self.phys_models[wid].predict(d)

            # -------------------------------
            # ML features
            # -------------------------------
            X = np.column_stack([
                d[self.independent_vars].values,
                phys[["qo_pred", "qw_pred", "qg_pred"]].values,
            ])

            Xp = self.poly.transform(self.scaler.transform(X))
            res = self.ml_residual.predict(Xp)

            # -------------------------------
            # Oil: gated ML correction
            # -------------------------------
            qo_phys = np.maximum(phys["qo_pred"].values, EPS)
            qo = qo_phys.copy()

            # Example gate 1: high water cut
            wc = phys["qw_pred"].values / np.maximum(qo_phys + phys["qw_pred"].values, EPS)
            oil_gate = wc > 0.3   # tune threshold

            # Example gate 2: large residual magnitude
            oil_gate |= np.abs(res[:, 0]) > 0.15  # log-space threshold

            qo[oil_gate] = np.expm1(
                np.log1p(qo_phys[oil_gate]) + res[oil_gate, 0]
            )

            qo = np.maximum(qo, 0.0)

            # -------------------------------
            # Gas: physics + ML residual
            # -------------------------------
            qg = np.expm1(
                np.log1p(np.maximum(phys["qg_pred"].values, EPS)) + res[:, 2]
            )
            qg = np.maximum(qg, 0.0)

            # -------------------------------
            # Water: regime-gated ML
            # -------------------------------
            qw_phys = np.maximum(phys["qw_pred"].values, 0.0)
            qw = np.zeros_like(qw_phys)

            # Only allow ML correction when water is already flowing
            mask = qw_phys > WATER_GATE_THRESHOLD

            qw[mask] = np.expm1(
                np.log1p(qw_phys[mask]) + res[mask, 1]
            )

            qw = np.maximum(qw, 0.0)

            # -------------------------------
            # Liquid consistency
            # -------------------------------
            qL = qo + qw
            qw = np.clip(qw, 0.0, qL)
            qo = np.maximum(0.0, qL - qw)

            outputs.append(pd.DataFrame(
                {
                    "qo_pred": qo,
                    "qw_pred": qw,
                    "qg_pred": qg,
                },
                index=d.index,
            ))

        return pd.concat(outputs).sort_index()

    # --------------------------------------------------
    # Physics-only score
    # --------------------------------------------------
    def score_physics(
        self,
        df
    ):
        results = {}

        for wid, d in df.groupby(self.well_id_col):
            p = self.phys_models[wid].predict(d)

            # -----------------------------
            # WGR
            # -----------------------------
            y_wgr = compute_wgr(
                d[self.y_qw_col].values,
                d[self.y_qg_col].values
            )
            p_wgr = compute_wgr(
                p["qw_pred"].values,
                p["qg_pred"].values
            )

            mask_wgr = np.isfinite(y_wgr) & np.isfinite(p_wgr)

            # -----------------------------
            # GOR
            # -----------------------------
            y_gor = compute_gor(
                d[self.y_qg_col].values,
                d[self.y_qo_col].values
            )
            p_gor = compute_gor(
                p["qg_pred"].values,
                p["qo_pred"].values
            )

            mask_gor = np.isfinite(y_gor) & np.isfinite(p_gor)

            results[wid] = {
                "qo": dict(zip(
                    METRICS,
                    regression_metrics(d[self.y_qo_col], p["qo_pred"])
                )),
                "qw": dict(zip(
                    METRICS,
                    regression_metrics(d[self.y_qw_col], p["qw_pred"])
                )),
                "qg": dict(zip(
                    METRICS,
                    regression_metrics(d[self.y_qg_col], p["qg_pred"])
                )),
                "wgr": (
                    dict(zip(
                        METRICS,
                        regression_metrics(y_wgr[mask_wgr], p_wgr[mask_wgr])
                    ))
                    if mask_wgr.sum() >= 2
                    else {m: np.nan for m in METRICS}
                ),
                "gor": (
                    dict(zip(
                        METRICS,
                        regression_metrics(y_gor[mask_gor], p_gor[mask_gor])
                    ))
                    if mask_gor.sum() >= 2
                    else {m: np.nan for m in METRICS}
                ),
            }

        return results


    # --------------------------------------------------
    # Hybrid score (per well)
    # --------------------------------------------------
    def score_hybrid(
        self,
        df
    ):
        results = {}

        # IMPORTANT: create lagged features once
        df_lag = self._create_lagged_features(df)

        # Predict once (hybrid model is global)
        pred = self.predict_hybrid(df)

        # Attach predictions to dataframe
        df_pred = df_lag.copy()
        for col in pred.columns:
            df_pred[col] = pred[col].values

        for wid, d in df_pred.groupby(self.well_id_col):

            # -----------------------------
            # WGR
            # -----------------------------
            y_wgr = compute_wgr(
                d[self.y_qw_col].values,
                d[self.y_qg_col].values
            )
            p_wgr = compute_wgr(
                d["qw_pred"].values,
                d["qg_pred"].values
            )

            mask_wgr = np.isfinite(y_wgr) & np.isfinite(p_wgr)

            # -----------------------------
            # GOR
            # -----------------------------
            y_gor = compute_gor(
                d[self.y_qg_col].values,
                d[self.y_qo_col].values
            )
            p_gor = compute_gor(
                d["qg_pred"].values,
                d["qo_pred"].values
            )

            mask_gor = np.isfinite(y_gor) & np.isfinite(p_gor)

            results[wid] = {
                "qo": dict(zip(
                    METRICS,
                    regression_metrics(d[self.y_qo_col], d["qo_pred"])
                )),
                "qw": dict(zip(
                    METRICS,
                    regression_metrics(d[self.y_qw_col], d["qw_pred"])
                )),
                "qg": dict(zip(
                    METRICS,
                    regression_metrics(d[self.y_qg_col], d["qg_pred"])
                )),
                "wgr": (
                    dict(zip(
                        METRICS,
                        regression_metrics(y_wgr[mask_wgr], p_wgr[mask_wgr])
                    ))
                    if mask_wgr.sum() >= 2
                    else {m: np.nan for m in METRICS}
                ),
                "gor": (
                    dict(zip(
                        METRICS,
                        regression_metrics(y_gor[mask_gor], p_gor[mask_gor])
                    ))
                    if mask_gor.sum() >= 2
                    else {m: np.nan for m in METRICS}
                ),
            }

        return results
    
    def score_mpfm(
        self,
        df
    ):
        """
        Compute MPFM performance metrics with respect to well test (reference),
        using the same output format as score_hybrid.
        """

        results = {}

        # IMPORTANT: create lagged features once (for alignment consistency)
        df_lag = self._create_lagged_features(df)

        for wid, d in df_lag.groupby(self.well_id_col):

            # -----------------------------
            # WGR
            # -----------------------------
            y_wgr = compute_wgr(
                d[self.y_qw_col].values,
                d[self.y_qg_col].values
            )
            p_wgr = compute_wgr(
                d[self.mpfm_qw_col].values,
                d[self.mpfm_qg_col].values
            )

            mask_wgr = np.isfinite(y_wgr) & np.isfinite(p_wgr)

            # -----------------------------
            # GOR
            # -----------------------------
            y_gor = compute_gor(
                d[self.y_qg_col].values,
                d[self.y_qo_col].values
            )
            p_gor = compute_gor(
                d[self.mpfm_qg_col].values,
                d[self.mpfm_qo_col].values
            )

            mask_gor = np.isfinite(y_gor) & np.isfinite(p_gor)

            results[wid] = {
                "qo": dict(zip(
                    METRICS,
                    regression_metrics(d[self.y_qo_col], d[self.mpfm_qo_col])
                )),
                "qw": dict(zip(
                    METRICS,
                    regression_metrics(d[self.y_qw_col], d[self.mpfm_qw_col])
                )),
                "qg": dict(zip(
                    METRICS,
                    regression_metrics(d[self.y_qg_col], d[self.mpfm_qg_col])
                )),
                "wgr": (
                    dict(zip(
                        METRICS,
                        regression_metrics(
                            y_wgr[mask_wgr],
                            p_wgr[mask_wgr]
                        )
                    ))
                    if mask_wgr.sum() >= 2
                    else {m: np.nan for m in METRICS}
                ),
                "gor": (
                    dict(zip(
                        METRICS,
                        regression_metrics(
                            y_gor[mask_gor],
                            p_gor[mask_gor]
                        )
                    ))
                    if mask_gor.sum() >= 2
                    else {m: np.nan for m in METRICS}
                ),
            }

        return results



    # --------------------------------------------------
    # Dense fill
    # --------------------------------------------------
    def generate_dense_well_rates(self, df):
        mask = df[self.dependant_vars].isna().any(axis=1)
        if not mask.any():
            return df

        pred = self.predict_hybrid(df.loc[mask])
        for col in self.dependant_vars:
            df.loc[mask, col] = pred[f"{col.split('_')[0]}_pred"].values
        return df

    # --------------------------------------------------
    # Save / Load
    # --------------------------------------------------
    def save(self, directory):
        os.makedirs(directory, exist_ok=True)
        joblib.dump(self.phys_models, f"{directory}/physics.pkl")
        joblib.dump(self.scaler, f"{directory}/scaler.pkl")
        joblib.dump(self.poly, f"{directory}/poly.pkl")
        joblib.dump(self.ml_residual, f"{directory}/ml.pkl")

    @classmethod
    def load(cls, directory, **kwargs):
        model = cls(**kwargs)
        model.phys_models = joblib.load(f"{directory}/physics.pkl")
        model.scaler = joblib.load(f"{directory}/scaler.pkl")
        model.poly = joblib.load(f"{directory}/poly.pkl")
        model.ml_residual = joblib.load(f"{directory}/ml.pkl")
        return model

    # --------------------------------------------------
    # Plotting
    # --------------------------------------------------
    def plot_predictions(
        self,
        df: pd.DataFrame,
        is_hybrid_model: bool = True,
        model_tag_prefix: str = None,
        plot_ratios: bool = False
    ):
        """
        Plot predictions using DatetimeIndex as X-axis and save plots.

        Comparison:
        - Well Test rates → reference / ground truth
        - MPFM rates → baseline measurement system
        - Model predictions → physics or hybrid VFM
        """

        # --------------------------------------------------
        # Safety check
        # --------------------------------------------------
        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError("DataFrame index must be a DatetimeIndex")

        # --------------------------------------------------
        # Output directory
        # --------------------------------------------------
        out_dir = os.path.join("doc", "4-thesis", "images", "results")
        os.makedirs(out_dir, exist_ok=True)

        model_tag = "hybrid" if is_hybrid_model else "physics"
        if model_tag_prefix:
            model_tag = f"{model_tag_prefix}_{model_tag}"

        # --------------------------------------------------
        # Plot styling (publication-quality, color-blind safe)
        # --------------------------------------------------
        COLORS = {
            "reference": "#000000",  # Well Test
            "mpfm": "#1f77b4",       # MPFM
            "hybrid": "#d62728",     # Hybrid prediction
            "physics": "#7f7f7f",    # Physics-only
        }

        pred_color = COLORS["hybrid"] if is_hybrid_model else COLORS["physics"]

        for wid in df[self.well_id_col].unique():
            d = df[df[self.well_id_col] == wid].sort_index()

            # --------------------------------------------------
            # Lagged data for alignment
            # --------------------------------------------------
            d_lag = self._create_lagged_features(d)
            if d_lag.empty:
                continue

            # --------------------------------------------------
            # Predictions
            # --------------------------------------------------
            p = (
                self.predict_hybrid(d_lag)
                if is_hybrid_model
                else self.predict_physics(d_lag)
            )

            x = d_lag.index

            # --------------------------------------------------
            # Rate plots: qo, qw, qg
            # --------------------------------------------------
            rate_cfg = {
                "qo": {
                    "truth": self.y_qo_col,
                    "mpfm": self.mpfm_qo_col,
                    "pred": "qo_pred",
                    "label": "Oil rate",
                },
                "qw": {
                    "truth": self.y_qw_col,
                    "mpfm": self.mpfm_qw_col,
                    "pred": "qw_pred",
                    "label": "Water rate",
                },
                "qg": {
                    "truth": self.y_qg_col,
                    "mpfm": self.mpfm_qg_col,
                    "pred": "qg_pred",
                    "label": "Gas rate",
                },
            }

            for rate, cfg in rate_cfg.items():
                fig, ax = plt.subplots(figsize=(10, 4))

                ax.plot(
                    x,
                    d_lag[cfg["truth"]].values,
                    label="Well Test (Reference)",
                    linewidth=2.5,
                    marker="o",
                    markersize=4,
                    color=COLORS["reference"],
                )

                if cfg["mpfm"] in d_lag.columns:
                    ax.plot(
                        x,
                        d_lag[cfg["mpfm"]].values,
                        label="MPFM",
                        linewidth=2,
                        linestyle="--",
                        marker="x",
                        markersize=4,
                        color=COLORS["mpfm"],
                    )

                ax.plot(
                    x,
                    p[cfg["pred"]].values,
                    label="Hybrid (Predicted)"
                    if is_hybrid_model
                    else "Physics Prediction",
                    linewidth=2,
                    marker="o",
                    markersize=4,
                    color=pred_color,
                )

                ax.set_xlabel("Time")
                ax.set_ylabel(f"{rate} (Sm$^3$/h)")
                ax.set_title(f"{wid} : {cfg['label']}")
                ax.legend()
                ax.grid(True, alpha=0.3)

                ax.xaxis.set_major_locator(mdates.AutoDateLocator())
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
                fig.autofmt_xdate()
                plt.tight_layout()

                fname = f"{wid}_{rate}_{model_tag}.png"
                fig.savefig(os.path.join(out_dir, fname), dpi=300)

                plt.show()
                plt.close(fig)

            if plot_ratios:
                # --------------------------------------------------
                # WGR plot (Well Test + MPFM + Predicted)
                # --------------------------------------------------
                y_wgr = compute_wgr(
                    d_lag[self.y_qw_col].values,
                    d_lag[self.y_qg_col].values,
                )

                p_wgr = compute_wgr(
                    p["qw_pred"].values,
                    p["qg_pred"].values,
                )

                mpfm_wgr = (
                    d_lag[self.mpfm_wgr_col].values
                    if self.mpfm_wgr_col in d_lag.columns
                    else None
                )

                mask = np.isfinite(y_wgr) & np.isfinite(p_wgr)
                if mpfm_wgr is not None:
                    mask &= np.isfinite(mpfm_wgr)

                if mask.sum() >= 2:
                    fig, ax = plt.subplots(figsize=(10, 4))

                    ax.plot(
                        x[mask],
                        y_wgr[mask],
                        label="WGR (Well Test)",
                        linewidth=2.5,
                        marker="o",
                        color=COLORS["reference"],
                    )

                    if mpfm_wgr is not None:
                        ax.plot(
                            x[mask],
                            mpfm_wgr[mask],
                            label="WGR (MPFM)",
                            linewidth=2,
                            linestyle="--",
                            marker="x",
                            color=COLORS["mpfm"],
                        )

                    ax.plot(
                        x[mask],
                        p_wgr[mask],
                        label="WGR (Predicted)",
                        linewidth=2,
                        marker="o",
                        color=pred_color,
                    )

                    ax.set_xlabel("Time")
                    ax.set_ylabel("WGR (qw / qg)")
                    ax.set_title(f"{wid} : WGR")
                    ax.legend()
                    ax.grid(True, alpha=0.3)

                    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
                    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
                    fig.autofmt_xdate()
                    plt.tight_layout()

                    fig.savefig(os.path.join(out_dir, f"{wid}_wgr_{model_tag}.png"), dpi=300)
                    plt.show()
                    plt.close(fig)

                # --------------------------------------------------
                # GOR plot (Well Test + MPFM + Predicted)
                # --------------------------------------------------
                y_gor = compute_gor(
                    d_lag[self.y_qg_col].values,
                    d_lag[self.y_qo_col].values,
                )

                p_gor = compute_gor(
                    p["qg_pred"].values,
                    p["qo_pred"].values,
                )

                mpfm_gor = (
                    d_lag[self.mpfm_gor_col].values
                    if self.mpfm_gor_col in d_lag.columns
                    else None
                )

                mask = np.isfinite(y_gor) & np.isfinite(p_gor)
                if mpfm_gor is not None:
                    mask &= np.isfinite(mpfm_gor)

                if mask.sum() >= 2:
                    fig, ax = plt.subplots(figsize=(10, 4))

                    ax.plot(
                        x[mask],
                        y_gor[mask],
                        label="GOR (Well Test)",
                        linewidth=2.5,
                        marker="o",
                        color=COLORS["reference"],
                    )

                    if mpfm_gor is not None:
                        ax.plot(
                            x[mask],
                            mpfm_gor[mask],
                            label="GOR (MPFM)",
                            linewidth=2,
                            linestyle="--",
                            marker="x",
                            color=COLORS["mpfm"],
                        )

                    ax.plot(
                        x[mask],
                        p_gor[mask],
                        label="GOR (Predicted)",
                        linewidth=2,
                        marker="o",
                        color=pred_color,
                    )

                    ax.set_xlabel("Time")
                    ax.set_ylabel("GOR (qg / qo)")
                    ax.set_title(f"{wid} : GOR")
                    ax.legend()
                    ax.grid(True, alpha=0.3)

                    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
                    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
                    fig.autofmt_xdate()
                    plt.tight_layout()

                    fig.savefig(os.path.join(out_dir, f"{wid}_gor_{model_tag}.png"), dpi=300)
                    plt.show()
                    plt.close(fig)

    def calibrate_physics_only(
        self,
        df: pd.DataFrame
    ):
        """
        Calibrate physics model for a new (unseen) well
        without retraining the ML residual model.
        """

        for wid, d in df.groupby(self.well_id_col):
            geom = self.well_geometry.get(wid)

            self.phys_models[wid] = PhysicsModel(
                geometry=geom,
                global_params=None,  # IMPORTANT: do not update global priors
            ).fit(d, self.y_qo_col, self.y_qg_col, self.y_qw_col)
