
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from src.vfm.model.hybrid.constants import *
from src.vfm.utils.metrics_utils import *
from src.vfm.model.hybrid.base_physics_informed import BasePhysicsInformedHybridModel
from src.vfm.model.physics.generic_physics import MultiphasePhysicsModel


# =====================================================
# Helpers
# =====================================================
def compute_wgr(qw, qg, min_qg=50.0, eps=EPS):
    """
    Compute Water-Gas Ratio (WGR) in a numerically and physically safe manner.

    WGR is computed only when gas rate exceeds a minimum threshold
    to avoid unphysical ratios during low-gas or liquid-loading conditions.
    """
    qw = np.asarray(qw, dtype=float)
    qg = np.asarray(qg, dtype=float)

    wgr = np.full_like(qw, np.nan, dtype=float)

    mask = (
        (qg > min_qg) &
        np.isfinite(qw) &
        np.isfinite(qg) &
        (qg > eps)
    )

    wgr[mask] = qw[mask] / qg[mask]
    return wgr


def compute_gor(qg, qo, min_qo=5.0, eps=EPS):
    """
    Compute Gas-Oil Ratio (GOR) in a numerically and physically safe manner.

    GOR is computed only when oil rate exceeds a minimum threshold
    to avoid unphysical inflation under near-zero oil flow conditions.
    """
    qg = np.asarray(qg, dtype=float)
    qo = np.asarray(qo, dtype=float)

    gor = np.full_like(qg, np.nan, dtype=float)

    mask = (
        (qo > min_qo) &
        np.isfinite(qg) &
        np.isfinite(qo) &
        (qo > eps)
    )

    gor[mask] = qg[mask] / qo[mask]
    return gor


# =============================
# Physics-Informed Hybrid Model
# =============================
class PhysicsInformedHybridModel(BasePhysicsInformedHybridModel):

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

        self.rmse_val = None

    def _build_ml_features(
        self,
        df: pd.DataFrame,
        phys: pd.DataFrame,
    ):
        """
        Build ML feature matrix for residual learning.

        The ML model learns corrections to physics-based predictions.
        Features therefore consist of:
            - Physics-predicted phase rates
            - Operating conditions influencing residual behavior

        Notes
        -----
        - True flow rates are NOT included (to avoid leakage)
        - Well identifiers are NOT included (global model)
        - All features are numerical and time-aligned
        """

        X = pd.DataFrame(index=df.index)

        # --------------------------------------------------
        # Physics predictions (primary features)
        # --------------------------------------------------
        X["qo_phys"] = phys["qo_pred"].values
        X["qw_phys"] = phys["qw_pred"].values
        X["qg_phys"] = phys["qg_pred"].values

        # --------------------------------------------------
        # Operating conditions (only if present)
        # --------------------------------------------------
        for col in [
            "whp",    # wellhead pressure
            "dhp",    # downhole pressure
            "dcp",    # downstream choke pressure
            "choke",  # choke opening
            "wht",    # wellhead temperature
            "dht",    # downhole temperature
        ]:
            if col in df.columns:
                X[col] = df[col].values

        # --------------------------------------------------
        # Lagged operating conditions
        # --------------------------------------------------
        for lag in range(1, self.lags + 1):
            for col in ["dhp", "whp"]:
                lag_col = f"{col}_lag{lag}"
                if lag_col in df.columns:
                    X[lag_col] = df[lag_col].values


        # --------------------------------------------------
        # Numerical safety
        # --------------------------------------------------
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(method="ffill").fillna(method="bfill")

        return X


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

    # ==================================================
    # Helper to build ML residual dataset
    # ==================================================
    def _build_residual_dataset(
        self,
        df: pd.DataFrame,
        phys: pd.DataFrame,
    ):
        """
        Build residual-learning dataset for the hybrid model.

        Residual targets (learned by ML):
            - Δlog(qo)
            - Δlog(WGR)
            - Δlog(qg)
        """

        # --------------------------------------------------
        # ML features (independent variables + lags)
        # --------------------------------------------------
        X = self._build_ml_features(df, phys)

        # --------------------------------------------------
        # True values
        # --------------------------------------------------
        qo_true = df[self.y_qo_col].values
        qw_true = df[self.y_qw_col].values
        qg_true = df[self.y_qg_col].values

        # --------------------------------------------------
        # Physics predictions
        # --------------------------------------------------
        qo_phys = phys["qo_pred"].values
        qw_phys = phys["qw_pred"].values
        qg_phys = phys["qg_pred"].values

        # --------------------------------------------------
        # Numerical safety (PHYSICALLY CORRECT)
        # --------------------------------------------------
        qo_true = np.maximum(qo_true, EPS)
        qg_true = np.maximum(qg_true, EPS)
        qw_true = np.maximum(qw_true, 0.0)

        qo_phys = np.maximum(qo_phys, EPS)
        qg_phys = np.maximum(qg_phys, EPS)
        qw_phys = np.maximum(qw_phys, 0.0)

        # --------------------------------------------------
        # Water–Gas Ratio (WGR)
        # --------------------------------------------------
        wgr_true = qw_true / np.clip(qg_true, EPS, None)
        wgr_phys = qw_phys / np.clip(qg_phys, EPS, None)

        # --------------------------------------------------
        # Log-space residual targets
        # ORDER MATTERS:
        #   [qo_residual, wgr_residual, qg_residual]
        # --------------------------------------------------
        Y = np.column_stack([
            np.log1p(qo_true) - np.log1p(qo_phys),
            np.log1p(wgr_true) - np.log1p(wgr_phys),
            np.log1p(qg_true) - np.log1p(qg_phys),
        ])

        if not np.isfinite(Y).all():
            raise ValueError("Residual targets contain NaNs or infs")

        return X, Y


    # --------------------------------------------------
    # Fit
    # --------------------------------------------------
    def fit(
        self,
        df: pd.DataFrame,
        df_val: pd.DataFrame | None = None,
    ):
        """
        Fit the hybrid Physics–ML model.

        Steps:
        1. Fit a physics model per well
        2. Create lagged features for ML residual learning
        3. Build log-space residual targets [qo, WGR, qg]
        4. Train a global ML residual model
        5. Store feature ordering to prevent silent drift
        6. Optionally compute validation diagnostics
        """

        # --------------------------------------------------
        # 1. Fit PHYSICS models (per well)
        # --------------------------------------------------
        self.phys_models = {}

        for wid, d in df.groupby(self.well_id_col):
            model = MultiphasePhysicsModel(
                well_id=wid,
                geometry=self.well_geometry.get(wid)
            )
            model.fit(d, self.y_qo_col, self.y_qg_col, self.y_qw_col)
            self.phys_models[wid] = model

        # --------------------------------------------------
        # 2. Prepare TRAINING data for ML residual learning
        # --------------------------------------------------
        df_train_lag = self._create_lagged_features(df)
        df_train_lag = df_train_lag.dropna()

        if df_train_lag.empty:
            raise ValueError("No training rows left after lagging.")

        # Physics predictions on lagged data
        phys_train = self.predict_physics(df_train_lag)

        # --------------------------------------------------
        # 3. Build residual dataset (X, Y)
        # --------------------------------------------------
        X_train_df, Y_train = self._build_residual_dataset(
            df_train_lag,
            phys_train,
        )

        # --------------------------------------------------
        # 4. FIX FOR ISSUE 1: store ML feature ordering
        # --------------------------------------------------
        self._ml_feature_columns = X_train_df.columns.tolist()

        X_train = X_train_df[self._ml_feature_columns].values

        # --------------------------------------------------
        # 5. Scale + polynomial expansion
        # --------------------------------------------------
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_train_poly = self.poly.fit_transform(X_train_scaled)

        # --------------------------------------------------
        # 6. Train ML residual model
        # --------------------------------------------------
        self.ml_residual.fit(X_train_poly, Y_train)

        # --------------------------------------------------
        # 7. OPTIONAL: validation diagnostics
        # --------------------------------------------------
        self.rmse_val = None

        if df_val is not None:

            df_val_lag = self._create_lagged_features(df_val)
            df_val_lag = df_val_lag.dropna()

            if not df_val_lag.empty:

                phys_val = self.predict_physics(df_val_lag)

                X_val_df, Y_val = self._build_residual_dataset(
                    df_val_lag,
                    phys_val,
                )

                # Enforce identical feature ordering
                X_val = X_val_df[self._ml_feature_columns].values

                X_val_scaled = self.scaler.transform(X_val)
                X_val_poly = self.poly.transform(X_val_scaled)

                Y_val_pred = self.ml_residual.predict(X_val_poly)

                # Per-output RMSE (interpretable)
                rmse = np.sqrt(np.mean((Y_val - Y_val_pred) ** 2, axis=0))

                self.rmse_val = {
                    "qo": rmse[0],
                    "wgr": rmse[1],
                    "qg": rmse[2],
                }

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
        Hybrid prediction combining physics-based rates with ML residual corrections.

        Gas and oil rates are corrected directly in log-space.
        Water rate is reconstructed via ML-corrected Water-Gas Ratio (WGR)
        to improve robustness for gas wells with intermittent water production.
        """

        df_lag = self._create_lagged_features(df)

        # --------------------------------------------------
        # Physics predictions
        # --------------------------------------------------
        phys = self.predict_physics(df_lag)

        # --------------------------------------------------
        # ML residual prediction
        # --------------------------------------------------
        X_df = self._build_ml_features(df_lag, phys)

        # Enforce training feature order
        missing = set(self._ml_feature_columns) - set(X_df.columns)
        if missing:
            raise RuntimeError(f"Missing ML features at inference: {missing}")

        X = X_df[self._ml_feature_columns].values

        Xs = self.scaler.transform(X)
        Xp = self.poly.transform(Xs)

        res = self.ml_residual.predict(Xp)

        # --------------------------------------------------
        # OPTIONAL: residual magnitude regularization
        # --------------------------------------------------
        res = np.clip(res, -RES_CLIP, RES_CLIP)


        # ==================================================
        # GAS (unchanged)
        # ==================================================
        qg_phys = np.maximum(phys["qg_pred"].values, EPS)

        qg = np.expm1(
            np.log1p(qg_phys) + res[:, 2]
        )
        qg = np.maximum(qg, 0.0)

        # ==================================================
        # OIL (unchanged, gated if applicable)
        # ==================================================
        qo_phys = np.maximum(phys["qo_pred"].values, EPS)

        qo = np.expm1(
            np.log1p(qo_phys) + res[:, 0]
        )
        qo = np.maximum(qo, 0.0)

        # ==================================================
        # WATER (NEW: WGR-based reconstruction)
        # ==================================================
        qw_phys = np.maximum(phys["qw_pred"].values, 0.0)

        # Physics WGR
        wgr_phys = qw_phys / np.clip(qg_phys, EPS, None)

        # ML-corrected WGR (log-space)
        log_wgr = np.log1p(wgr_phys) + res[:, 1]
        wgr_hybrid = np.expm1(log_wgr)

        # Gating: ML water allowed only if physics predicts water
        mask = qw_phys > WATER_GATE_THRESHOLD

        qw = np.zeros_like(qg)
        qw[mask] = qg[mask] * wgr_hybrid[mask]
        qw = np.maximum(qw, 0.0)

        # ==================================================
        # LIQUID CONSISTENCY (preserved)
        # ==================================================
        qL = qo + qw
        qw = np.clip(qw, 0.0, qL)
        qo = np.maximum(0.0, qL - qw)

        # --------------------------------------------------
        # Output dataframe
        # --------------------------------------------------
        return pd.DataFrame(
            {
                "qo_pred": qo,
                "qw_pred": qw,
                "qg_pred": qg,
            },
            index=df_lag.index,
        )


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
            # Numerical safety
            # -----------------------------
            qw_true = np.maximum(d[self.y_qw_col].values, 0.0)
            qg_true = np.maximum(d[self.y_qg_col].values, EPS)
            qo_true = np.maximum(d[self.y_qo_col].values, EPS)

            qw_pred = np.maximum(p["qw_pred"].values, 0.0)
            qg_pred = np.maximum(p["qg_pred"].values, EPS)
            qo_pred = np.maximum(p["qo_pred"].values, EPS)

            # -----------------------------
            # WGR
            # -----------------------------
            y_wgr = compute_wgr(qw_true, qg_true)
            p_wgr = compute_wgr(qw_pred, qg_pred)

            mask_wgr = np.isfinite(y_wgr) & np.isfinite(p_wgr)

            # -----------------------------
            # GOR
            # -----------------------------
            y_gor = compute_gor(qg_true, qo_true)
            p_gor = compute_gor(qg_pred, qo_pred)

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
            # Numerical safety
            # -----------------------------
            qw_true = np.maximum(d[self.y_qw_col].values, 0.0)
            qg_true = np.maximum(d[self.y_qg_col].values, EPS)
            qo_true = np.maximum(d[self.y_qo_col].values, EPS)

            qw_pred = np.maximum(d["qw_pred"].values, 0.0)
            qg_pred = np.maximum(d["qg_pred"].values, EPS)
            qo_pred = np.maximum(d["qo_pred"].values, EPS)

            # -----------------------------
            # WGR
            # -----------------------------
            y_wgr = compute_wgr(qw_true, qg_true)
            p_wgr = compute_wgr(qw_pred, qg_pred)

            mask_wgr = np.isfinite(y_wgr) & np.isfinite(p_wgr)

            # -----------------------------
            # GOR
            # -----------------------------
            y_gor = compute_gor(qg_true, qo_true)
            p_gor = compute_gor(qg_pred, qo_pred)

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
        MAP = {
            self.y_qo_col: "qo_pred",
            self.y_qw_col: "qw_pred",
            self.y_qg_col: "qg_pred",
        }

        for col in self.dependant_vars:
            df.loc[mask, col] = pred[MAP[col]].values

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

            self.phys_models[wid] = MultiphasePhysicsModel(
                well_id=wid,
                geometry=geom,
                global_params=None,  # IMPORTANT: do not update global priors
            ).fit(d, self.y_qo_col, self.y_qg_col, self.y_qw_col)
