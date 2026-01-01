
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
def compute_wgr(qw, qg, min_qg=50.0):
    """
    Compute Water–Gas Ratio (WGR) in a numerically and physically safe manner.

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
        (qg > EPS)
    )

    wgr[mask] = qw[mask] / qg[mask]
    return wgr


def compute_gor(qg, qo, min_qo=5.0):
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
        (qo > EPS)
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
        temp_phys_models: dict[str, MultiphasePhysicsModel] = {}

        for wid, d in df.groupby(self.well_id_col):
            try:
                temp_phys_models[wid] = MultiphasePhysicsModel(well_id=wid, geometry=self.well_geometry.get(wid)).fit(
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
                self.phys_models[wid] = MultiphasePhysicsModel(
                    well_id=wid,
                    geometry=geom,
                    global_params=global_physics_params,
                ).fit(d, self.y_qo_col, self.y_qg_col, self.y_qw_col)
            except Exception as e:
                # Fallback: original unconstrained physics model
                print(f"[WARN] Geometry-aware fit failed for well {wid}, falling back: {e}")
                self.phys_models[wid] = MultiphasePhysicsModel(well_id=wid).fit(
                    d, self.y_qo_col, self.y_qg_col, self.y_qw_col
                )

        # ==================================================
        # 4. Helper to build ML residual dataset
        # ==================================================
        def _build_residual_dataset(
            self,
            df: pd.DataFrame,
            phys: pd.DataFrame,
        ):
            """
            Build residual-learning dataset for the hybrid model.

            The ML model learns residuals in log-space relative to the physics model.
            For water, residual learning is performed on Water–Gas Ratio (WGR)
            instead of direct water rate to improve numerical stability and
            physical consistency for gas wells.

            Residual targets learned by ML:
                - Δlog(qo)
                - Δlog(WGR)
                - Δlog(qg)
            """

            # -------------------------------
            # True values
            # -------------------------------
            qo_true = df[self.y_qo_col].values
            qw_true = df[self.y_qw_col].values
            qg_true = df[self.y_qg_col].values

            # -------------------------------
            # Physics predictions
            # -------------------------------
            qo_phys = phys["qo_pred"].values
            qw_phys = phys["qw_pred"].values
            qg_phys = phys["qg_pred"].values

            # -------------------------------
            # Numerical safety
            # -------------------------------
            qo_true = np.maximum(qo_true, EPS)
            qg_true = np.maximum(qg_true, EPS)
            qw_true = np.maximum(qw_true, 0.0)

            qo_phys = np.maximum(qo_phys, EPS)
            qg_phys = np.maximum(qg_phys, EPS)
            qw_phys = np.maximum(qw_phys, 0.0)

            # -------------------------------
            # Water–Gas Ratio (WGR)
            # -------------------------------
            wgr_true = qw_true / np.clip(qg_true, EPS, None)
            wgr_phys = qw_phys / np.clip(qg_phys, EPS, None)

            # -------------------------------
            # Log-space residual targets
            # Order is IMPORTANT:
            #   [qo_residual, wgr_residual, qg_residual]
            # -------------------------------
            y_res = np.column_stack([
                np.log1p(qo_true) - np.log1p(qo_phys),
                np.log1p(wgr_true + WGR_EPS) - np.log1p(wgr_phys + WGR_EPS),
                np.log1p(qg_true) - np.log1p(qg_phys),
            ])

            return y_res


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
            self.rmse_val = np.sqrt(mean_squared_error(Y_val, Y_val_pred))
            print(f"[Validation] Residual RMSE = {self.rmse_val:.4f}")

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

        # --------------------------------------------------
        # Physics predictions
        # --------------------------------------------------
        phys = self.predict_physics(df)

        # --------------------------------------------------
        # ML residual prediction
        # --------------------------------------------------
        X = self._build_ml_features(df, phys)
        res = self.ml_residual.predict(X)

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
        log_wgr = np.log1p(wgr_phys + WGR_EPS) + res[:, 1]
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
            index=df.index,
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

            self.phys_models[wid] = MultiphasePhysicsModel(
                well_id=wid,
                geometry=geom,
                global_params=None,  # IMPORTANT: do not update global priors
            ).fit(d, self.y_qo_col, self.y_qg_col, self.y_qw_col)
