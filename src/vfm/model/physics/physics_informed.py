import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import least_squares
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import HistGradientBoostingRegressor

# =====================================================
# Constants & helpers
# =====================================================

EPS = 1e-6
METRICS = ["r2", "mae", "rmse"]

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

    mask = np.isfinite(y_true) & np.isfinite(y_pred)

    if mask.sum() < 2:
        return {
            "r2": np.nan,
            "mae": np.nan,
            "rmse": np.nan,
            "mre": np.nan,
        }

    y_true = y_true[mask]
    y_pred = y_pred[mask]

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mre = np.mean(np.abs(y_true - y_pred) / np.maximum(np.abs(y_true), EPS)) * 100

    return r2, mae, rmse, mre


# =====================================================
# Physics Model
# =====================================================

class PhysicsModel:
    P_SCALE = 100.0       # ~100 bar
    T_SCALE = 100.0       # ~100 °C (scaling only)

    def __init__(self, estimate_pres_offset=10.0, fit_pres=True):
        self.fit_pres = fit_pres
        self.estimate_pres_offset = estimate_pres_offset
        self.params_ = None

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

    def _unpack(self, x, n_wc):
        i = 0
        P_res = x[i] if self.fit_pres else None; i += self.fit_pres
        qL_max, a, b = x[i:i+3]; i += 3
        Cg, k_ch, ch0 = x[i:i+3]; i += 3
        A_wc = x[i:i+n_wc]
        return P_res, qL_max, a, b, Cg, k_ch, ch0, A_wc

    def residuals(self, x, df, y_qo, y_qg, y_qw):
        n_wc = 8
        P_res, qL_max, a, b, Cg, k_ch, ch0, A = self._unpack(x, n_wc)

        if P_res is None:
            P_res = df["dhp"].max() + self.estimate_pres_offset

        Pwf = df["dhp"].values
        pr = np.clip(Pwf / P_res, 0, 1.5)

        qL = np.maximum(0.0, qL_max * (1 - a * pr - b * pr**2))
        wc = logistic(self._feature_matrix_for_wc(df) @ A)

        qw = wc * qL
        qo = (1 - wc) * qL

        dp = np.sqrt(np.maximum(0.0, P_res - Pwf)) / self.P_SCALE
        choke_eff = logistic(k_ch * (df["choke"].values - ch0))
        qg = Cg * dp * choke_eff

        return np.concatenate([
            (qo - y_qo) / max(np.std(y_qo), EPS),
            (qw - y_qw) / max(np.std(y_qw), EPS),
            (qg - y_qg) / max(np.std(y_qg), EPS),
        ])

    def fit(self, df, y_qo, y_qg, y_qw):
        """
        Calibrate physics model parameters for a single well.
        """

        # --------------------------------------------------
        # 1. Clean input data (CRITICAL)
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
        dhp_min = float(np.nanmin(dhp))

        if not np.isfinite(dhp_max):
            raise ValueError("Invalid DHP values for physics calibration")

        qL_mean = float(np.nanmean(yqo + yqw))
        qL_mean = max(qL_mean, EPS)

        # --------------------------------------------------
        # 3. Robust initial guess + bounds
        # --------------------------------------------------
        n_wc = 8

        # Reservoir pressure
        P_res0 = max(dhp_max + self.estimate_pres_offset, dhp_max * 1.05)
        P_res_lb = dhp_max * 1.01
        P_res_ub = dhp_max * 3.0

        # Initial parameter vector
        x0 = (
            [P_res0, qL_mean, 0.2, 0.5, 50.0, 5.0, 0.3]
            + [0.0] * n_wc
        )

        lb = (
            [P_res_lb, 0.0, 0.0, 0.0, 0.0, 0.01, 0.0]
            + [-5.0] * n_wc
        )

        ub = (
            [P_res_ub, 1e5, 1.0, 2.0, 1e5, 50.0, 1.0]
            + [5.0] * n_wc
        )

        # --------------------------------------------------
        # 4. Final numeric safety check
        # --------------------------------------------------
        x0 = np.asarray(x0, dtype=float)
        lb = np.asarray(lb, dtype=float)
        ub = np.asarray(ub, dtype=float)

        if not np.all((x0 > lb) & (x0 < ub)):
            raise ValueError("Initial guess x0 is not strictly within bounds")

        # --------------------------------------------------
        # 5. Least squares optimization
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

        # --------------------------------------------------
        # 6. Store calibrated parameters
        # --------------------------------------------------
        P_res, qL_max, a, b, Cg, k_ch, ch0, A = self._unpack(res.x, n_wc)

        self.params_ = dict(
            P_res=P_res,
            qL_max=qL_max,
            a=a,
            b=b,
            Cg=Cg,
            k_choke=k_ch,
            choke0=ch0,
            A_wc=A,
        )

        return self


    def predict(self, df):
        p = self.params_
        Pwf = df["dhp"].values
        pr = np.clip(Pwf / p["P_res"], 0, 1.5)

        qL = np.maximum(0.0, p["qL_max"] * (1 - p["a"]*pr - p["b"]*pr**2))
        wc = logistic(self._feature_matrix_for_wc(df) @ p["A_wc"])

        qw = wc * qL
        qo = (1 - wc) * qL

        dp = np.sqrt(np.maximum(0.0, p["P_res"] - Pwf)) / self.P_SCALE
        choke_eff = logistic(p["k_choke"] * (df["choke"].values - p["choke0"]))
        qg = p["Cg"] * dp * choke_eff

        return pd.DataFrame(
            {"qo_pred": qo, "qw_pred": qw, "qg_pred": qg},
            index=df.index
        )

# =============================
# Physics-Informed Hybrid Model
# =============================
class PhysicsInformedHybridModel:

    def __init__(self, dependant_vars, independent_vars,
                 well_id_col="well_id", degree=1, lags=1):

        self.dependant_vars = dependant_vars
        self.independent_vars = independent_vars
        self.well_id_col = well_id_col
        self.degree = degree
        self.lags = lags

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
        df,
        df_val=None,
        y_qo_col="qo_mpfm",
        y_qg_col="qg_mpfm",
        y_qw_col="qw_mpfm",
    ):
        """
        Fit physics models per well and a global ML residual model.
        """

        # --------------------------------------------------
        # 1. Fit physics models per well (TRAIN ONLY)
        # --------------------------------------------------
        self.phys_models = {}

        for wid, d in df.groupby(self.well_id_col):
            self.phys_models[wid] = PhysicsModel().fit(
                d, y_qo_col, y_qg_col, y_qw_col
            )

        # --------------------------------------------------
        # 2. Helper to build ML residual dataset
        # --------------------------------------------------
        def _build_residual_dataset(df_in):
            df_lag = self._create_lagged_features(df_in)

            X_all, y_all = [], []

            # Every column used in X MUST be finite
            model_input_cols = (
                self.independent_vars +
                [y_qo_col, y_qw_col, y_qg_col]
            )

            # Minimum rows needed AFTER lagging
            min_rows = max(5, self.lags + 3)

            for wid, d in df_lag.groupby(self.well_id_col):

                if wid not in self.phys_models:
                    continue

                # 🔑 CRITICAL FIX: drop NaNs for ALL used columns
                d = d.dropna(subset=model_input_cols)

                if len(d) < min_rows:
                    # Optional debug (leave enabled until stable)
                    print(f"[SKIP] Well {wid}: only {len(d)} usable rows")
                    continue

                phys = self.phys_models[wid].predict(d)

                X = np.column_stack([
                    d[self.independent_vars].values,
                    phys[["qo_pred", "qw_pred", "qg_pred"]].values,
                ])

                y_true = np.maximum(
                    d[[y_qo_col, y_qw_col, y_qg_col]].values, EPS
                )
                y_phys = np.maximum(phys.values, EPS)

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


        # --------------------------------------------------
        # 3. Build TRAIN residual dataset
        # --------------------------------------------------
        X_train, Y_train = _build_residual_dataset(df)

        Xs_train = self.scaler.fit_transform(X_train)
        Xp_train = self.poly.fit_transform(Xs_train)

        if not np.isfinite(Y_train).all():
            raise ValueError("NaNs or infs in training residual targets")

        # --------------------------------------------------
        # 4. Fit ML residual model (TRAIN ONLY)
        # --------------------------------------------------
        self.ml_residual.fit(Xp_train, Y_train)

        # --------------------------------------------------
        # 5. (Optional) Validation dataset — diagnostics only
        # --------------------------------------------------
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
        df,
        y_qo_col="qo_mpfm",
        y_qg_col="qg_mpfm",
        y_qw_col="qw_mpfm",
    ):
        results = {}

        for wid, d in df.groupby(self.well_id_col):
            p = self.phys_models[wid].predict(d)

            # -----------------------------
            # WGR
            # -----------------------------
            y_wgr = compute_wgr(
                d[y_qw_col].values,
                d[y_qg_col].values
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
                d[y_qg_col].values,
                d[y_qo_col].values
            )
            p_gor = compute_gor(
                p["qg_pred"].values,
                p["qo_pred"].values
            )

            mask_gor = np.isfinite(y_gor) & np.isfinite(p_gor)

            results[wid] = {
                "qo": dict(zip(
                    METRICS,
                    regression_metrics(d[y_qo_col], p["qo_pred"])
                )),
                "qw": dict(zip(
                    METRICS,
                    regression_metrics(d[y_qw_col], p["qw_pred"])
                )),
                "qg": dict(zip(
                    METRICS,
                    regression_metrics(d[y_qg_col], p["qg_pred"])
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
        df,
        y_qo_col="qo_mpfm",
        y_qg_col="qg_mpfm",
        y_qw_col="qw_mpfm",
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
                d[y_qw_col].values,
                d[y_qg_col].values
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
                d[y_qg_col].values,
                d[y_qo_col].values
            )
            p_gor = compute_gor(
                d["qg_pred"].values,
                d["qo_pred"].values
            )

            mask_gor = np.isfinite(y_gor) & np.isfinite(p_gor)

            results[wid] = {
                "qo": dict(zip(
                    METRICS,
                    regression_metrics(d[y_qo_col], d["qo_pred"])
                )),
                "qw": dict(zip(
                    METRICS,
                    regression_metrics(d[y_qw_col], d["qw_pred"])
                )),
                "qg": dict(zip(
                    METRICS,
                    regression_metrics(d[y_qg_col], d["qg_pred"])
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
        df,
        y_qo_col="qo_mpfm",
        y_qw_col="qw_mpfm",
        y_qg_col="qg_mpfm",
        time_col="time_idx",
        is_hybrid_model: bool = True,
    ):
        for wid in df[self.well_id_col].unique():
            d = df[df[self.well_id_col] == wid]

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
                self.predict_hybrid(d)
                if is_hybrid_model
                else self.predict_physics(d_lag)
            )

            # X-axis
            x = d_lag[time_col].values if time_col else np.arange(len(d_lag))

            # --------------------------------------------------
            # Rate plots: qo, qw, qg
            # --------------------------------------------------
            plot_map = {
                "qo_pred": y_qo_col,
                "qw_pred": y_qw_col,
                "qg_pred": y_qg_col,
            }

            for pred_col, ycol in plot_map.items():
                plt.figure(figsize=(10, 4))

                plt.plot(
                    x,
                    d_lag[ycol].values,
                    label="MPFM (Actual)",
                    linewidth=2,
                    marker="o",
                    markersize=4,
                )

                plt.plot(
                    x,
                    p[pred_col].values,
                    label="Predicted",
                    linewidth=2,
                    marker="o",
                    markersize=4,
                )

                plt.xlabel(time_col)
                rate_name = pred_col.replace("_pred", "")
                plt.ylabel(f"{rate_name} (Sm$^3$/h)")
                plt.title(f"{wid} : {rate_name}")

                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.show()

            # --------------------------------------------------
            # WGR plot (qw / qg)
            # --------------------------------------------------
            y_wgr = compute_wgr(
                d_lag[y_qw_col].values,
                d_lag[y_qg_col].values,
            )
            p_wgr = compute_wgr(
                p["qw_pred"].values,
                p["qg_pred"].values,
            )

            mask = np.isfinite(y_wgr) & np.isfinite(p_wgr)
            if mask.sum() >= 2:
                plt.figure(figsize=(10, 4))

                plt.plot(
                    x[mask],
                    y_wgr[mask],
                    label="WGR (Actual)",
                    linewidth=2,
                    marker="o",
                    markersize=4,
                )

                plt.plot(
                    x[mask],
                    p_wgr[mask],
                    label="WGR (Predicted)",
                    linewidth=2,
                    marker="o",
                    markersize=4,
                )

                plt.xlabel(time_col)
                plt.ylabel("wgr (qw / qg)")
                plt.title(f"{wid} : Water-Gas Ratio")

                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.show()

            # --------------------------------------------------
            # GOR plot (qg / qo)
            # --------------------------------------------------
            y_gor = compute_gor(
                d_lag[y_qg_col].values,
                d_lag[y_qo_col].values,
            )
            p_gor = compute_gor(
                p["qg_pred"].values,
                p["qo_pred"].values,
            )

            mask = np.isfinite(y_gor) & np.isfinite(p_gor)
            if mask.sum() >= 2:
                plt.figure(figsize=(10, 4))

                plt.plot(
                    x[mask],
                    y_gor[mask],
                    label="gor (Actual)",
                    linewidth=2,
                    marker="o",
                    markersize=4,
                )

                plt.plot(
                    x[mask],
                    p_gor[mask],
                    label="GOR (Predicted)",
                    linewidth=2,
                    marker="o",
                    markersize=4,
                )

                plt.xlabel(time_col)
                plt.ylabel("GOR (qg / qo)")
                plt.title(f"{wid} : Gas-Oil Ratio")

                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.show()
