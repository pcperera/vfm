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

# =====================================================
# Constants & helpers
# =====================================================

EPS = 1e-6
METRICS = ["r2", "mae", "rmse", "mre"]

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

def logistic(x):
    x = np.clip(x, -50, 50)
    return 1 / (1 + np.exp(-x))

def regression_metrics(y_true, y_pred):
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
        df = df.dropna(subset=[y_qo, y_qg, y_qw])
        yqo, yqg, yqw = df[y_qo].values, df[y_qg].values, df[y_qw].values

        dhp_max = df["dhp"].max()
        qL_mean = np.mean(yqo + yqw)
        n_wc = 8

        x0 = [dhp_max * 1.1, qL_mean, 0.2, 0.6, 50.0, 10.0, 0.3] + [0.0] * n_wc
        lb = [dhp_max, 0, 0, 0, 0, 0.1, 0] + [-5]*n_wc
        ub = [dhp_max*2, 1e5, 1, 2, 1e5, 50, 1] + [5]*n_wc

        res = least_squares(
            self.residuals, x0, bounds=(lb, ub),
            args=(df, yqo, yqg, yqw), max_nfev=30000
        )

        P_res, qL_max, a, b, Cg, k_ch, ch0, A = self._unpack(res.x, n_wc)
        self.params_ = dict(
            P_res=P_res, qL_max=qL_max, a=a, b=b,
            Cg=Cg, k_choke=k_ch, choke0=ch0, A_wc=A
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
            GradientBoostingRegressor(
                n_estimators=100,        # ↓ drastically
                max_depth=2,             # ↓ shallow trees
                min_samples_leaf=30,     # ↑ strong smoothing
                learning_rate=0.05,
                subsample=0.7,
                random_state=42,
            )
        )


    # --------------------------------------------------
    # Lag features
    # --------------------------------------------------
    def _create_lagged_features(self, df, drop_na=True):
        df_lag = df.copy()
        for lag in range(1, self.lags + 1):
            for col in ["dhp", "whp"]:
                df_lag[f"{col}_lag{lag}"] = (
                    df_lag.groupby(self.well_id_col)[col].shift(lag)
                )
        return df_lag.dropna() if drop_na else df_lag

    # --------------------------------------------------
    # Fit
    # --------------------------------------------------
    def fit(self, df,
            y_qo_col="qo_mpfm",
            y_qg_col="qg_mpfm",
            y_qw_col="qw_mpfm"):

        # Physics per well
        self.phys_models = {}
        for wid, d in df.groupby(self.well_id_col):
            self.phys_models[wid] = PhysicsModel().fit(
                d, y_qo_col, y_qg_col, y_qw_col
            )

        # ML residual training
        df_lag = self._create_lagged_features(df)

        X_all, y_all = [], []

        for wid, d in df_lag.groupby(self.well_id_col):
            phys = self.phys_models[wid].predict(d)

            X = np.column_stack([
                d[self.independent_vars].values,
                phys[["qo_pred", "qw_pred", "qg_pred"]].values,
            ])

            y_true = d[[y_qo_col, y_qw_col, y_qg_col]].values
            y_phys = phys.values

            y_res = np.log1p(y_true) - np.log1p(np.maximum(y_phys, EPS))

            X_all.append(X)
            y_all.append(y_res)

        Xs = self.scaler.fit_transform(np.vstack(X_all))
        Xp = self.poly.fit_transform(Xs)

        self.ml_residual.fit(Xp, np.vstack(y_all))
        return self

    # --------------------------------------------------
    # Predict
    # --------------------------------------------------
    def predict(self, df):
        df_lag = self._create_lagged_features(df)
        outputs = []

        for wid, d in df_lag.groupby(self.well_id_col):
            phys = self.phys_models[wid].predict(d)

            X = np.column_stack([
                d[self.independent_vars].values,
                phys.values,
            ])

            Xp = self.poly.transform(self.scaler.transform(X))
            res = self.ml_residual.predict(Xp)

            y_hat = np.expm1(np.log1p(np.maximum(phys.values, EPS)) + res)

            qL = phys["qo_pred"].values + phys["qw_pred"].values
            qw = np.clip(y_hat[:,1], 0, qL)
            qo = np.maximum(0.0, qL - qw)
            qg = y_hat[:,2]

            outputs.append(pd.DataFrame(
                {"qo_pred": qo, "qw_pred": qw, "qg_pred": qg},
                index=d.index
            ))

        return pd.concat(outputs).sort_index()
    
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
    # Physics-only score
    # --------------------------------------------------
    def physics_score(self, df,
                      y_qo_col="qo_mpfm",
                      y_qg_col="qg_mpfm",
                      y_qw_col="qw_mpfm"):

        results = {}

        for wid, d in df.groupby(self.well_id_col):
            p = self.phys_models[wid].predict(d)

            y_wgr = compute_wgr(d[y_qw_col].values, d[y_qg_col].values)
            p_wgr = compute_wgr(p["qw_pred"].values, p["qg_pred"].values)

            results[wid] = {
                "qo": dict(zip(METRICS, regression_metrics(d[y_qo_col], p["qo_pred"]))),
                "qw": dict(zip(METRICS, regression_metrics(d[y_qw_col], p["qw_pred"]))),
                "qg": dict(zip(METRICS, regression_metrics(d[y_qg_col], p["qg_pred"]))),
                "wgr": dict(zip(METRICS, regression_metrics(y_wgr, p_wgr))),
            }

        return results

    # --------------------------------------------------
    # Hybrid score 
    # --------------------------------------------------
    def hybrid_score(
        self,
        df,
        y_qo_col="qo_mpfm",
        y_qg_col="qg_mpfm",
        y_qw_col="qw_mpfm",
    ):
        # IMPORTANT: use lagged dataframe
        df_lag = self._create_lagged_features(df)
        pred = self.predict(df)

        # Water–gas ratio (safe)
        y_wgr = compute_wgr(
            df_lag[y_qw_col].values,
            df_lag[y_qg_col].values
        )
        p_wgr = compute_wgr(
            pred["qw_pred"].values,
            pred["qg_pred"].values
        )

        return {
            "qo": dict(zip(
                METRICS,
                regression_metrics(df_lag[y_qo_col], pred["qo_pred"])
            )),
            "qw": dict(zip(
                METRICS,
                regression_metrics(df_lag[y_qw_col], pred["qw_pred"])
            )),
            "qg": dict(zip(
                METRICS,
                regression_metrics(df_lag[y_qg_col], pred["qg_pred"])
            )),
            "wgr": dict(zip(
                METRICS,
                regression_metrics(y_wgr, p_wgr)
            )),
        }


    # --------------------------------------------------
    # Dense fill
    # --------------------------------------------------
    def generate_dense_well_rates(self, df):
        mask = df[self.dependant_vars].isna().any(axis=1)
        if not mask.any():
            return df

        pred = self.predict(df.loc[mask])
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

            # IMPORTANT: lagged data for alignment
            d_lag = self._create_lagged_features(d)
            if d_lag.empty:
                continue

            # Predict
            p = (
                self.predict(d)
                if is_hybrid_model
                else self.predict_physics(d_lag)
            )

            # X-axis
            x = d_lag[time_col].values if time_col else np.arange(len(d_lag))

            plot_map = {
                "qo_pred": y_qo_col,
                "qw_pred": y_qw_col,
                "qg_pred": y_qg_col,
            }

            for pred_col, ycol in plot_map.items():
                plt.figure(figsize=(10, 4))

                # Actual values (line + markers)
                plt.plot(
                    x,
                    d_lag[ycol].values,
                    label="Actual",
                    # linestyle="--",
                    linewidth=2,
                    marker="o",
                    markersize=4,
                )

                # Predicted values (line + markers)
                plt.plot(
                    x,
                    p[pred_col].values,
                    label="Predicted",
                    # linestyle="--",
                    linewidth=2,
                    marker="o",
                    markersize=4,
                )

                plt.xlabel(time_col)
                rate_column = pred_col.replace("_pred", "")
                plt.ylabel(f"{rate_column} (Sm$^3$/h)")
                plt.title(f"{wid} : {rate_column}")

                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.show()
