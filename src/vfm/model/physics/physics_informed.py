import numpy as np
import pandas as pd
import os, joblib
from scipy.optimize import least_squares
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import PolynomialFeatures

# -------------------------
# Helpers
# -------------------------
def logistic(x):
    x = np.clip(x, -50, 50)
    return 1 / (1 + np.exp(-x))

def regression_metrics(y_true, y_pred):
    """
    Compute R2, MAE, and RMSE.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return r2, mae, rmse


# -------------------------
# Physics-based model with Closure Law
# -------------------------
class PhysicsModel:
    P_SCALE = 1e7   # pressure scaling (~100 bar in Pa)
    T_SCALE = 300.0 # temperature scaling (K)

    def __init__(self, estimate_pres_offset=1e6, fit_pres=True):
        self.fit_pres = fit_pres
        self.estimate_pres_offset = estimate_pres_offset
        self.params_ = None

    def _feature_matrix_for_wc(self, df):
        choke = df["choke"].values
        dcp   = df["dcp"].values / self.P_SCALE
        dht   = df["dht"].values / self.T_SCALE
        wht   = df["wht"].values / self.T_SCALE

        X = np.vstack([
            np.ones(len(df)),
            choke,
            dcp,
            dht,
            wht,
            choke**2,
            dcp**2,
            dht**2,
            wht**2,
            choke * dcp,
            choke * dht,
            choke * wht,
            dcp * dht,
            dcp * wht,
            dht * wht
        ]).T
        return X

    def _pack_params(self, x, n_wc_features):
        if self.fit_pres:
            P_res, qL_max, Cg = x[0], x[1], x[2]
            A = x[3:3+n_wc_features]
        else:
            P_res = None
            qL_max, Cg = x[0], x[1]
            A = x[2:2+n_wc_features]
        return P_res, qL_max, Cg, A

    def residuals(self, x, df, y_qo, y_qg, y_qw):
        P_res, qL_max, Cg, A = self._pack_params(x, 15)

        if P_res is None:
            P_res = df["dhp"].max() + self.estimate_pres_offset

        Pwf = df["dhp"].values

        # ---- Liquid (dimensionless IPR) ----
        pr = np.clip(Pwf / P_res, 0.0, 1.2)
        qL = qL_max * (1 - 0.2*pr - 0.8*pr**2)
        qL = np.maximum(0.0, qL)

        # ---- Water cut ----
        wc = logistic(self._feature_matrix_for_wc(df) @ A)
        qw_pred = wc * qL
        qo_pred = (1 - wc) * qL

        # ---- Gas (scaled pressure drawdown) ----
        dp = np.sqrt(np.maximum(0.0, P_res**2 - Pwf**2)) / self.P_SCALE
        qg_pred = Cg * dp

        eps = 1e-8
        r_qo = (qo_pred - y_qo) / max(np.std(y_qo), eps)
        r_qw = (qw_pred - y_qw) / max(np.std(y_qw), eps)
        r_qg = (qg_pred - y_qg) / max(np.std(y_qg), eps)

        return np.concatenate([r_qo, r_qw, r_qg])

    def fit(self, df, y_qo_col, y_qg_col, y_qw_col):
        df_fit = df[
            ["dhp","dht","whp","wht","choke","dcp",
             y_qo_col, y_qg_col, y_qw_col]
        ].dropna()

        y_qo = df_fit[y_qo_col].values
        y_qg = df_fit[y_qg_col].values
        y_qw = df_fit[y_qw_col].values

        dhp_max = df_fit["dhp"].max()

        if self.fit_pres:
            x0 = [dhp_max * 1.1, np.mean(y_qo + y_qw), 100.0] + [0.0]*15
            lb = [dhp_max, 0.0, 0.0] + [-5]*15
            ub = [dhp_max * 1.5, 1e5, 1e5] + [5]*15
        else:
            x0 = [np.mean(y_qo + y_qw), 100.0] + [0.0]*15
            lb = [0.0, 0.0] + [-5]*15
            ub = [1e5, 1e5] + [5]*15

        res = least_squares(
            self.residuals,
            x0,
            bounds=(lb, ub),
            args=(df_fit, y_qo, y_qg, y_qw),
            max_nfev=20000,
        )

        P_res, qL_max, Cg, A = self._pack_params(res.x, 15)

        if P_res is None:
            P_res = dhp_max * 1.1

        self.params_ = {
            "P_res": P_res,
            "qL_max": qL_max,
            "Cg": Cg,
            "A": np.array(A),
        }
        return self

    def predict(self, df):
        if self.params_ is None:
            raise RuntimeError("Physics model not fitted")

        P_res = self.params_["P_res"]
        qL_max = self.params_["qL_max"]
        Cg = self.params_["Cg"]
        A = self.params_["A"]

        Pwf = df["dhp"].values
        pr = np.clip(Pwf / P_res, 0.0, 1.2)

        qL = qL_max * (1 - 0.2*pr - 0.8*pr**2)
        qL = np.maximum(0.0, qL)

        wc = logistic(self._feature_matrix_for_wc(df) @ A)
        qw = wc * qL
        qo = (1 - wc) * qL

        dp = np.sqrt(np.maximum(0.0, P_res**2 - Pwf**2)) / self.P_SCALE
        qg = Cg * dp

        return pd.DataFrame(
            {"qo_pred": qo, "qw_pred": qw, "qg_pred": qg, "wc_pred": wc},
            index=df.index,
        )

    def score(self, df, y_qo_col="qo_well_test", y_qg_col="qg_well_test", y_qw_col="qw_well_test"):
        pred = self.predict(df)

        r2_qo, mae_qo, rmse_qo = regression_metrics(df[y_qo_col], pred["qo_pred"])
        r2_qw, mae_qw, rmse_qw = regression_metrics(df[y_qw_col], pred["qw_pred"])
        r2_qg, mae_qg, rmse_qg = regression_metrics(df[y_qg_col], pred["qg_pred"])

        return {
            "qo": {"r2": r2_qo, "mae": mae_qo, "rmse": rmse_qo},
            "qw": {"r2": r2_qw, "mae": mae_qw, "rmse": rmse_qw},
            "qg": {"r2": r2_qg, "mae": mae_qg, "rmse": rmse_qg},
        }

    def save(self, path: str):
        """
        Save fitted physics model parameters.
        """
        if self.params_ is None:
            raise RuntimeError("Physics model is not fitted.")
        joblib.dump(self.params_, path)

    def load(self, path: str):
        """
        Load fitted physics model parameters.
        """
        self.params_ = joblib.load(path)
        return self


# -------------------------
# Hybrid Physics + ML Model with Water Cut Prediction
# -------------------------
class PhysicsInformedHybridModel:
    def __init__(self, dependant_vars: list[str], independent_vars: list[str], degree=2, lags=1, well_id_col: str = "well_id",):
        self.well_id_col = well_id_col
        self.phys_models: dict[str, PhysicsModel] = {}
        self.independent_vars = independent_vars
        self.dependant_vars = dependant_vars
        self.degree = degree
        self.lags = lags

        
        # PolynomialFeatures instance (will be fitted during fit())
        self.poly = PolynomialFeatures(degree=self.degree, include_bias=False)
        self.ml_qo = GradientBoostingRegressor(n_estimators=500, max_depth=6, learning_rate=0.05)
        self.ml_wc = GradientBoostingRegressor(n_estimators=500, max_depth=6, learning_rate=0.05)
        self.ml_qg = GradientBoostingRegressor(n_estimators=500, max_depth=6, learning_rate=0.05)

    def _create_lagged_features(self, df: pd.DataFrame, drop_na: bool = True) -> pd.DataFrame:
        """
        Create lagged features for safe columns only (dhp, whp).
        If drop_na is True (training mode) drop rows with NaNs so ML sees clean examples.
        If drop_na is False (prediction mode) keep rows — don't drop them.
        """
        df_lagged = df.copy()
        for lag in range(1, self.lags + 1):
            for col in ["dhp", "whp"]:
                df_lagged[f"{col}_lag{lag}"] = df_lagged[col].shift(lag)

        if drop_na:
            df_lagged = df_lagged.dropna()

        return df_lagged

    def _transform_features(self, df: pd.DataFrame):
        """
        Transform features with a pre-fitted polynomial transformer.
        Assumes self.poly was fitted previously in fit().
        """
        X = df[self.independent_vars].values
        return self.poly.transform(X)

    def fit(self, df, y_qo_col="qo_well_test", y_qg_col="qg_well_test", y_qw_col="qw_well_test"):

        # ---------- Fit physics model PER WELL ----------
        self.phys_models = {}

        for well_id, df_well in df.groupby(self.well_id_col):
            phys = PhysicsModel()
            phys.fit(df_well, y_qo_col, y_qg_col, y_qw_col)
            self.phys_models[well_id] = phys

        # ---------- ML training uses ALL wells ----------
        df_lagged = self._create_lagged_features(df, drop_na=True)

        pred_phys_list = []
        for well_id, df_well in df_lagged.groupby(self.well_id_col):
            pred = self.phys_models[well_id].predict(df_well)
            pred_phys_list.append(pred)

        pred_phys = pd.concat(pred_phys_list).sort_index()

        wc_actual = df_lagged[y_qw_col] / (df_lagged[y_qw_col] + df_lagged[y_qo_col] + 1e-8)

        res_qo = df_lagged[y_qo_col] - pred_phys["qo_pred"]
        res_wc = wc_actual - pred_phys["wc_pred"]
        res_qg = df_lagged[y_qg_col] - pred_phys["qg_pred"]

        X_train = df_lagged[self.independent_vars].values
        self.poly.fit(X_train)
        X_poly = self.poly.transform(X_train)

        self.ml_qo.fit(X_poly, res_qo)
        self.ml_wc.fit(X_poly, res_wc)
        self.ml_qg.fit(X_poly, res_qg)

        return self

    def predict(self, df):

        df_lagged = self._create_lagged_features(df, drop_na=False)

        pred_phys_list = []
        for well_id, df_well in df_lagged.groupby(self.well_id_col):
            if well_id not in self.phys_models:
                raise ValueError(f"No physics model found for well_id={well_id}")

            pred = self.phys_models[well_id].predict(df_well)
            pred_phys_list.append(pred)

        pred_phys = pd.concat(pred_phys_list).sort_index()

        X_poly = self._transform_features(df_lagged)

        pred_hybrid = pred_phys.copy()
        pred_hybrid["qo_pred"] += self.ml_qo.predict(X_poly)

        wc_corrected = pred_hybrid["wc_pred"] + self.ml_wc.predict(X_poly)
        wc_corrected = np.clip(wc_corrected, 0.0, 1.0)

        total_liquid = pred_hybrid["qo_pred"] + pred_hybrid["qw_pred"]
        total_liquid = np.where(total_liquid <= 0, 1e-8, total_liquid)

        pred_hybrid["qw_pred"] = wc_corrected * total_liquid
        pred_hybrid["qo_pred"] = (1 - wc_corrected) * total_liquid

        pred_hybrid["qg_pred"] += self.ml_qg.predict(X_poly)

        return pred_hybrid


    def hybrid_score(
        self,
        df,
        y_qo_col="qo_well_test",
        y_qg_col="qg_well_test",
        y_qw_col="qw_well_test",
    ):
        """
        Hybrid model performance.
        Returns per-well scores and global pooled scores.
        """

        pred = self.predict(df)

        start_idx = self.lags if self.lags > 0 else 0

        results = {}
        qo_all, qw_all, qg_all = [], [], []
        qo_p_all, qw_p_all, qg_p_all = [], [], []

        for well_id, df_well in df.groupby(self.well_id_col):
            pred_well = pred.loc[df_well.index]

            y_qo = df_well[y_qo_col].iloc[start_idx:]
            y_qw = df_well[y_qw_col].iloc[start_idx:]
            y_qg = df_well[y_qg_col].iloc[start_idx:]

            p_qo = pred_well["qo_pred"].iloc[start_idx:]
            p_qw = pred_well["qw_pred"].iloc[start_idx:]
            p_qg = pred_well["qg_pred"].iloc[start_idx:]

            results[well_id] = {
                "qo": dict(zip(
                    ["r2", "mae", "rmse"],
                    regression_metrics(y_qo, p_qo)
                )),
                "qw": dict(zip(
                    ["r2", "mae", "rmse"],
                    regression_metrics(y_qw, p_qw)
                )),
                "qg": dict(zip(
                    ["r2", "mae", "rmse"],
                    regression_metrics(y_qg, p_qg)
                )),
            }

            qo_all.append(y_qo); qo_p_all.append(p_qo)
            qw_all.append(y_qw); qw_p_all.append(p_qw)
            qg_all.append(y_qg); qg_p_all.append(p_qg)

        # ---- Global pooled score ----
        results["__global__"] = {
            "qo": dict(zip(
                ["r2", "mae", "rmse"],
                regression_metrics(pd.concat(qo_all), pd.concat(qo_p_all))
            )),
            "qw": dict(zip(
                ["r2", "mae", "rmse"],
                regression_metrics(pd.concat(qw_all), pd.concat(qw_p_all))
            )),
            "qg": dict(zip(
                ["r2", "mae", "rmse"],
                regression_metrics(pd.concat(qg_all), pd.concat(qg_p_all))
            )),
        }

        return results

    def physics_score(
        self,
        df,
        y_qo_col="qo_well_test",
        y_qg_col="qg_well_test",
        y_qw_col="qw_well_test",
    ):
        """
        Physics-only performance per well and globally.
        """

        results = {}
        qo_all, qw_all, qg_all = [], [], []
        qo_p_all, qw_p_all, qg_p_all = [], [], []

        for well_id, df_well in df.groupby(self.well_id_col):
            if well_id not in self.phys_models:
                raise ValueError(f"No physics model for well_id={well_id}")

            phys = self.phys_models[well_id]
            pred = phys.predict(df_well)

            y_qo = df_well[y_qo_col]
            y_qw = df_well[y_qw_col]
            y_qg = df_well[y_qg_col]

            p_qo = pred["qo_pred"]
            p_qw = pred["qw_pred"]
            p_qg = pred["qg_pred"]

            results[well_id] = {
                "qo": dict(zip(
                    ["r2", "mae", "rmse"],
                    regression_metrics(y_qo, p_qo)
                )),
                "qw": dict(zip(
                    ["r2", "mae", "rmse"],
                    regression_metrics(y_qw, p_qw)
                )),
                "qg": dict(zip(
                    ["r2", "mae", "rmse"],
                    regression_metrics(y_qg, p_qg)
                )),
            }

            qo_all.append(y_qo); qo_p_all.append(p_qo)
            qw_all.append(y_qw); qw_p_all.append(p_qw)
            qg_all.append(y_qg); qg_p_all.append(p_qg)

        results["__global__"] = {
            "qo": dict(zip(
                ["r2", "mae", "rmse"],
                regression_metrics(pd.concat(qo_all), pd.concat(qo_p_all))
            )),
            "qw": dict(zip(
                ["r2", "mae", "rmse"],
                regression_metrics(pd.concat(qw_all), pd.concat(qw_p_all))
            )),
            "qg": dict(zip(
                ["r2", "mae", "rmse"],
                regression_metrics(pd.concat(qg_all), pd.concat(qg_p_all))
            )),
        }

        return results


    def generate_dense_well_rates(self, df: pd.DataFrame) -> pd.DataFrame:
        df_dense = df.copy()
        targets = self.dependant_vars

        # Rows where at least one rate is missing
        mask_missing = df_dense[targets].isna().any(axis=1)
        if not mask_missing.any():
            print("No missing rates found.")
            return df_dense

        # Only rows with missing values (keep full index)
        df_missing = df_dense.loc[mask_missing]

        # Predict using hybrid model (predict will not drop rows)
        pred = self.predict(df_missing)

        # Build preds_df aligned to df_missing.index
        preds_df = pd.DataFrame({
            "qo_well_test": pred["qo_pred"].values,
            "qw_well_test": pred["qw_pred"].values,
            "qg_well_test": pred["qg_pred"].values,
        }, index=df_missing.index)

        # Fill missing values only (do not overwrite existing real values)
        for col in targets:
            df_dense.loc[df_dense[col].isna(), col] = preds_df.loc[df_dense[col].isna(), col]

        return df_dense
    
    def save(self, directory: str):
        os.makedirs(directory, exist_ok=True)

        joblib.dump(self.phys_models, os.path.join(directory, "physics_models.pkl"))
        joblib.dump(self.poly, os.path.join(directory, "poly.pkl"))
        joblib.dump(self.ml_qo, os.path.join(directory, "ml_qo.pkl"))
        joblib.dump(self.ml_wc, os.path.join(directory, "ml_wc.pkl"))
        joblib.dump(self.ml_qg, os.path.join(directory, "ml_qg.pkl"))

        meta = {
            "independent_vars": self.independent_vars,
            "dependant_vars": self.dependant_vars,
            "degree": self.degree,
            "lags": self.lags,
            "well_id_col": self.well_id_col,
        }
        joblib.dump(meta, os.path.join(directory, "meta.pkl"))


    @classmethod
    def load(cls, directory: str):
        meta = joblib.load(os.path.join(directory, "meta.pkl"))

        model = cls(
            dependant_vars=meta["dependant_vars"],
            independent_vars=meta["independent_vars"],
            degree=meta["degree"],
            lags=meta["lags"],
            well_id_col=meta["well_id_col"],
        )

        model.phys_models = joblib.load(os.path.join(directory, "physics_models.pkl"))
        model.poly = joblib.load(os.path.join(directory, "poly.pkl"))
        model.ml_qo = joblib.load(os.path.join(directory, "ml_qo.pkl"))
        model.ml_wc = joblib.load(os.path.join(directory, "ml_wc.pkl"))
        model.ml_qg = joblib.load(os.path.join(directory, "ml_qg.pkl"))

        return model
