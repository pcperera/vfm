import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from sklearn.metrics import r2_score
from xgboost import XGBRegressor

# -------------------------
# Helpers
# -------------------------
def logistic(x):
    # Clip to avoid overflow
    x = np.clip(x, -50, 50)
    return 1 / (1 + np.exp(-x))

# -------------------------
# Physics-Informed Model
# -------------------------
class PhysicsModel:
    """
    Predicts oil, water, and gas rates using physics-informed surrogate:
      qL = J * (P_res - dhp)
      wc = logistic(A @ enhanced_features)
      qo = (1 - wc) * qL
      qw = wc * qL
      qg = Cg * sqrt(max(P_res^2 - dhp^2, 0))
    """
    def __init__(self, estimate_pres_offset=1.0, fit_pres=True):
        self.fit_pres = fit_pres
        self.estimate_pres_offset = estimate_pres_offset
        self.params_ = None

    def _feature_matrix_for_wc(self, df):
        choke = df["choke"].values
        dcp = df["dcp"].values
        dht = df["dht"].values
        wht = df["wht"].values

        # Include nonlinear and interaction terms
        features = np.vstack([
            np.ones(len(df)),        # intercept
            choke,
            dcp,
            dht,
            wht,
            choke**2,
            dcp**2,
            dht**2,
            wht**2,
            choke*dcp,
            choke*dht,
            choke*wht,
            dcp*dht,
            dcp*wht,
            dht*wht
        ]).T
        return features  # shape (n_samples, 15)

    def _pack_params(self, x, n_wc_features):
        if self.fit_pres:
            P_res = x[0]
            J = x[1]
            Cg = x[2]
            A = x[3:3 + n_wc_features]
        else:
            P_res = None
            J = x[0]
            Cg = x[1]
            A = x[2:2 + n_wc_features]
        return P_res, J, Cg, A

    def residuals(self, x, df, y_qo, y_qg, y_qw):
        P_res, J, Cg, A = self._pack_params(x, n_wc_features=15)
        if P_res is None:
            P_res = df["dhp"].max() + self.estimate_pres_offset

        qL = J * (P_res - df["dhp"].values)
        Xwc = self._feature_matrix_for_wc(df)
        wc = logistic(Xwc.dot(A))

        qo_pred = np.maximum(0.0, (1 - wc) * qL)
        qw_pred = np.maximum(0.0, wc * qL)
        inside = np.maximum(0.0, P_res**2 - df["dhp"].values**2)
        qg_pred = Cg * np.sqrt(inside)

        # scale residuals
        eps = 1e-8
        scale_qo = max(np.std(y_qo), eps)
        scale_qg = max(np.std(y_qg), eps)
        scale_qw = max(np.std(y_qw), eps)

        r_qo = (qo_pred - y_qo) / scale_qo
        r_qg = (qg_pred - y_qg) / scale_qg
        r_qw = (qw_pred - y_qw) / scale_qw

        return np.concatenate([r_qo, r_qg, r_qw])

    def fit(self, df, y_qo_col="qo_well_test", y_qg_col="qg_well_test", y_qw_col="qw_well_test"):
        # Drop NaNs
        cols = ["dhp", "dht", "whp", "wht", "choke", "dcp", y_qo_col, y_qg_col, y_qw_col]
        df_fit = df[cols].dropna().copy()
        if len(df_fit) == 0:
            raise ValueError("No data to fit after dropping NaNs")

        y_qo = df_fit[y_qo_col].values
        y_qg = df_fit[y_qg_col].values
        y_qw = df_fit[y_qw_col].values

        # initial guess and bounds
        if self.fit_pres:
            x0 = [df_fit["dhp"].max() + self.estimate_pres_offset, 0.5, 0.1] + [0.0]*15
            lb = [df_fit["dhp"].max(), 0.0, 0.0] + [-10]*15
            ub = [df_fit["dhp"].max()+1000, 100.0, 100.0] + [10]*15
        else:
            x0 = [0.5, 0.1] + [0.0]*15
            lb = [0.0, 0.0] + [-10]*15
            ub = [100.0, 100.0] + [10]*15

        res = least_squares(self.residuals, x0, args=(df_fit, y_qo, y_qg, y_qw),
                            bounds=(lb, ub), verbose=1, max_nfev=5000)

        P_res, J, Cg, A = self._pack_params(res.x, n_wc_features=15)
        if P_res is None:
            P_res = df_fit["dhp"].max() + self.estimate_pres_offset

        self.params_ = {"P_res": float(P_res), "J": float(J), "Cg": float(Cg),
                        "A": np.array(A), "success": res.success, "message": res.message}

        # Fit residual ML models
        df_pred = self.predict(df_fit)
        self.residual_models = {}
        features = df_fit[["dhp", "dht", "whp", "wht", "choke", "dcp"]]

        for target, residual_col in zip(
            ["qo", "qg", "qw"],
            [y_qo_col, y_qg_col, y_qw_col]
        ):
            residuals = df_fit[residual_col] - df_pred[f"{target}_pred"]
            model = XGBRegressor(n_estimators=200, max_depth=3, learning_rate=0.1)
            model.fit(features, residuals)
            self.residual_models[target] = model

        return self

    def predict(self, df):
        if self.params_ is None:
            raise RuntimeError("Model not fitted yet")

        P_res = self.params_["P_res"]
        J = self.params_["J"]
        Cg = self.params_["Cg"]
        A = self.params_["A"]

        qL = J * (P_res - df["dhp"].values)
        Xwc = self._feature_matrix_for_wc(df)
        wc = logistic(Xwc.dot(A))

        qo_pred = np.maximum(0.0, (1 - wc) * qL)
        qw_pred = np.maximum(0.0, wc * qL)
        inside = np.maximum(0.0, P_res**2 - df["dhp"].values**2)
        qg_pred = Cg * np.sqrt(inside)

        # Apply residual corrections if available
        features = df[["dhp", "dht", "whp", "wht", "choke", "dcp"]]
        if hasattr(self, "residual_models"):
            qo_pred += self.residual_models["qo"].predict(features)
            qg_pred += self.residual_models["qg"].predict(features)
            qw_pred += self.residual_models["qw"].predict(features)

        return pd.DataFrame({"qo_pred": qo_pred, "qg_pred": qg_pred,
                             "qw_pred": qw_pred, "wc_pred": wc}, index=df.index)

    def score(self, df, y_qo_col="qo_well_test", y_qg_col="qg_well_test", y_qw_col="qw_well_test"):
        df_pred = self.predict(df)
        df_all = pd.concat([df_pred, df[[y_qo_col, y_qg_col, y_qw_col]]], axis=1).dropna()
        r2_qo = r2_score(df_all[y_qo_col], df_all["qo_pred"])
        r2_qg = r2_score(df_all[y_qg_col], df_all["qg_pred"])
        r2_qw = r2_score(df_all[y_qw_col], df_all["qw_pred"])
        return {"r2_qo": r2_qo, "r2_qg": r2_qg, "r2_qw": r2_qw}


# -------------------------
# Hybrid - Physics + ML model
# -------------------------
class PhysicsInformedModel:
    """
    Fits Physics-informed model first, then trains ML model to predict residuals.
    """
    def __init__(self):
        self.phys_model = PhysicsModel()
        self.ml_qo = XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.1)
        self.ml_qg = XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.1)
        self.ml_qw = XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.1)
        self.features = ["dhp","dht","whp","wht","choke","dcp"]

    def fit(self, df):
        # Fit physics model
        self.phys_model.fit(df)
        pred_phys = self.phys_model.predict(df)

        # Compute residuals
        res_qo = df["qo_well_test"] - pred_phys["qo_pred"]
        res_qg = df["qg_well_test"] - pred_phys["qg_pred"]
        res_qw = df["qw_well_test"] - pred_phys["qw_pred"]

        # Fit ML models to residuals
        self.ml_qo.fit(df[self.features], res_qo)
        self.ml_qg.fit(df[self.features], res_qg)
        self.ml_qw.fit(df[self.features], res_qw)
        return self

    def predict(self, df):
        pred_phys = self.phys_model.predict(df)
        pred_hybrid = pred_phys.copy()
        pred_hybrid["qo_pred"] += self.ml_qo.predict(df[self.features])
        pred_hybrid["qg_pred"] += self.ml_qg.predict(df[self.features])
        pred_hybrid["qw_pred"] += self.ml_qw.predict(df[self.features])
        return pred_hybrid

    def score(self, df):
        pred = self.predict(df)
        r2_qo = r2_score(df["qo_well_test"], pred["qo_pred"])
        r2_qg = r2_score(df["qg_well_test"], pred["qg_pred"])
        r2_qw = r2_score(df["qw_well_test"], pred["qw_pred"])
        return {"r2_qo": r2_qo, "r2_qg": r2_qg, "r2_qw": r2_qw}
