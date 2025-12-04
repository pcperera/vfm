import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from sklearn.metrics import r2_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import PolynomialFeatures

# -------------------------
# Helpers
# -------------------------
def logistic(x):
    x = np.clip(x, -50, 50)
    return 1 / (1 + np.exp(-x))

# -------------------------
# Physics-based model with Closure Law
# -------------------------
class PhysicsModel:
    def __init__(self, estimate_pres_offset=1.0, fit_pres=True):
        self.fit_pres = fit_pres
        self.estimate_pres_offset = estimate_pres_offset
        self.params_ = None

    def _feature_matrix_for_wc(self, df):
        choke, dcp, dht, wht = df["choke"].values, df["dcp"].values, df["dht"].values, df["wht"].values
        X = np.vstack([
            np.ones(len(df)),
            choke, dcp, dht, wht,
            choke**2, dcp**2, dht**2, wht**2,
            choke*dcp, choke*dht, choke*wht,
            dcp*dht, dcp*wht, dht*wht
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
        P_res, qL_max, Cg, A = self._pack_params(x, n_wc_features=15)
        if P_res is None:
            P_res = df["dhp"].max() + self.estimate_pres_offset

        Pwf = df["dhp"].values
        qL = qL_max * (1 - 0.2*Pwf/P_res - 0.8*(Pwf/P_res)**2)
        qL = np.maximum(0.0, qL)

        Xwc = self._feature_matrix_for_wc(df)
        wc = logistic(Xwc.dot(A))
        qw_pred = wc * qL
        qo_pred = (1 - wc) * qL
        qg_pred = Cg * np.sqrt(np.maximum(0.0, P_res**2 - Pwf**2))

        eps = 1e-8
        r_qo = (qo_pred - y_qo) / max(np.std(y_qo), eps)
        r_qw = (qw_pred - y_qw) / max(np.std(y_qw), eps)
        r_qg = (qg_pred - y_qg) / max(np.std(y_qg), eps)
        return np.concatenate([r_qo, r_qw, r_qg])

    def fit(self, df, y_qo_col="qo_well_test", y_qg_col="qg_well_test", y_qw_col="qw_well_test"):
        df_fit = df[["dhp","dht","whp","wht","choke","dcp",y_qo_col,y_qg_col,y_qw_col]].dropna()
        y_qo, y_qg, y_qw = df_fit[y_qo_col].values, df_fit[y_qg_col].values, df_fit[y_qw_col].values

        if self.fit_pres:
            x0 = [df_fit["dhp"].max()+1.0, 500.0, 0.1] + [0.0]*15
            lb = [df_fit["dhp"].max(), 0.0, 0.0] + [-10]*15
            ub = [df_fit["dhp"].max()+1000, 10000.0, 100.0] + [10]*15
        else:
            x0 = [500.0, 0.1] + [0.0]*15
            lb = [0.0, 0.0] + [-10]*15
            ub = [10000.0, 100.0] + [10]*15

        res = least_squares(self.residuals, x0, bounds=(lb, ub),
                            args=(df_fit, y_qo, y_qg, y_qw), max_nfev=10000)
        P_res, qL_max, Cg, A = self._pack_params(res.x, 15)
        if P_res is None: P_res = df_fit["dhp"].max() + 1.0

        self.params_ = {"P_res": P_res, "qL_max": qL_max, "Cg": Cg, "A": np.array(A),
                        "success": res.success, "message": res.message}
        return self

    def predict(self, df):
        if self.params_ is None:
            raise RuntimeError("Physics model not fitted")
        P_res, qL_max, Cg, A = self.params_["P_res"], self.params_["qL_max"], self.params_["Cg"], self.params_["A"]

        Pwf = df["dhp"].values
        qL = qL_max * (1 - 0.2*Pwf/P_res - 0.8*(Pwf/P_res)**2)
        qL = np.maximum(0.0, qL)

        Xwc = self._feature_matrix_for_wc(df)
        wc = logistic(Xwc.dot(A))
        qw_pred = wc * qL
        qo_pred = (1 - wc) * qL
        qg_pred = Cg * np.sqrt(np.maximum(0.0, P_res**2 - Pwf**2))

        return pd.DataFrame({"qo_pred": qo_pred, "qg_pred": qg_pred, "qw_pred": qw_pred, "wc_pred": wc}, index=df.index)

    def score(self, df, y_qo_col="qo_well_test", y_qg_col="qg_well_test", y_qw_col="qw_well_test"):
        pred = self.predict(df)
        r2_qo = r2_score(df[y_qo_col], pred["qo_pred"])
        r2_qw = r2_score(df[y_qw_col], pred["qw_pred"])
        r2_qg = r2_score(df[y_qg_col], pred["qg_pred"])
        return {"r2_qo": r2_qo, "r2_qw": r2_qw, "r2_qg": r2_qg}

# -------------------------
# Hybrid Physics + ML Model with Water Cut Prediction
# -------------------------
class PhysicsInformedHybridModel:
    def __init__(self, degree=2, lags=1):
        self.phys_model = PhysicsModel()
        self.features = ["dhp","dht","whp","wht","choke","dcp"]
        self.degree = degree
        self.lags = lags
        
        self.poly = PolynomialFeatures(degree=self.degree, include_bias=False)
        self.ml_qo = GradientBoostingRegressor(n_estimators=500, max_depth=6, learning_rate=0.05)
        self.ml_wc = GradientBoostingRegressor(n_estimators=500, max_depth=6, learning_rate=0.05)
        self.ml_qg = GradientBoostingRegressor(n_estimators=500, max_depth=6, learning_rate=0.05)

    def _create_lagged_features(self, df):
        df_lagged = df.copy()
        for lag in range(1, self.lags+1):
            for col in ["dhp","whp","qo_well_test","qw_well_test"]:
                df_lagged[f"{col}_lag{lag}"] = df_lagged[col].shift(lag)
        df_lagged.dropna(inplace=True)
        return df_lagged

    def _transform_features(self, df):
        X = df[self.features].values
        return self.poly.fit_transform(X)

    def fit(self, df, y_qo_col="qo_well_test", y_qg_col="qg_well_test", y_qw_col="qw_well_test"):
        # Physics model
        self.phys_model.fit(df, y_qo_col, y_qg_col, y_qw_col)
        df_lagged = self._create_lagged_features(df)
        
        pred_phys = self.phys_model.predict(df_lagged)
        # Water cut
        wc_actual = df_lagged[y_qw_col] / (df_lagged[y_qw_col] + df_lagged[y_qo_col] + 1e-8)
        wc_pred = pred_phys["wc_pred"]
        
        # Residuals
        res_qo = df_lagged[y_qo_col] - pred_phys["qo_pred"]
        res_wc = wc_actual - wc_pred
        res_qg = df_lagged[y_qg_col] - pred_phys["qg_pred"]

        X_poly = self._transform_features(df_lagged)
        self.ml_qo.fit(X_poly, res_qo)
        self.ml_wc.fit(X_poly, res_wc)
        self.ml_qg.fit(X_poly, res_qg)
        return self

    def predict(self, df):
        df_lagged = self._create_lagged_features(df)
        pred_phys = self.phys_model.predict(df_lagged)
        X_poly = self._transform_features(df_lagged)

        # Correct physics predictions
        pred_hybrid = pred_phys.copy()
        pred_hybrid["qo_pred"] += self.ml_qo.predict(X_poly)
        wc_corrected = pred_phys["wc_pred"] + self.ml_wc.predict(X_poly)
        wc_corrected = np.clip(wc_corrected, 0.0, 1.0)
        pred_hybrid["qw_pred"] = wc_corrected * (pred_hybrid["qo_pred"] + pred_hybrid["qw_pred"])
        pred_hybrid["qo_pred"] = (1 - wc_corrected) * (pred_hybrid["qo_pred"] + pred_hybrid["qw_pred"])
        pred_hybrid["qg_pred"] += self.ml_qg.predict(X_poly)
        return pred_hybrid

    def score(self, df, y_qo_col="qo_well_test", y_qg_col="qg_well_test", y_qw_col="qw_well_test"):
        pred = self.predict(df)
        r2_qo = r2_score(df[y_qo_col].iloc[self.lags:], pred["qo_pred"])
        r2_qw = r2_score(df[y_qw_col].iloc[self.lags:], pred["qw_pred"])
        r2_qg = r2_score(df[y_qg_col].iloc[self.lags:], pred["qg_pred"])
        return {"r2_qo": r2_qo, "r2_qw": r2_qw, "r2_qg": r2_qg}

    def physics_score(self, df, y_qo_col="qo_well_test", y_qg_col="qg_well_test", y_qw_col="qw_well_test"):
        pred_phys = self.phys_model.predict(df)
        r2_qo = r2_score(df[y_qo_col], pred_phys["qo_pred"])
        r2_qw = r2_score(df[y_qw_col], pred_phys["qw_pred"])
        r2_qg = r2_score(df[y_qg_col], pred_phys["qg_pred"])
        return {"r2_qo": r2_qo, "r2_qw": r2_qw, "r2_qg": r2_qg}
