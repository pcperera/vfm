from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor
from src.vfm.model.hybrid.base_physics_informed import BasePhysicsInformedHybridModel
from src.vfm.constants import METRICS
import numpy as np
import pandas as pd

class LatentPhysicsInformedHybridModel(BasePhysicsInformedHybridModel):
    """
    Latent-space physics-informed hybrid model.
    """

    def __init__(
        self,
        physics_model_cls,
        well_id_col:str = "well_id", 
        y_qo_col:str = "qo_well_test",
        y_qg_col:str ="qg_well_test",
        y_qw_col:str ="qw_well_test",
        mpfm_qo_col: str = "qo_mpfm",
        mpfm_qg_col: str = "qg_mpfm",
        mpfm_qw_col: str = "qw_mpfm",
        mpfm_gor_col: str = "gor_mpfm",
        mpfm_wgr_col: str = "wgr_mpfm",
        lags=1,
        random_state=42,
    ):
        self.physics_model_cls = physics_model_cls
        self.well_id_col = well_id_col
        self.lags = lags
        self.random_state = random_state
        self.y_qo_col = y_qo_col
        self.y_qg_col = y_qg_col
        self.y_qw_col = y_qw_col
        self.mpfm_qo_col = mpfm_qo_col
        self.mpfm_qg_col = mpfm_qg_col
        self.mpfm_qw_col = mpfm_qw_col
        self.mpfm_gor_col = mpfm_gor_col
        self.mpfm_wgr_col = mpfm_wgr_col
        self.phys_models = {}
        self.ml_models = {}
        self.scalers = {}

    # --------------------------------------------------
    # Lag features
    # --------------------------------------------------
    def _create_lagged_features(self, df):
        df = df.copy()
        for lag in range(1, self.lags + 1):
            for col in ["dhp", "whp", "choke"]:
                if col in df.columns:
                    df[f"{col}_lag{lag}"] = (
                        df.groupby(self.well_id_col)[col].shift(lag)
                    )
        return df

    # --------------------------------------------------
    # Fit
    # --------------------------------------------------
    def fit(self, df):
        dp = df["dhp"] - df["whp"]
        print("dp min:", dp.min())
        print("dp median:", dp.median())
        print("dp <= 0 count:", (dp <= 0).sum())

        df = self._create_lagged_features(df).dropna()

        # Fit physics
        for wid, d in df.groupby(self.well_id_col):
            phys = self.physics_model_cls()
            phys.fit(d)
            self.phys_models[wid] = phys

        latent_res = {}
        latent_X = {}

        for wid, d in df.groupby(self.well_id_col):
            phys = self.phys_models[wid]

            z_true = phys.forward_latent(d, use_measured=True)
            z_phys = phys.forward_latent(d, use_measured=False)

            X = d.drop(columns=[self.well_id_col], errors="ignore").values

            for name in phys.latent_names:
                r = z_true[name] - z_phys[name]
                latent_res.setdefault(name, []).append(r)
                latent_X.setdefault(name, []).append(X)

        for name in latent_res:
            y = np.concatenate(latent_res[name])
            X = np.vstack(latent_X[name])

            scaler = StandardScaler()
            Xs = scaler.fit_transform(X)

            ml = HistGradientBoostingRegressor(
                max_depth=4,
                learning_rate=0.05,
                max_iter=200,
                random_state=self.random_state,
            )
            ml.fit(Xs, y)

            self.scalers[name] = scaler
            self.ml_models[name] = ml

        return self

    def predict_physics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Pure physics prediction (no ML correction).
        """

        df = self._create_lagged_features(df).dropna()
        preds = []

        for wid, d in df.groupby(self.well_id_col):
            phys = self.phys_models[wid]

            # Physics latent
            z_phys = phys.forward_latent(d, use_measured=False)

            # Reconstruct directly from physics
            pred = phys.reconstruct(z_phys, d)
            pred[self.well_id_col] = wid
            preds.append(pred)

        return pd.concat(preds).sort_index()

    # --------------------------------------------------
    # Predict
    # --------------------------------------------------
    def predict_hybrid(self, df):
        df = self._create_lagged_features(df).dropna()
        out = []

        for wid, d in df.groupby(self.well_id_col):
            phys = self.phys_models[wid]
            z_phys = phys.forward_latent(d, use_measured=False)

            X = d.drop(columns=[self.well_id_col], errors="ignore").values
            z_corr = {}

            for name in phys.latent_names:
                z = z_phys[name]
                if name in self.ml_models:
                    dz = self.ml_models[name].predict(
                        self.scalers[name].transform(X)
                    )
                    z_corr[name] = z + dz
                else:
                    z_corr[name] = z

            pred = phys.reconstruct(z_corr, d)
            pred[self.well_id_col] = wid
            out.append(pred)

        return pd.concat(out).sort_index()

    def score_physics(
        self,
        df,
    ):
        results = {}

        df_pred = self.predict_physics(df)

        for wid, d in df.groupby(self.well_id_col):
            p = df_pred[df_pred[self.well_id_col] == wid]

            # -----------------------------
            # Align ground truth to predictions
            # -----------------------------
            d_aligned = d.loc[p.index]

            # -----------------------------
            # IMPORTANT: use physics model columns
            # -----------------------------
            phys = self.phys_models[wid]

            results[wid] = {
                "qo": dict(zip(
                    METRICS,
                    self.regression_metrics(
                        d_aligned[phys.y_qo_col].values,
                        p["qo_pred"].values,
                    ),
                )),
                "qw": dict(zip(
                    METRICS,
                    self.regression_metrics(
                        d_aligned[phys.y_qw_col].values,
                        p["qw_pred"].values,
                    ),
                )),
                "qg": dict(zip(
                    METRICS,
                    self.regression_metrics(
                        d_aligned[phys.y_qg_col].values,
                        p["qg_pred"].values,
                    ),
                )),
            }

        return results


    def score_hybrid(
        self,
        df: pd.DataFrame,
    ):
        """
        Score hybrid predictions per well.
        """

        df_pred = self.predict_hybrid(df)
        results = {}

        for wid, d in df.groupby(self.well_id_col):
            p = df_pred[df_pred[self.well_id_col] == wid]

            # -----------------------------
            # Align ground truth to predictions
            # -----------------------------
            d_aligned = d.loc[p.index]

            # -----------------------------
            # IMPORTANT: use physics model columns
            # -----------------------------
            phys = self.phys_models[wid]

            results[wid] = {
                "qo": dict(zip(
                    METRICS,
                    regression_metrics(
                        d_aligned[phys.y_qo_col].values,
                        p["qo_pred"].values,
                    ),
                )),
                "qw": dict(zip(
                    METRICS,
                    regression_metrics(
                        d_aligned[phys.y_qw_col].values,
                        p["qw_pred"].values,
                    ),
                )),
                "qg": dict(zip(
                    METRICS,
                    regression_metrics(
                        d_aligned[phys.y_qg_col].values,
                        p["qg_pred"].values,
                    ),
                )),
            }

        return results
