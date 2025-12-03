

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor


class GradientBoost:
    def __init__(self, independent_vars: list[str], dependent_vars: list[str], df_target: pd.DataFrame):
        self._independent_vars = independent_vars
        self._dependent_vars = dependent_vars
        self._df_target = df_target
        self._models: dict[str, XGBRegressor] = {}

    def train(self):
        for target in self._dependent_vars:
            X = self._df_target[self._independent_vars]
            y = self._df_target[target]
            
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
            
            model = XGBRegressor(
                n_estimators=800,
                learning_rate=0.03,
                max_depth=6,
                subsample=0.85,
                colsample_bytree=0.8,
                objective="reg:squarederror",
                tree_method="hist",   # fast CPU training
                random_state=42
            )
            
            model.fit(X_train, y_train)
            self._models[target] = model
            print(f"{target} surrogate model trained, R2 score: {model.score(X_val, y_val):.3f}")


    def generate_dense_well_rates(self, df: pd.DataFrame) -> pd.DataFrame:
        df_dense = df.copy()

        for target in self._dependent_vars:
            missing_mask  = df_dense[df_dense[target].isna()]

            if not missing_mask .empty:
                X_pred = missing_mask [self._independent_vars]
                y_pred = self._models[target].predict(X_pred)
                df_dense.loc[missing_mask .index, target] = y_pred

            df_dense[f"{target}_imputed"] = df_dense[target].isna()

        return df_dense

        