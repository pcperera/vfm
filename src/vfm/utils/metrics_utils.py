from src.vfm.model.hybrid.constants import *
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np

METRICS = [
    "r2",
    "mae",
    "rmse",
    "mape (%)",   # aka MARE
    "mpe (%)",    # aka signed MRE / avg discrepancy
]

def get_global_metrics(y_true, y_pred):
    """
    Compute global regression metrics by pooling all samples.
    """
    if len(y_true) < 2:
        return {m: np.nan for m in METRICS}

    return dict(zip(
        METRICS,
        regression_metrics(y_true, y_pred)
    ))


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
    