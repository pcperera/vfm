EPS = 1e-6

METRICS = [
    "r2",
    "mae",
    "rmse",
    "mape (%)",   # aka MARE
    "mpe (%)",    # aka signed MRE / avg discrepancy
]

P_SCALE = 100.0       # ~100 bar
T_SCALE = 100.0       # ~100 °C (scaling only)


# =====================================================
# Gas lift
# =====================================================
GL_MASS_TO_STD_VOL = 1.25  # kg/Sm3 (typical lift gas density)