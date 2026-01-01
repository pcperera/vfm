EPS = 1e-9

METRICS = [
    "r2",
    "mae",
    "rmse",
    "mape (%)",   # aka MARE
    "mpe (%)",    # aka signed MRE / avg discrepancy
]

# --------------------------------------------------
# Scaling constants
# --------------------------------------------------
P_SCALE = 100.0       # ~100 bar (numerical scaling only)
T_SCALE = 100.0       # ~100 °C (numerical scaling only)


# =====================================================
# Gas lift
# =====================================================
GL_MASS_TO_STD_VOL = 1.25  # kg/Sm3 (typical lift gas density)
