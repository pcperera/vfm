
# Physics-Based Calibration (MultiphasePhysicsModel)

## Overview

Physics-based calibration estimates physically meaningful parameters so
that physics equations reproduce observed oil, water, and gas rates.
This is not a black-box model; instead, it combines physics equations
with data-driven parameter estimation.

------------------------------------------------------------------------

## Step 1: Data Preparation

-   Remove rows with missing values in required columns:
    -   dhp (pressure)
    -   choke
    -   dcp, dht, wht (sensor variables)
    -   qo, qg, qw (measured rates)
-   Ensure minimum number of samples (\~10)
-   Convert all values to numeric arrays

------------------------------------------------------------------------

## Step 2: Initial Parameter Estimates

### Reservoir Pressure Initialization

If geometry (bh_tvd) is available: - Use hydrostatic relation: P = ρgh -
Estimate: P_res = dhp_max + hydro

Bounds: - Lower: dhp_max + 0.7 × hydro - Upper: dhp_max + 1.3 × hydro

If geometry not available: - Use heuristic: P_res ≈ dhp_max + offset

------------------------------------------------------------------------

### Liquid Rate Initialization

-   Compute: qL_mean = mean(qo + qw)
-   Set initial: qL_max ≈ qL_mean

------------------------------------------------------------------------

## Step 3: Physics Model Formulation

### 3.1 Liquid Flow (IPR-like)

Define pressure ratio: pr = Pwf / P_res

Liquid rate: qL = qL_max × (1 − a·pr − b·pr²)

Constraints: - qL ≥ 0 - pr ∈ \[0, 1.5\]

Parameters: - qL_max (productivity) - a, b (nonlinearity)

------------------------------------------------------------------------

### 3.2 Water-Cut Closure

Feature matrix includes: - choke, dcp, dht, wht, and interactions

Water cut: wc = 0.02 + 0.96 × sigmoid(X·A)

Ensures: - wc ∈ \[0.02, 0.98\]

Phase split: qw = wc × qL qo = (1 − wc) × qL

Parameters: - A_wc (coefficients)

------------------------------------------------------------------------

### 3.3 Reservoir Gas Rate

Pressure drawdown: dp = sqrt(max(P_res − Pwf, 0))

Choke effect: choke_eff = sigmoid(k_ch × (choke − ch0))

Gas rate: qg_res = Cg × dp × choke_eff × gas_scaling

Parameters: - Cg (gas productivity) - k_ch, ch0 (choke response)

------------------------------------------------------------------------

### 3.4 Gas Lift Contribution

If gas lift data exists: qg_lift = C_gl × (gl_mass / conversion) ×
gl_open_ratio

If not: qg_lift = 0

Total gas: qg = qg_res + qg_lift

Parameters: - C_gl (gas lift efficiency)

------------------------------------------------------------------------

## Step 4: Residual Construction

Compute errors: - Oil: (qo_pred − qo_actual) - Water: (qw_pred −
qw_actual) - Gas: (qg_pred − qg_actual)

Normalize: - Divide each by its standard deviation

Final residual vector: - Concatenate all three phases

------------------------------------------------------------------------

## Step 5: Optimization

Use nonlinear least squares: - Objective: minimize sum of squared
residuals

Algorithm: - Iterative parameter updates - Evaluate physics model at
each step - Stop when convergence criteria met

------------------------------------------------------------------------

## Step 6: Physical Constraints

Apply bounds: - qL_max ≥ 0 - 0 ≤ a ≤ 1 - 0 ≤ b ≤ 2 - 0 ≤ Cg, C_gl -
Water-cut coefficients bounded

Ensures physically valid solutions

------------------------------------------------------------------------

## Step 7: Geometry-Constrained Calibration

If bh_tvd available: - Compute hydrostatic pressure - Restrict reservoir
pressure within physical bounds

Prevents unrealistic calibration

------------------------------------------------------------------------

## Step 8: Partial Pooling (Regularization)

If global parameters exist: - Compute: alpha = n_obs / n_ref

Update parameters: param = alpha × local + (1 − alpha) × global

Interpretation: - Sparse data → rely on global physics - Dense data →
rely on well-specific calibration

------------------------------------------------------------------------

## Step 9: Store Calibrated Parameters

Final parameters include: - P_res - qL_max, a, b - Cg, C_gl - k_choke,
choke0 - A_wc

Stored for prediction

------------------------------------------------------------------------

## Role in Physics-Informed Residual Learning

Physics model predicts: qo_pred, qw_pred, qg_pred

Residual model learns: error = actual − physics

Final model: Final = Physics + Residual

------------------------------------------------------------------------

## Summary

Physics-based calibration process: 1. Clean data 2. Initialize
parameters 3. Define physics equations 4. Compute residuals 5. Optimize
parameters 6. Apply constraints 7. Use geometry bounds 8. Apply partial
pooling 9. Store parameters

This ensures: - Physical interpretability - Robust performance -
Compatibility with hybrid ML models


# Physics-Informed Hybrid Model for Virtual Flow Metering

## Overview

This model combines physics-based predictions with machine learning
residual correction.

Final Prediction: Physics Model + ML Residual

------------------------------------------------------------------------

## Step 1: Physics Model Calibration

-   Calibrate per well
-   Generate qo_phys, qw_phys, qg_phys

------------------------------------------------------------------------

## Step 2: Lag Features

-   dhp_lag, whp_lag
-   Capture temporal effects

------------------------------------------------------------------------

## Step 3: ML Features

-   Physics predictions
-   Operating conditions
-   Lagged variables

------------------------------------------------------------------------

## Step 4: Residual Targets (Log Space)

Δ = log(1 + y_true) − log(1 + y_phys)

Targets: - qo - WGR - qg

------------------------------------------------------------------------

## Step 5: Regime Assignment

ΔP = dhp − whp

Regimes: - below_normal - normal - above_normal

------------------------------------------------------------------------

## Step 6: Train ML Models

-   One model per regime
-   Gradient boosting regressors

------------------------------------------------------------------------

## Step 7: Feature Processing

-   Standard scaling
-   Polynomial features

------------------------------------------------------------------------

## Step 8: Per-Well Bias

-   Mean residual per well
-   Applied in log space

------------------------------------------------------------------------

## Prediction Pipeline

### Physics Prediction

Compute baseline rates

### ML Residual Prediction

Apply regime-based model

### Apply Bias

Add well-specific correction

### Final Outputs

-   Gas: corrected with constraints
-   Oil: direct correction
-   Water: reconstructed via WGR

------------------------------------------------------------------------

## Summary

-   Physics ensures consistency
-   ML improves accuracy
-   Hybrid approach balances both
