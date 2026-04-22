- [Physics-Based Calibration (MultiphasePhysicsModel)](#physics-based-calibration-multiphasephysicsmodel)
  - [Overview](#overview)
  - [Step 1: Data Preparation](#step-1-data-preparation)
  - [Step 2: Initial Parameter Estimates](#step-2-initial-parameter-estimates)
    - [Reservoir Pressure Initialization](#reservoir-pressure-initialization)
    - [Liquid Rate Initialization](#liquid-rate-initialization)
  - [Step 3: Physics Model Formulation](#step-3-physics-model-formulation)
    - [3.1 Liquid Flow (IPR-like)](#31-liquid-flow-ipr-like)
    - [3.2 Water-Cut Closure](#32-water-cut-closure)
    - [3.3 Reservoir Gas Rate](#33-reservoir-gas-rate)
    - [3.4 Gas Lift Contribution](#34-gas-lift-contribution)
  - [Step 4: Residual Construction](#step-4-residual-construction)
  - [Step 5: Optimization](#step-5-optimization)
  - [Step 6: Physical Constraints](#step-6-physical-constraints)
  - [Step 7: Geometry-Constrained Calibration](#step-7-geometry-constrained-calibration)
  - [Step 8: Partial Pooling (Regularization)](#step-8-partial-pooling-regularization)
  - [Step 9: Store Calibrated Parameters](#step-9-store-calibrated-parameters)
  - [Role in Physics-Informed Residual Learning](#role-in-physics-informed-residual-learning)
  - [Example: Numerical Example: Physics-Based Calibration for a Single Well](#example-numerical-example-physics-based-calibration-for-a-single-well)
  - [Step 0: Given Data](#step-0-given-data)
    - [1: Estimate Reservoir Pressure](#1-estimate-reservoir-pressure)
    - [2: Initialize Parameters](#2-initialize-parameters)
    - [3: Compute Pressure Ratio](#3-compute-pressure-ratio)
    - [4: Liquid Rate Prediction](#4-liquid-rate-prediction)
    - [5: Water Cut and Phase Split](#5-water-cut-and-phase-split)
    - [6: Gas Rate Prediction](#6-gas-rate-prediction)
    - [7: Residual Vector](#7-residual-vector)
    - [8: Optimization Iteration](#8-optimization-iteration)
    - [9: Final Calibrated Parameters](#9-final-calibrated-parameters)
    - [Model Usage](#model-usage)
  - [Summary](#summary)
- [Physics-Informed Residual Learning Hybrid Model](#physics-informed-residual-learning-hybrid-model)
  - [1. Overview](#1-overview)
    - [Core Idea](#core-idea)
  - [2. Architecture](#2-architecture)
  - [3. Training Pipeline](#3-training-pipeline)
    - [3.1 Physics Model Calibration](#31-physics-model-calibration)
    - [3.2 Lag Feature Engineering](#32-lag-feature-engineering)
    - [3.3 Feature Engineering](#33-feature-engineering)
      - [Physics Features](#physics-features)
      - [Operating Conditions](#operating-conditions)
      - [Lag Features](#lag-features)
    - [3.4 Residual Target Construction](#34-residual-target-construction)
    - [3.5 Regime Assignment](#35-regime-assignment)
    - [3.6 ML Model Training](#36-ml-model-training)
    - [3.7 Per-Well Bias Calibration](#37-per-well-bias-calibration)
  - [4. Prediction Pipeline](#4-prediction-pipeline)
    - [4.1 Physics Prediction](#41-physics-prediction)
    - [4.2 ML Residual Prediction](#42-ml-residual-prediction)
    - [4.3 Residual Clipping](#43-residual-clipping)
    - [4.4 Final Rate Reconstruction](#44-final-rate-reconstruction)
      - [Gas](#gas)
      - [Oil](#oil)
      - [Water (WGR-based)](#water-wgr-based)
    - [4.5 Liquid Consistency](#45-liquid-consistency)
    - [4.6 Optional Calibration](#46-optional-calibration)
  - [5. Evaluation](#5-evaluation)
  - [6. Design Principles](#6-design-principles)
  - [7. Advantages](#7-advantages)
  - [8. Summary](#8-summary)
- [FAQ](#faq)
  - [1. Why use a hybrid Physics + ML model?](#1-why-use-a-hybrid-physics--ml-model)
  - [⚙️ 2. Why residual learning instead of direct prediction?](#️-2-why-residual-learning-instead-of-direct-prediction)
  - [📊 3. Why are residuals computed in log space?](#-3-why-are-residuals-computed-in-log-space)
  - [💧 4. Why model WGR instead of water rate directly?](#-4-why-model-wgr-instead-of-water-rate-directly)
  - [💧 5. How are zero water rates handled?](#-5-how-are-zero-water-rates-handled)
  - [🌊 6. How is water breakthrough handled?](#-6-how-is-water-breakthrough-handled)
  - [7. Does the physics model allow zero water?](#7-does-the-physics-model-allow-zero-water)
  - [8. Why is water prediction gated by physics?](#8-why-is-water-prediction-gated-by-physics)
  - [🛢️ 9. How is gas lift handled?](#️-9-how-is-gas-lift-handled)
  - [10. What is gas flow coefficient (Cg)?](#10-what-is-gas-flow-coefficient-cg)
  - [11. What is the Jacobian?](#11-what-is-the-jacobian)
  - [12. Is calibration done per data point?](#12-is-calibration-done-per-data-point)
  - [13. How does optimization work?](#13-how-does-optimization-work)
  - [14. When does optimization stop?](#14-when-does-optimization-stop)
  - [15. How do you prevent overfitting?](#15-how-do-you-prevent-overfitting)
  - [16. Why regime-aware ML models?](#16-why-regime-aware-ml-models)
  - [17. Why global ML model instead of per-well?](#17-why-global-ml-model-instead-of-per-well)
  - [18. What is per-well bias?](#18-what-is-per-well-bias)
  - [19. How are unseen wells handled?](#19-how-are-unseen-wells-handled)
  - [20. How is data quality ensured?](#20-how-is-data-quality-ensured)
  - [21. What are the limitations?](#21-what-are-the-limitations)
  - [22. Possible improvements?](#22-possible-improvements)
  - [23. Key contribution](#23-key-contribution)



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

## Example: Numerical Example: Physics-Based Calibration for a Single Well

---

## Step 0: Given Data

| dhp (bar) | choke | qo (Sm³/h) | qw (Sm³/h) | qg (Sm³/h) | qL (Sm³/h) |
| --------- | ----- | ---------- | ---------- | ---------- | ---------- |
| 180       | 0.6   | 100        | 20         | 500        | 120        |
| 170       | 0.7   | 120        | 25         | 550        | 145        |
| 160       | 0.8   | 140        | 30         | 600        | 170        |
| 150       | 0.9   | 160        | 35         | 650        | 195        |
| 140       | 1.0   | 180        | 40         | 700        | 220        |

**Note:**
qL = qo + qw

---

### 1: Estimate Reservoir Pressure

Given:

* dhp_max = 180 bar
* bh_tvd = 2000 m
* ρ = 850 kg/m³

Hydrostatic pressure:

```
P = ρ g h
  = 850 × 9.81 × 2000 / 10^5
  = 166.77 bar
```

Initial estimate:

```
P_res = dhp_max + hydro
      = 180 + 166.77
      = 346.77 bar
```

Bounds:

```
Lower = 180 + 0.7 × 166.77 ≈ 296.7 bar
Upper = 180 + 1.3 × 166.77 ≈ 396.8 bar
```

---

### 2: Initialize Parameters

```
qL_max = mean(qL)
       = (120 + 145 + 170 + 195 + 220) / 5
       = 170
```

Initial guesses:

* a = 0.2
* b = 0.5
* Cg = 50

---

### 3: Compute Pressure Ratio

For first row:

```
pr = Pwf / P_res
   = 180 / 346.77
   ≈ 0.52
```

---

### 4: Liquid Rate Prediction

```
qL = qL_max × (1 − a·pr − b·pr²)
   = 170 × (1 − 0.2×0.52 − 0.5×0.52²)
   = 170 × (1 − 0.104 − 0.135)
   = 170 × 0.761
   = 129.37
```

Actual:

```
qL = 120
```

Error:

```
+9.37
```

---

### 5: Water Cut and Phase Split

Actual:

```
wc = qw / qL
   = 20 / 120
   = 0.167
```

Assume model:

```
wc ≈ 0.18
```

Predicted:

```
qw = 0.18 × 129.37 = 23.29
qo = 129.37 − 23.29 = 106.08
```

Errors:

* qw error ≈ +3.29
* qo error ≈ +6.08

---

### 6: Gas Rate Prediction

```
dp = sqrt(P_res − Pwf)
   = sqrt(346.77 − 180)
   = sqrt(166.77)
   ≈ 12.91
```

Assume:

```
choke_eff ≈ 0.7
```

```
qg = Cg × dp × choke_eff
   = 50 × 12.91 × 0.7
   ≈ 451.85
```

Actual:

```
qg = 500
```

Error:

```
−48.15
```

---

### 7: Residual Vector

For this point:

```
qo error ≈ +6.08
qw error ≈ +3.29
qg error ≈ −48.15
```

Across all data points:

* Residuals are combined
* Normalized
* Passed to optimizer

---

### 8: Optimization Iteration

Try new parameters:

* qL_max = 180
* a = 0.25
* b = 0.45

```
qL = 180 × (1 − 0.25×0.52 − 0.45×0.52²)
   = 180 × 0.748
   ≈ 134.64
```

Closer to actual (120)

Optimizer keeps updating parameters to reduce error

---

### 9: Final Calibrated Parameters

| Parameter | Value   |
| --------- | ------- |
| P_res     | 335 bar |
| qL_max    | 165     |
| a         | 0.28    |
| b         | 0.42    |
| Cg        | 55      |

---

### Model Usage

For any new input (dhp, choke, etc.):

* Predict qo
* Predict qw
* Predict qg

Using calibrated physics equations.

------------------------------------------------------------------------

## Summary

Physics calibration tunes model parameters so that physics-based predictions of oil, water, and gas rates match real well data as closely as possible.

For any new input:

* Compute qL from pressure
* Split into qo and qw using water cut
* Compute qg from pressure drawdown

Calibration process:

1. Start with physics-based estimates
2. Compare predictions with real data
3. Compute residual errors
4. Adjust parameters iteratively
5. Converge to best-fit physical model


# Physics-Informed Residual Learning Hybrid Model

## 1. Overview

This model implements a **Physics-Informed Residual Learning
Architecture** for multiphase flow prediction.

### Core Idea

Final Prediction = Physics Model + Machine Learning Residual

Where: - Physics model captures first-order flow behavior - ML model
learns systematic errors

------------------------------------------------------------------------

## 2. Architecture

The system consists of:

1.  Per-well Physics Models
2.  Regime-aware ML Residual Models
3.  Per-well Bias Calibration

------------------------------------------------------------------------

## 3. Training Pipeline

### 3.1 Physics Model Calibration

-   Fit `MultiphasePhysicsModel` per well
-   Inputs: pressure, choke, temperature
-   Outputs:
    -   qo_phys
    -   qw_phys
    -   qg_phys

------------------------------------------------------------------------

### 3.2 Lag Feature Engineering

-   Create lagged variables:
    -   dhp_lag1, dhp_lag2, ...
    -   whp_lag1, whp_lag2, ...
-   Drop unsafe rows with missing lags

------------------------------------------------------------------------

### 3.3 Feature Engineering

#### Physics Features

-   qo_phys, qw_phys, qg_phys

#### Operating Conditions

-   dhp, whp, choke, dcp
-   wht, dht

#### Lag Features

-   dhp_lag*, whp_lag*

Constraints: - No target leakage - No well ID

------------------------------------------------------------------------

### 3.4 Residual Target Construction

Residuals computed in log-space:

Δ = log(1 + y_true) − log(1 + y_phys)

Targets: - Δlog(qo) - Δlog(WGR) - Δlog(qg)

WGR = qw / qg

------------------------------------------------------------------------

### 3.5 Regime Assignment

Based on pressure drawdown:

ΔP = dhp − whp

Regimes: - below_normal - normal - above_normal

------------------------------------------------------------------------

### 3.6 ML Model Training

-   One model per regime
-   Algorithm: HistGradientBoostingRegressor
-   Multi-output: qo, WGR, qg residuals

Steps: 1. Scale features 2. Apply polynomial expansion 3. Train per
regime

The residual learning component in the proposed physics-informed hybrid framework requires a model capable of capturing complex, nonlinear deviations between physics-based predictions and observed multiphase flow rates. These residuals arise from unmodeled physical phenomena, sensor noise, and operational variability, and therefore exhibit structured but nontrivial patterns.

Selected **HistGradientBoostingRegressor (HGBR)** as the residual learner for the following reasons:

1. **Suitability for Structured Tabular Data**
   The input space consists of engineered features derived from physics predictions, operating conditions (pressure, choke, temperature), and lagged variables. Tree-based gradient boosting methods are well-established as state-of-the-art for such tabular datasets, outperforming neural networks in many practical scenarios where data is heterogeneous and moderately sized.

2. **Ability to Model Nonlinear and Interaction Effects**
   Residuals in multiphase flow systems are inherently nonlinear and involve complex interactions (e.g., pressure–choke coupling, thermal effects). HGBR captures such interactions implicitly through hierarchical tree splits, eliminating the need for extensive manual feature engineering.

3. **Consistency with Residual Learning Paradigm**
   Gradient boosting is inherently an additive error-correction method, where successive trees iteratively minimize residuals. This aligns naturally with the proposed architecture, in which the physics model provides a first-order approximation and the ML component refines the remaining error.

4. **Computational Efficiency and Scalability**
   The histogram-based implementation significantly reduces computational complexity by binning continuous features, enabling efficient training on large, high-frequency time-series datasets. This is particularly important for multi-well deployment scenarios and near-real-time applications such as virtual flow metering.

5. **Robustness to Noise and Measurement Uncertainty**
   Oilfield data sources (e.g., MPFM, SCADA sensors) are subject to noise and bias. Ensemble methods like HGBR mitigate overfitting through aggregation and regularization (e.g., learning rate, early stopping), making them well-suited for noisy industrial datasets.

6. **Stability Under Log-Transformed Targets**
   Residuals are modeled in log-space to stabilize variance and handle wide dynamic ranges in flow rates. HGBR does not assume linearity or Gaussianity of targets, and empirically performs well under such transformations.

7. **Compatibility with Regime-Based Modeling**
   The model is trained separately across operating regimes defined by pressure drawdown. HGBR performs effectively even with moderately sized subsets, allowing regime-specific specialization without excessive hyperparameter tuning.

8. **Practical Considerations and Reproducibility**
   HGBR is part of the standard scikit-learn library, ensuring ease of integration, reproducibility, and maintainability without reliance on external dependencies. This is advantageous for industrial deployment and long-term support.


In summary, HistGradientBoostingRegressor provides an effective balance between model expressiveness, computational efficiency, and robustness. Its alignment with the residual learning paradigm and its strong empirical performance on structured engineering datasets make it a suitable choice for the proposed physics-informed hybrid modeling framework.

------------------------------------------------------------------------

### 3.7 Per-Well Bias Calibration

Compute mean residual:

bias = mean(Y_true − Y_pred)

Stored per well: - \[b_qo, b_wgr, b_qg\]

------------------------------------------------------------------------

## 4. Prediction Pipeline

### 4.1 Physics Prediction

-   Compute qo_phys, qw_phys, qg_phys

------------------------------------------------------------------------

### 4.2 ML Residual Prediction

-   Build features
-   Assign regime
-   Predict residuals

------------------------------------------------------------------------

### 4.3 Residual Clipping

-   Limit residual magnitude for stability

------------------------------------------------------------------------

### 4.4 Final Rate Reconstruction

#### Gas

qg = exp(log(1 + qg_phys) + res_qg) − 1

Constraint: 0.6 × qg_phys ≤ qg ≤ 1.8 × qg_phys

------------------------------------------------------------------------

#### Oil

qo = exp(log(1 + qo_phys) + res_qo) − 1

------------------------------------------------------------------------

#### Water (WGR-based)

1.  Compute WGR_phys
2.  Apply residual
3.  Reconstruct:

qw = qg × WGR_hybrid

Gating: - Only if physics predicts water

------------------------------------------------------------------------

### 4.5 Liquid Consistency

-   Ensure qw ≤ total liquid
-   Maintain qo + qw consistency

------------------------------------------------------------------------

### 4.6 Optional Calibration

-   Blend predictions with true data (if available)

------------------------------------------------------------------------

## 5. Evaluation

Metrics: - RMSE - MAE - R²

Evaluated for: - qo, qw, qg - WGR, GOR

Supports: - Physics-only evaluation - Hybrid evaluation - MPFM
comparison

------------------------------------------------------------------------

## 6. Design Principles

-   Physics-first modeling
-   Residual learning
-   Log-space stability
-   Regime awareness
-   Per-well adaptation
-   Strong constraints

------------------------------------------------------------------------

## 7. Advantages

-   Physically interpretable
-   Robust to sparse data
-   Handles multiphase complexity
-   Suitable for real-time VFM

------------------------------------------------------------------------

## 8. Summary

The model: 1. Learns physics per well 2. Learns residuals globally 3.
Adapts to regimes 4. Applies well-specific corrections

Result: - High accuracy - Strong physical consistency


------------------------------------------------------------------------


# FAQ

---

## 1. Why use a hybrid Physics + ML model?


Pure physics models are interpretable but simplified, while ML models require large data and may violate physics.
This work combines both:

* Physics model → captures governing relationships
* ML residual model → corrects systematic errors

Result: **accuracy + physical consistency + generalization**

---

## ⚙️ 2. Why residual learning instead of direct prediction?


The ML model learns corrections:

```
Δ = log(1 + y_true) - log(1 + y_phys)
```

This simplifies learning and preserves physics structure.

---

## 📊 3. Why are residuals computed in log space?



```
Δ = log(1 + y_true) - log(1 + y_phys)
  = log((1 + y_true) / (1 + y_phys))
```

Model learns **relative (percentage) errors**, improving stability.

---

## 💧 4. Why model WGR instead of water rate directly?



```
WGR = qw / qg
qw = qg * WGR
```

Ensures stable and physically consistent water prediction.

---

## 💧 5. How are zero water rates handled?



* Step 1: Replace zeros with small epsilon

```
qw = ε  (where ε = min_nonzero_qw / 1000)
```

* Step 2: Remove zero-water rows during training

Ensures numerical stability + reliable learning.

---

## 🌊 6. How is water breakthrough handled?



* Physics model:

```
qw = wc * qL
```

* ML model:

```
Δ_WGR = log(1 + WGR_true) - log(1 + WGR_phys)
```

Breakthrough is **implicitly learned via WGR residuals**

---

## 7. Does the physics model allow zero water?



```
wc = 0.02 + 0.96 * sigmoid(X · A)
```

Minimum water cut exists → true dry conditions approximated.

---

## 8. Why is water prediction gated by physics?



```
if qw_phys > threshold:
    qw = qg * WGR
else:
    qw = 0
```

Prevents ML from predicting unphysical water.

---

## 🛢️ 9. How is gas lift handled?



```
qg = qg_res + qg_lift

qg_res  = Cg * sqrt(P_res - P_wf) * sigmoid(...)
qg_lift = C_gl * (GL_mass / conversion) * open_ratio
```

---

## 10. What is gas flow coefficient (Cg)?



```
qg ∝ Cg * sqrt(P_res - P_wf)
```

Represents gas productivity of the well.

---

## 11. What is the Jacobian?



```
J = ∂(errors) / ∂(parameters)
```

Computed automatically by optimizer.

---

## 12. Is calibration done per data point?



```
Loss = Σ (y_pred_i - y_true_i)^2
```

One parameter set fitted to **all data points**

---

## 13. How does optimization work?



```
θ_new = θ - α * ∇Loss
```

Uses nonlinear least squares to minimize total error.

---

## 14. When does optimization stop?



* Small loss change
* Small parameter updates
* Small gradient

---

## 15. How do you prevent overfitting?



* Physics constraints
* Residual learning
* Log-space modeling
* Regime-aware ML

---

## 16. Why regime-aware ML models?



Different regimes:

```
ΔP = dhp - whp
```

Separate ML models per regime improve accuracy.

---

## 17. Why global ML model instead of per-well?



* Uses all wells’ data
* Better generalization

Adaptation via bias.

---

## 18. What is per-well bias?



```
bias = mean(Δ_true - Δ_pred)
Δ_final = Δ_ML + bias
```

Corrects systematic well-specific errors.

---

## 19. How are unseen wells handled?



```
Δ_final = Δ_ML   (initially, no bias)
```

Bias added later when data is available.

---

## 20. How is data quality ensured?



* Remove non-physical data
* Enforce choke-flow consistency
* Filter extreme values

---

## 21. What are the limitations?



* No explicit breakthrough model
* Minimum water cut enforced
* Simplified gas-lift coupling

---

## 22. Possible improvements?



* Explicit breakthrough modeling
* Better multiphase coupling
* Online adaptation

---

## 23. Key contribution



Physics-informed residual learning:

```
y_final = exp(log(1 + y_phys) + Δ_ML) - 1
```

Combines physics + ML effectively.

> Physics defines structure; ML corrects errors for accurate virtual flow metering.

---
