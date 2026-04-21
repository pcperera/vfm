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
  - [4. Prediction Pipeline](#4-prediction-pipeline)
    - [4.1 Physics Prediction](#41-physics-prediction)
    - [4.2 ML Residual Prediction](#42-ml-residual-prediction)
    - [4.3 Apply Bias](#43-apply-bias)
    - [4.4 Residual Clipping](#44-residual-clipping)
    - [4.5 Final Rate Reconstruction](#45-final-rate-reconstruction)
      - [Gas](#gas)
      - [Oil](#oil)
      - [Water (WGR-based)](#water-wgr-based)
    - [4.6 Liquid Consistency](#46-liquid-consistency)
    - [4.7 Optional Calibration](#47-optional-calibration)
  - [5. Evaluation](#5-evaluation)
  - [6. Design Principles](#6-design-principles)
  - [7. Advantages](#7-advantages)
  - [8. Summary](#8-summary)



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

### 4.3 Apply Bias

-   Add per-well bias in log-space

------------------------------------------------------------------------

### 4.4 Residual Clipping

-   Limit residual magnitude for stability

------------------------------------------------------------------------

### 4.5 Final Rate Reconstruction

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

### 4.6 Liquid Consistency

-   Ensure qw ≤ total liquid
-   Maintain qo + qw consistency

------------------------------------------------------------------------

### 4.7 Optional Calibration

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
