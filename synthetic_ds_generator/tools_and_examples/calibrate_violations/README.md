# Calibration Notebooks

This directory contains notebooks and tools for calibrating violation levels in synthetic dataset generation. The calibration strategy is designed to systematically determine appropriate parameter ranges that create meaningful challenges for causal discovery algorithms.

## Calibration Strategy

The calibration process uses a **baseline method validation approach** where we:

1. **Use Cross-Correlation as Reference Method**: We employ cross-correlation as our baseline causal discovery method to calibrate.

2. **Target Performance Range**: For each violation type, we aim to find parameter levels where:
   - **First level**: Performance close to the no-violation baseline (< 0.9 AUROC)
   - **Second level**: Performance degraded to near-random levels  **IF POSSIBLE** otherwise maximum violation (~0.5 AUROC) 
   - **Intermediate levels**: Equally spaced between these extremes to create a smooth difficulty gradient

3. **Standard Test Configuration**: All calibration uses a consistent setup:
   - 5 variables
   - 3 maximum lags
   - High link probability regime
   - 250 timesteps
   - 10 samples for statistical reliability

## Notebook Descriptions

### `calibrate_nl_levels.ipynb (4 violations)`
Calibrates **nonlinearity violation levels** by testing different nonlinear function complexities:

### `calibrate_noise_levels.ipynb (6 + 2 + 1 + 7)` 
Calibrates **noise violation levels** across different noise types:
- **Innovation noise violations** 
- **Observation noise violations**:

### `calibrate_general.ipynb (13)`
Calibrates **confounding everything else**:
- **Lagged confounding** ($V_{conf,l}$): Tests hidden confounders with varying link probabilities (0→0.7)
- **Instantaneous confounding** ($V_{conf,i}$): Tests immediate confounding effects
- **Exogenous influences**: Calibrates external variable impact on the system



## Usage

The calibration notebooks were ran  in sequence to establish violation parameter ranges before generating large-scale synthetic datasets. The resulting parameter configurations ensure that synthetic data provides meaningful and graduated challenges for evaluating causal discovery algorithms.