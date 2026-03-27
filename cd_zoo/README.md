# Causal Discovery Zoo
This is a collection of Causal Discovery methods for time-series it is part of the TCD-Arena package.
While this repo can be used in a standalone manner, if you want to work in the TCD-arena framework, please refer to the main repository.

---

## 📑 Table of Contents
- [Overview](#-overview)
- [Methods Overview](#-causal-discovery-methods-overview)
- - [Repository Structure](#-repository-structure)
- [Installation & Setup](#-installation--setup)
- [Supported Data Formats](#-supported-data-formats)
- [Quick Start](#-quick-start)
- [General Workflow](#-general-workflow)
- [Contributing](#-contributing)
- [Citation](#-citation)
- [License](#-license)
- [Detailed Method Descriptions](#-detailed-method-descriptions)

---

## 📋 Overview

This repository provides a unified framework for evaluating causal discovery algorithms on time series data. It includes:

1. **Unified Benchmarking Framework**: Standardized evaluation pipeline for causal discovery methods
2. **15+ Causal Discovery Methods**: Implementations and wrappers arround state-of-the-art algorithms to provide a consistent interface
3. **Flexible Data Formats**: Support for multiple input/output formats (split, joint, TCD-Arena)
4. **Hydra Configuration Management**: Easy experiment configuration and hyperparameter tuning
5. **Comprehensive Evaluation Tools**: Multiple scoring metrics (F1, precision, recall, SHD, ROC-AUC)
6. **Batch Processing**: Scripts for large-scale parallel experiments

**Key Features:**
- ✅ Supports both classical (VAR, VARLiNGAM) and modern deep learning methods (TCDF, Causal Pretraining)
- ✅ Splits lagged and instantaneous causal relationships during evaluation
- ✅ Per-sample and aggregate evaluation metrics
- ✅ Easy integration of new methods via standardized interface
- ✅ Compatible with TCD-Arena benchmarking framework


### If you run this repo as a drop in replacement in TCD arena run: 

```bash
./cd_zoo/scripts/run_all_methods_on_data_release.sh --method varlingam # Beware this runs ALL datasets. 
# You can restrict the data by either
./cd_zoo/scripts/run_all_methods_on_data_release.sh --method varlingam string_that_should_be_contained_in_violation_name (e.g. obs)
#Or
./cd_zoo/scripts/run_all_methods_on_data_release.sh --method varlingam string_that_should_be_contained_in_violation_name (e.g. obs) maximum_number_of_datasets
```

Additional functionality can be found below.


## 🎯 Current Causal Discovery Methods Overview

| Method | Script Method Name | Type | Deep Learning | Instantaneous | Window Graph | Summary Graph | Publication |
|:-------|:-------------------|:-----|:--------------|:--------------|:-------------|:-------------|:------------|
| **Direct Cross-Correlation** | direct_crosscorr | Heuristic | ❌ | ❌ | ✅ | ✅ | Simple Correlation |
| **VAR** | var | Granger | ❌ | ❌ | ✅ | ✅ | [Stock & Watson (2001)](https://www.princeton.edu/~mwatson/papers/Stock_Watson_JEP_2001.pdf) |
| **VAR-LINGAM** | varlingam | Granger + Noise | ❌ | ✅ | ✅ | ✅ | [Hyvärinen et al. (2010)](https://onlinelibrary.wiley.com/doi/full/10.1111/j.1468-0084.2012.00710) |
| **DYNOTEARS** | dynotears | Score | ❌ | ✅ | ✅ | ✅ | [Pamfil et al. (2020)](https://arxiv.org/abs/2002.00498) |
| **PCMCI** | pcmci | Constraint | ❌ | ❌ | ✅ | ✅ | [Runge et al. (2019)](https://www.science.org/doi/10.1126/sciadv.aau4996) |
| **PCMCI+** | pcmciplus | Constraint | ❌ | ✅ | ✅ | ✅ | [Runge (2020)](https://arxiv.org/abs/2003.03685) |
| **FPCMCI**  | fpcmci | Constraint | ❌ | ✅  | ✅ | ✅ | [Castri et al.(2023)](https://proceedings.mlr.press/v213/castri23a.html) |
| **Causal Pretraining** | cp | Ammortized | ✅ | ❌ | ✅ | ✅ | [Stein et al. (2024)](https://arxiv.org/abs/2402.09305) |
| **NTS-NOTEARS** | ntsnotears | Score | ✅ | ✅ | ✅ | ✅ | [Sun et al. (2021)](https://arxiv.org/abs/2109.04286) |
| **SVAR-RFCI** | svarrfci | Constraint | ❌ | ✅ | ✅ | ✅ | [Colombo et al. (2012)](https://www.jmlr.org/papers/v13/colombo12a.html) |
| **Cross-Correlation Peak** | crosscorr_peak | Heuristic | ❌ | ❌ | ❌ | ✅ | [Stein et al. (2025)](https://arxiv.org/abs/2503.17452) |
| **Physical Principle** | physical | Heuristic | ❌ | ❌ | ❌ | ✅ |  [Stein et al. (2025)](https://arxiv.org/abs/2503.17452) |
| **CrossCorr + Physical** | crosscorr_physical | Heuristic | ❌ | ❌ | ❌ | ✅ |  [Stein et al. (2025)](https://arxiv.org/abs/2503.17452) |
| **CDMI** | cdmi | Granger | ✅ | ❌ | ❌ | ✅ | [Ahmad et al. (2022)](https://arxiv.org/abs/2207.04055) |
| **SVAR-FCI** | svarfci | Constraint | ❌ | ✅ | ✅ | ✅ | [Malinsky & Spirtes (2018)](http://proceedings.mlr.press/v92/malinsky18a) |




## 📁 Repository Structure

```
full_cd_zoo/
├── benchmark.py                      # Main benchmarking script for TCD-Arena style evaluation
├── example_run.py                    # Example usage script for simple data formats
├── run_methods_on_causal_rivers.py   # Script for CausalRivers dataset (⚠️ Data not available.)
├── simple_usage.ipynb                # Trivial tutorial notebook
├── methods/                          # Causal discovery method implementations
│   ├── var.py         # Classical time series methods
│   ├──...
├── config/                           # Hydra configuration files
│   ├── benchmark.yaml                # Main benchmark configuration
│   ├── example_run.yaml              # Simple run configuration
│   ├── method/                       # Method-specific configurations
│   ├── ci_test/                      # Conditional independence test configurations
│   └── data_preprocess/              # Data preprocessing configurations
├── tools/                            # Utility functions and scoring tools
│   ├── tools.py                      # Core utilities
│   ├── scoring_tools.py              # Evaluation metrics
│   ├── method_loader.py              # Dynamic method loading
│   └── helpers.py                    # Helper functions
├── sample_datasets/                  # Example datasets in different formats
│   ├── split_format/                 # Per-sample file format
│   └── joint_format/                 # Batch processing format
│   └── no_violation_small/                 # TCD-Arena format
├── envs/                             # Conda environment configuration files
│   ├── cd_zoo.yml                    # Main environment
│   ├── pcmci.yml, cdmi.yml, deep.yml # Method-specific environments
│   └── base.yml, dyno.yml            # Additional environments
├── scripts/                          # Scripts used throughout the TCD-Arena experiments
│   ├── execute_all_methods.sh        # Run all benchmark.py with all available methods on specified TCD-Arena Folder
│   ├── execute_all_for_causal_rivers.sh # Run methods on CausalRivers dataset
│   ├── execute_method_on_all_folders.sh # Apply single method across folders
│   └── execute_nl_ci_tests.sh        # Test nonlinear CI tests
```

## 🔧 Installation & Setup

### Basic Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/TCD-Arena/cd_zoo
   cd cd_zoo
   ```

2. **Install the main environment**
   ```bash
   
   conda env create -f envs/cd_zoo_basic.yml
   conda activate cd_zoo_basic
   ```

3. **Install the torch environment (optional)**
   ```bash
   conda env create -f envs/cd_zoo_torch.yml
   conda activate cd_zoo_torch
   ```
### Method-Specific Environments
For FPCMCI, please follow the instructions and install on top of the base environment: https://github.com/lcastri/causalflow
For CDMI, we have a custom environment in envs/cd_zoo_cdmi.yaml
   

## 📊 Supported Data Formats

The framework supports three input formats for maximum flexibility:

### Format 1: Split Files (Per-Sample Processing)
**Structure**: Individual files for each sample
- **Time Series File**: CSV/numpy file containing time series data (time × variables)
- **Labels File**: NetworkX pickle or adjacency matrix with ground truth causal graph
- **Use Case**: Individual sample analysis, debugging, small-scale experiments
- **Example**: `sample_datasets/split_format/`

**Directory structure:**
```
split_format/
├── 0_data.csv
├── 0_label.csv
├── 1_data.csv
└── 1_label.csv
```

### Format 2: Joint Files (Batch Processing)
**Structure**: Combined files for multiple samples
- **Time Series CSV**: Contains data for all samples with sample identifier column
- **Labels Pickle**: List of NetworkX graph objects (one per sample)
- **Use Case**: Multi-sample processing, efficient batch evaluation
- **Example**: `sample_datasets/joint_format/`

**Files:**
```
joint_format/
├── data.csv      # All samples combined
└── labels.p   # List of NetworkX graphs
```

### Format 3: TCD-Arena Structure (Large-Scale Benchmarking)
**Structure**: Hierarchical folder organization for systematic evaluation
- **Directory hierarchy**: `main_folder/violation_type/data_regime_and_level/files`
- **Data structure**: Numpy arrays for efficient batch processing + config
- **Use Case**: Large-scale benchmarking across multiple data regimes and violation types
- **Example**: Used in `benchmark.py` script

**Directory structure:**
```
tcd_arena_format/
├── no_violation_small/           # violation+ size: e.g. Small graph structure, no violations
│   ├── 0/                        # Dataset instance 0 (a certain violation level + specific data regime)
│   │   ├── time_series.npy       # Shape: (n_samples, n_variables, n_timesteps)
│   │   ├── lagged_graphs.npy     # Shape: (n_samples, n_variables, n_variables, max_lag)
│   │   └── instant_graphs.npy    # Shape: (n_samples, n_variables, n_variables)
│   │   └── instant_graphs.npy    # Shape: (n_samples, n_variables, n_variables)
│   │   └── resample_statistics.npy    # See Synth-TS generator
│   │   └── config.yaml    # Parameters used to generate the sample (See Synth-TS-generator)


│   ├── 1/                        # Dataset instance 1
│   │   ├── time_series.npy
│   │   ├── lagged_graphs.npy
│   │   └── instant_graphs.npy
│   └── ...                       # Additional dataset instances
│
├── other_violation_size/           # Baseline: Large data regime, no violations
│   ├── 0/
│   ├── 1/
│   └── ...
```


## 🚀 Quick Start

### Three Ways to Get Started

#### 1. Trivial Tutorial
```bash
jupyter notebook simple_usage.ipynb
```
You can explore basic usage with guided examples in the notebook.

#### 2. Simple Example Run
```bash
# Run with custom method and dataset
python example_run.py \
    loading_mode="joint" \
    data_path="sample_datasets/joint_format/" \
    method=var \
    data_preprocess.normalize=False
```

Supported loading modes:
- `loading_mode="single"`: Load individual sample files
- `loading_mode="joint"`: Load batch files (CSV + pickle)

#### 3. TCD-Arena Style Benchmarking
```bash
# Run full benchmark on a dataset folder
python benchmark.py \
    method=var \
    data_path="/path/to/dataset/folder" \
    which_dataset=0
```

#### 4. Batch Processing with Scripts
```bash

# If you run this as a drop in replacement in TCD arena run: 
./cd_zoo/scripts/run_all_methods_on_data_release.sh --method varlingam # Beware this runs ALL datasets. 
# You can restrict the data by either
./cd_zoo/scripts/run_all_methods_on_data_release.sh --method varlingam string_that_should_be_contained_in_violation_name (e.g. obs)
#Or
./cd_zoo/scripts/run_all_methods_on_data_release.sh --method varlingam string_that_should_be_contained_in_violation_name (e.g. obs) maximum_number_of_datasets


# Run all methods on a dataset
./scripts/execute_all_methods.sh /path/to/dataset

# Run specific method on all folders
./scripts/execute_method_on_all_folders.sh var /path/to/datasets
```


## 📂 General Workflow

The benchmarking framework follows a systematic process:

### 1. **Data Loading & Preprocessing**
- Loads time series data in various formats (split, joint, or TCD-Arena structure)
- Applies optional preprocessing: normalization, standardization, detrending
- Validates data shape and handles missing values

### 2. **Method Selection & Configuration** 
- Dynamically imports the specified causal discovery method via Hydra config
- Loads method-specific hyperparameters from YAML configuration files
- Supports 15 methods ranging from classical (VAR, VARLiNGAM) to deep learning (NTS-NOTEARS, Causal Pretraining)

### 3. **Causal Discovery Execution**
- Each method implements a unified interface: `run_method(data, config)`
- Input: Multivariate time series (variables × time)
- Output: 
  - Lagged causal graph (effect × cause × lag)
  - Instantaneous causal graph (effect × cause) *if method supports it*
- Tracks execution time for performance analysis
- Supports parallel processing across datasets

### 4. **Comprehensive Evaluation**
- Computes multiple metrics: F1-score, precision, recall, accuracy, SHD, ROC-AUC
- Evaluates at three granularities:
  - **Summary graph**: Aggregated causal relationships
  - **Window graph**: Lag-specific relationships
  - **Instantaneous graph**: Same-time-step relationships
- Optional: Removes autoregressive relationships for cleaner evaluation
- Supports both per-sample and aggregate metrics

### 5. **Results Export & Analysis**
- Saves predictions, ground truth, and metrics in a structured format
- Outputs runtime statistics for performance profiling
- Compatible with downstream analysis and visualization tools
- Results organized by timestamp and experiment configuration


### 6. CausalRivers Integration

The suite includes complete integration with the CausalRivers benchmark:

```bash

# Clone causalrivers
cd ensembling_experiments/causalrivers

# Install CausalRivers environment
./install.sh
conda activate causalrivers

# Generate benchmark datasets
cd causalrivers
python 0_generate_datasets.py

# Transfer samples to TCD-Format by running: 
cd ..
python prepare_causal_rivers.py

# Execute methods: 
python run_methods_on_causal_rivery.py


# Run benchmarking
python benchmark.py method=var \
    data_path=product/rivers_ts_east_germany.csv \
    method.max_lag=5
```


---

## 🤝 Contributing

We welcome contributions to the Causal Discovery Zoo! Here's how you can help:

### 🔧 Development Status
- Full Release 1.0
- We are committed to adding more methods in the same manner in the future. 

### 🔧 Adding a New Method

1. Create method implementation in `methods/your_method.py`
2. Implement the standard interface: `def run_your_method(data_sample, cfg)`
3. Add a configuration file in `config/method/your_method.yaml`
4. Update method loader in `tools/method_loader.py.`
5. Add to the test script and validate good performance on the linear systems
6. Extend the documentation


## 📄 Citation

When you use specific methods, please also cite the original papers listed in the methods table above.
If you use the Causal Discovery Zoo in your research, please cite the TCD-Arena paper:


```bibtex
@inproceedings{
stein2026tcdarena,
title={{TCD}-Arena: Assessing Robustness of Time Series Causal Discovery Methods Against Assumption Violations},
author={Gideon Stein and Niklas Penzel and Tristan Piater and Joachim Denzler},
booktitle={The Fourteenth International Conference on Learning Representations},
year={2026},
url={https://openreview.net/forum?id=MtdrOCLAGY}
}
```

---

## 📜 License

This project is part of the TCD-Arena release. Please refer to the main repository for licensing information.

---


# 📖 Detailed Method Descriptions

*The following sections provide detailed mathematical descriptions and implementation details for each causal discovery method.*


## Cross-correlation-Peak
Naive strategy that directs edges by looking for the lag that maximizes the cross-correlation between two time series:

$$  \underset{k \in {-k, \dots, k}}{\text{argmax}}  \hat{\rho}{X,Y}(k) = \frac{\frac{1}{n}\sum{t=1}^{n-|k|} (X_t - \bar{X})(Y_{t+k} - \bar{Y})}{s_X s_Y} ] $$

if $k$ is positive $X \rightarrow Y$ is infered. If k is negative $X \leftarrow Y$ is inferred.

Naive strategy. Has no guarantees to work.


**HPS**: 

    - max_lag: maximum lag that is evaluated in both directions.
    - filter_mode: As the method only directs each edge,an additional filter can be applied that, for each parent, selects the strongest correlation as the only child 


-----

## Physical principle
A simple var model is fitted to the time series process:

$$\mathbf{Y}_t = \mathbf{c} + \sum_{i=1}^p \mathbf{A}_i \mathbf{Y}_{t-i} + \boldsymbol{\varepsilon}_t$$

Naive strategy that simply directs all edges via the mean of the time series: 

$ x_1 \rightarrow x_2 \text{ if } \bar{x}_1 > \bar{x}_2 \text{ else } x_1 \leftarrow x_2 $


**HPS**: 

    - reverse_physical: Principle can be reversed.
    - filter_mode: : As the method only directs each edge,an additional filter can be applied that, for each parent, selects either the largest mean or the next closest mean.

-----

## Cross Correlation Peak + Reverse Physical
Combines both naive principles and only considers the union over the links

$$ \text{CrossCorr } \cup   \text{ Phsyical}$$

**HPS**: 

    - max_lag: maximum lag that is evaluated in both directions.
    - reverse_physical: Principle can be reversed.
    - filter_mode: Applies one of the three filters after the union operator.

-----

## Direct Cross Correlation

Attempts to recover the window causal graph by simply thresholding the lagged-cross-correlation Tensor.


$$ \text{Causal Strength}_{X \rightarrow Y, k}  = \frac{\frac{1}{n}\sum{t=1}^{n-|k|} (X_t - \bar{X})(Y_{t-k} - \bar{Y})}{s_X s_Y} $$



**HPS**: 

    - max_lag: maximum lag that is evaluated in both directions.

-----

## VAR
A simple var model is fitted to the time series process:

$$\mathbf{Y}_t = \mathbf{c} + \sum_{i=1}^p \mathbf{A}_i \mathbf{Y}_{t-i} + \boldsymbol{\varepsilon}_t$$

Now we either use the coefficients directly ($\mathbf{A}$) or rely on p-values of the fitted model.


**HPS**: 

    - max_lag: the order of the model
    - base_on: Whether to base decisions on 'coefficients' or 'pvalues'
    - absolute_coefficients: Whether or not to consider the absolute values of the coefficients.
-----


## Varlingam

VAR model for lagged effects + Lingam for instantanous effects.  

$$x(t) = \sum_{\tau=0}^{k} B_\tau x(t - \tau) + e(t)$$


**Key assumptions**:

- linearity
- non gaussian error
- acyclicity


**HPS**: 

    - max_lag: the order of the model (can be disabled by setting criterion)
    - criterion: Model selection criterion ('aic', 'fpe', 'hqic', 'bic', or None). If set, automatically searches for best lag.
    - prune: Whether to prune weak edges from the causal graph

-----

## Dynotears

Reformulates the structure as an optimization problem that can be solved via standard solvers by adding the aciclicity constraint via augmented lagrangian.


$$\min_{W, A} \, f(W, A) \quad \text{s.t.} \quad W \text{ is acyclic}, \tag{5}$$
$$f(W, A) = \ell(W, A) + \lambda_W \| W \|_1 + \lambda_A \| A \|_1.$$
$$\ell(W, A) = \frac{1}{2n} \| X - XW - YA \|_F^2.$$

$$F(W, A) = f(W, A) + \frac{\lambda}{2} h(W)^2 + \alpha h(W) \tag{7}$$

$$\begin{equation}
h(W) = \text{tr}\left( e^{W W^\top} \right) - d \quad \text{s.t.} \quad h(W) = 0 \iff W \text{ is acyclic}. \tag{8}
\end{equation}$$




**HPS**: 

    - max_lag: the order of the model
    - lambda_w: l1 loss for instantaneous coefficients
    - lambda_a: l1 loss for lagged coefficients
    - max_iter: number of dual ascent steps
    - h_tol: breaks earlier if acyclicity constraint is satisfied (tolerance for h(W))




-----

## PCMCI
PCMCI combines the PC algorithm with momentary conditional independence (MCI) testing.

**Algorithm**:
1. **Condition-selection step**: For each variable, identify potential parents $\widehat{\mathcal{P}}$ using iterative conditional independence tests (variant of PC1).
2. **MCI step**: Test conditional independence between variables given their selected conditions to remove false positives caused by autocorrelation.

For variables $X_t$ and $Y_{t-\tau}$, test:
$$X_t \perp\!\!\!\perp Y_{t-\tau} \mid \widehat{\mathcal{P}}(X_t) \setminus \{Y_{t-\tau}\}, \widehat{\mathcal{P}}(Y_{t-\tau})$$

**Key innovation**: Restores well-calibrated false positive rates and improves statistical power in highly autocorrelated time series.


**HPS**: 

    - max_lag: the order of the model
    - ci_test: Conditional independence test that is used.

----
## PCMCI+
Extends PCMCI to discover both lagged and contemporaneous (instantaneous) causal relationships.

**Algorithm**:
1. **Lagged discovery**: Apply PCMCI to identify lagged parents $\mathcal{P}^-(X_t)$
2. **Contemporaneous discovery**: For same-time variables, test:

$$X_t \perp\!\!\!\perp Y_t \mid \mathcal{P}^-(X_t), \mathcal{P}^-(Y_t), \mathcal{S}$$

where $\mathcal{S}$ is a separating set of contemporaneous variables.

3. **Orientation**: Orient contemporaneous links using conditional independence tests and orientation rules to ensure acyclicity within each time slice.

**Key innovation**: Handles both lagged and instantaneous causal effects while maintaining computational efficiency.

**HPS**: 

    - max_lag: the order of the model
    - ci_test: Conditional independence test that is used
    - reset_lagged_links: Whether to reset lagged links during orientation
    - contemp_collider_rule: Rule for resolving contemporaneous colliders ('majority', 'conservative', 'contemporary')
-----

## FPCMCI
Filtered PCMCI. Uses Transfer Entropy to filter out variables before applying PCMCI


**HPS**: 

    - max_lag: the order of the model
    - ci_test: Conditional independence test that is used (e.g., 'parcorr', 'robust_parcorr')

-----


## Causal Discovery using Model Invariance (CDMI)

CDMI uses Deep-Ar for Granger causality testing with knockoff-based variable selection.

**Algorithm**:
1. **Fit predictive model**: Train neural network $\hat{f}$ to predict $X_t$ from lagged variables:

$$\hat{X}_t = \hat{f}(X_{t-1}, X_{t-2}, \ldots, X_{t-p})$$

2. **Generate knockoffs**: Create knockoff variables $\tilde{X}$ that preserve the conditional distribution structure

3. **Test for causality**: For each variable $Y$, compute test statistic based on prediction improvement:

$$T_Y = \text{Importance}(Y) - \text{Importance}(\tilde{Y})$$

4. **Threshold**: Variables with $T_Y > \lambda$ are identified as causes

**Key innovation**: Combines deep learning's expressiveness with rigorous statistical guarantees via knockoff framework.


**HPS**: 

    - batch_size: Batch size for training
    - freq: Frequency parameter (required by gluonts)
    - device: Device to run on ('cpu' or 'cuda')
    - learning_rate: Learning rate for training
    - epochs: Number of training epochs
    - num_layers: Number of LSTM layers
    - num_cells: Number of LSTM cells per layer
    - dropout_rate: Dropout rate
    - context_length: Length of context window
    - num_samples: Number of samples for prediction
    - intervention_type: Type of intervention ('knockoff')
    - error_metric: Error metric for evaluation ('mae')
    - significance_test: Statistical test ('kolmo')
    - mean_std: Mean standard deviation threshold
    - step_size: Step size for windowing
    - prediction_length: Length of prediction horizon
    - normalize_effect_strength: Whether to normalize effect strengths
    - select_automatic_data_split: Whether to automatically split data
    - training_length: Length of training data
    - num_windows: Number of windows for training
    - use_cached_model: Whether to use cached model
    - save_intermediate: Whether to save intermediate results

-----
## Causal Pretraining
A Direct mapping from Time series to Window causal graph that is learned in a supervised manner from synthetic data samples. It is parameterized by an arbitrary neural network.

A direct mapping from Time Series (TS) to a Causal Graph learned via supervised learning on massive synthetic datasets. Parameterized by a neural network (e.g., Transformer).
$$ f(\mathbf{X}_{t-p:t}) \rightarrow \mathbf{A} \in [0,1]^{d \times d} $$


**HPS**: 

    - batch_size: Batch size (only relevant for very long time series)
    - architecture: 'transformer' or unidirectional architecture
    - weight_path: Path to pretrained model weights
    - batch_aggregation: How to aggregate causal estimations over batch dimension (e.g., 'mean', only relevant if time series is longer than 600)



-----


-----
## NTSNOTEARS

Nonlinear Time Series NOTEARS extends DYNOTEARS by replacing linear relationships with neural networks to capture nonlinear temporal dependencies.

**Model**: For each variable $X_t^i$:

$$X_t^i = \sum_{j=1}^d f_{ij}(X_t^j) + \sum_{j=1}^d \sum_{\tau=1}^p g_{ij}^\tau(X_{t-\tau}^j) + \varepsilon_t^i$$

where:
- $f_{ij}$: Neural network modeling instantaneous effects (subject to acyclicity)
- $g_{ij}^\tau$: 1D CNN modeling lagged effects at lag $\tau$

**Optimization**: Same acyclicity constraint as DYNOTEARS, but applied **only** to the contemporaneous network weights $W_f$:
$$\min_{f, g} \, \mathcal{L}(f, g) + \lambda h(W_f) \quad \text{s.t.} \quad h(W_f) = 0$$
where $W_f$ is a matrix representing the $L_2$ norms of the first layers of the contemporaneous neural networks $f_{ij}$. Lagged networks $g$ are unconstrained since time prevents cyclic dependencies.

**Key innovation**: Captures nonlinear relationships while maintaining differentiability for efficient optimization.



**HPS**: 

    - max_lag: the order of the model
    - seed: Random seed for reproducibility
    - h_tol: Tolerance for acyclicity constraint
    - rho_max: Maximum value for augmented Lagrangian penalty parameter
    - lambda1: L1 regularization weight for instantaneous effects
    - lambda2: L1 regularization weight for lagged effects

-----

## SVAR-FCI

Structural VAR with Fast Causal Inference (SVAR-FCI) extends the FCI algorithm to time series data with potential latent confounders.

**Model**: Assumes a structural VAR with latent variables:

$$X_t = \sum_{\tau=1}^p B_\tau X_{t-\tau} + \Gamma L_t + \varepsilon_t$$

where $L_t$ represents unobserved confounders.

**Algorithm**:
1. **Adjacency search**: Build skeleton graph using conditional independence tests on time series
2. **Orientation**: Apply orientation rules considering possible latent confounders
3. **Output**: Returns Partial Ancestral Graph (PAG) with edges:
   - $\rightarrow$: Direct causal effect
   - $\leftrightarrow$: Latent confounder
   - $\circ$: Ambiguous orientation

**Key assumption**: Causal sufficiency not required—can handle hidden confounders.

**HPS**:

    - max_lag: Maximum time lag to consider
    - ci_test: Conditional independence test
    - alpha: Significance level for independence tests

-----

## SVAR-RFCI

Really Fast Causal Inference for Structural VAR (SVAR-RFCI) is a computationally efficient variant of SVAR-FCI.

**Key difference from SVAR-FCI**: Uses more aggressive conditional independence test ordering to reduce computational cost while maintaining correctness guarantees.

**Computational improvement**: 
- SVAR-FCI: Tests all possible separating sets up to maximum size
- SVAR-RFCI: Uses heuristics to prioritize likely separating sets

**Trade-off**: Faster runtime with same asymptotic correctness but may require more data for finite-sample accuracy.

**Output**: Same PAG structure as SVAR-FCI with uncertainty about edge orientations in presence of latent confounders.

**HPS**:

    - max_lag: Maximum time lag to consider
    - ci_test: Conditional independence test  
    - alpha: Significance level for independence tests

-----

