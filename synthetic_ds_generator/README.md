# Synthetic Time Series Generator

A comprehensive and modular synthetic time series generator for creating datasets with customizable causal structures, nonlinear relationships, and various assumption violations. This tool is designed for research in causal discovery, time series analysis, and testing the robustness of machine learning algorithms.

## 🎯 Key Features

- **Modular Design**: Individual components for different aspects of data generation that are individually seedable, allowing for counterfactual data generation.
- **A number of dataset scripts**: We provide 33 violation cases that can be gradually scaled in intensity. 
- **Individual configurations**: The generator is highly customizable and allows for multi-violation dataset generation.
- **Hydra Configuration**: Parameter management through YAML configuration files
- **Deterministic Generation**: Everything is reproducible based on random seeds.
- **Data control through hashing**: We include data hashes for the original experiments.

## 🚀 Quick Start


### 🔧 Installation and Dependencies

### Requirements
```bash
conda env create -f synth_ts.yml
```

### TCD-Arena datasets: 

In our paper, we use this generator to build **33 datasets** that violate various Causal Discovery assumptions: 
They can be directly recreated individually using the bash scripts lieing in the data_scripts folders or by running: 


```bash
./create_all_violation_datasets.sh 99 /path/to/save/folder/ #99 is the seed that we use during all experiments
```
To validate the generations, you can test them via the hashes of the original dataset: 
```bash
cd tools_and_examples/component_tests
python test_ds_determinism.sh /path/to/save/folder/ --use_saved   --only_check obs # otherwise full test
```


### Basic Usage

Generate a new dataset using the command line:
```bash
python generate_dataset.py -m name=my_dataset
```
Various features can directly specified via the command line, using Hydra configs, e.g. :

```bash
python generate_dataset.py -m name=my_dataset n_vars=7 generator.instant.link_proba=0.1 generator.obs_n.autoregressive=True generator.obs_n.snr=2 nonlinear_proba=0.2
```
You can find a summary of available functionality below.

Alternatively, use the programmatic interface:
```python
from data_generator import SyntheticDataGenerator
from components import *

# Initialize generator with components
generator = SyntheticDataGenerator(...)
ts, lagged_links, inst_links , ... = generator.get_sample()
```




## 📁 Project Structure

```
synthetic_ds_generator/
├── data_generator.py                # Main Generator class
├── generate_dataset.py              # CLI script for dataset generation
├── generator_tutorial.ipynb         # Main tutorial notebook
├── create_all_violation_datasets.sh # Script to create all violation datasets from the original publication
├── synth_ts.yml                     # Conda environment file
├── components/                      # Modular components
│   ├── lagged_effects.py               # Lagged causal structure sampling and effect generation
│   ├── noise_generator.py              # Innovation and observation noise sampler
│   ├── exog_influences.py              # Exogenous variable effects
│   ├── instantanous_effects.py         # Contemporaneous relationships sampling and effect generation
│   ├── nl_sampler.py                   # Nonlinear function sampling used for various effects
│   ├── nl_tools.py                     # Nonlinear function utilities
│   └── tools.py                        # General utility functions
├── config/                          # Hydra configuration files
│   ├── generate_dataset.yaml           # Main configuration file
│   └── generator/                      # Component-specific configs
├── data_scripts/                    # Dataset generation scripts
│   ├── main_data_scripts/              # Primary violation datasets
│   ├── double_data_scripts/            # Double violation combinations #UNFINISHED
│   └── tiny_large_data_scripts/        # Size variant datasets # UNFINISHED
├── tools_and_examples/              # Tools, tutorials, and examples
│   ├── component_tutorials/            # Jupyter notebooks demonstrating each component
│   ├── component_tests/                # Unit tests for components and dataset consistency
│   ├── calibrate_violations/           # Violation calibration (stepsize) tools 
│   ├── data_hashes/                    # Dataset hashes of the original datasets to confirm consistency.
│   ├── masks/                          # Predefined structural masks for generation purposes
│   ├── semi_synthetic_bases/           # Real-world noise patterns
│   ├── violation_depictions/           # Visualization of violations
│   ├── additional_violation_resources/ # Additional Notebooks for various things
│   ├── graphic_tools.py                # Visualization utilities
│   └── img/                           # violation graphics
```


## 🔧 Core Components

### 1. SyntheticDataGenerator (Main Wrapper)
The central class that orchestrates all components to generate synthetic time series data.

**Features:**
- Modular component integration
- Change point support for regime changes
- Masking for enforcing specific structures
- Normalization (min-max, standardization)
- Missing data simulation and interpolation
- Divergence detection and resampling 
- Confounding variable simulation

**🛠️ Parameters:**
- `rng`: Random seed used for all pseudo‑random operations to ensure reproducibility (default: 42)
- `normalize`: Normalization method applied to generated series. Supported values: 'minmax', 'standardization', or None. If 'standardization' is chosen, standardization_factor is applied to mix the standardized values with the original ones (default: None)
- `time_series_n`: Number of time steps to generate for each variable (default: 600)
- `standardization_factor`: Mix parameter to blend standardized and non-standardized time series (1 denotes full standardization) (default: 1)
- `change_points`: Sorted indices (time steps) where regime/structural changes occur. Values should be within [0, time_series_n) (default: [])
- `drop_struc_for_window`: If True, temporarily remove/disable structural links within designated windows (useful to simulate information gaps or masked intervals) (default: False)
- `nonstationary`: When True, allow time-varying coefficients or processes that may be nonstationary (e.g., drifting parameters, change-point driven parameter resampling) (default: False)
- `remove_n_variables_for_confounding`: Number of observed variables to drop from the model to simulate unobserved confounding (i.e., treated as hidden/unmeasured) (default: 0)
- `section_tolerance`: Maximum resample attempts for a single section. process return None if the limit is reached (default: 50)
- `sample_attempts`: Number of attempts to sample a valid process (runs e.g. stability checks for VAR). If sampling fails after this many attempts, an error is raised (default: 10)
- `missingness_type`: Missing data mechanism. Common choices: 'MCAR' (Missing Completely At Random), 'MAR' (Missing At Random), 'MNAR' (Missing Not At Random). Probabilities of missingness are always rescaled to match the interpolate coefficient (default: "MCAR")
- `interpolate`: Fraction of values to mask via missingness_type and then fill via interpolation (e.g., linear interpolation). Value in [0.0, 1.0]; 0.0 disables interpolation (default: 0.0)

**Component Parameters:**
- `inno_n`: Innovation noise generator component (NoiseGenerator instance)
- `obs_n`: Observation noise generator component (NoiseGenerator instance)
- `exog`: Exogenous variable generator component (ExogenousInfluences instance)
- `struc`: Structural causal model generator component (LaggedEffects instance)
- `instant`: Instantaneous effect generator component (InstantaneousEffects instance)

**Main Entry Point:**

#### `get_sample(link_mask=None, nl_mask=None, instant_link_mask=None, instant_nl_mask=None)`

The primary method for generating synthetic time series data. This method orchestrates the entire data generation process, handling change points, normalization, missing data simulation, and confounding variable removal.

**🛠️ Parameters:**
- `link_mask`: Optional numpy array to enforce specific structural links in the lagged causal graph
- `nl_mask`: Optional mask to control which lagged relationships are nonlinear  
- `instant_link_mask`: Optional mask to enforce specific instantaneous causal links
- `instant_nl_mask`: Optional mask to control which instantaneous relationships are nonlinear

**Returns:**
A tuple containing:
- `combined_ts`: Main time series data (n_vars × time_steps)
- `link_stack`: lagged structural coefficient matrices across change points (n_changepoints × n_vars × n_vars × max_lags)
- `instant_links_stack`: Instantaneous effect matrices across change points (n_changepoints × n_vars × n_vars)
- `exog_links_stack`: Exogenous variable effect matrices (or None if no exogenous effects)
- `combined_exog`: Exogenous variable time series (or None if no exogenous variables)
- `nl_namings_stack`: Nonlinear function specifications for lagged effects (or None if no nonlinear effects)
- `instant_nl_namings_stack`: Nonlinear function specifications for instantaneous effects (or None)
- `exog_nl_namings_stack`: Nonlinear function specifications for exogenous effects (or None)
- `[sample_attempt, section_attempts]`: Number of resampling attempts for debugging (Full resample, section resample)


### 2. LaggedEffects
Generates the core causal structure and lagged relationships.

**Features:**
- VAR(p) process generation with stability checking
- Linear and nonlinear relationships
- Custom lag structures
- Link probability control
- Parameter range specification
- Nonstationarity support

**🛠️ Parameters:**
- `rng`: Random seed for reproducibility (default: 42)
- `n_vars`: Number of variables in the system (default: 3)
- `max_lags`: Maximum lag order for the VAR process (default: 2)
- `link_proba`: Probability of link existence between variables (default: 0.15)
- `param_range`: Range [min, max] for linear coefficients (default: [0.3, 0.5])
- `mirror_range`: Allow negative coefficients by random sign flipping (default: False)
- `nonlinear_proba`: Probability of nonlinear relationships (default: 0.0)
- `nl_sampler`: Nonlinear function sampler object (default: None)
- `nonstationary_change`: Magnitude of coefficient changes for nonstationarity (default: 0.1)
- `test_for_var_stability`: Whether to test VAR process stability (default: True)
- `stability_tolerance`: Maximum attempts to find stable process (default: 100)
- `alternative_coeff_ts`: Number of variables with alternative coefficient ranges (default: 0)
- `alternativ_parameter_range`: Range for alternative coefficients (default: [0.3, 0.5])
- `alternative_link_proba`: Link probability for alternative coefficients (default: 0.15)



### 3. NoiseGenerator
Handles both innovation and observation noise with multiple types.

**Noise Types:**
- **Additive**: Standard Gaussian or custom distribution noise
- **Multiplicative**: Noise that scales with signal magnitude
- **Time-dependent**: Noise that varies over time
- **Salt and Pepper**: Sparse shock noise
- **Autoregressive**: AR(p) noise processes
- **Common Innovation**: Shared noise across variables (confounding)
- **Semi-synthetic**: Real-world noise patterns

**🛠️ Parameters:**
- `rng`: Random seed for reproducibility (default: 42)
- `modus`: Mode of operation - "obs" (observational) or "inno" (innovation) (default: "obs")
- `additive`: Whether to include additive noise (default: True)
- `multiplicative`: Whether to include multiplicative noise (default: False)
- `time_dependent`: Whether to include time-dependent noise (default: False)
- `autoregressive`: Whether to include autoregressive noise (default: False)
- `common`: Whether to include common innovation noise (default: False)
- `semi_synthetic`: Whether to use real-world noise patterns (default: False)
- `shock`: Whether to include shock noise (default: False)
- `shock_proba`: Probability of shock events (default: 0.05)
- `time_dependent_cycle`: Type of time-dependent scaling ("annual_sin", "increase_per_step", "mean_var_shift") (default: "mean_var_shift")
- `shock_size`: Amplitude of shocks relative to signal (default: 5)
- `snr`: Desired signal-to-noise ratio for observation noise (default: 1)
- `non_additive_noise_proba`: Probability of using non-additive noise types in innovation mode. Required that at least one non-additive noise type is set to True (default: 0)
- `autoregressive_innovation`: Autoregressive coefficient for AR noise (default: 0.5)
- `non_gaussian_additive`: Proportion of mixing between non-Gaussian  and gaussian additive noise (default: 0)
- `non_equal_variance_range`: Range for variable-specific noise variance ([low, high]) or False for equal variance (default: False)
- `scale_for_time_dependent`: Scaling factor for time-dependent noise (intensity of the dependency) (default: 0.005)
- `which_non_gaussian`: Type of non-Gaussian distribution ("weibull", "exponential", "uniform", "inv_normal"). Only used when non_gaussian_additive > 0 (default: "weibull")

### 4. ExogenousInfluences
Models external variables affecting the main system.

**Features:**
- Multiple exogenous variables
- Linear and nonlinear exogenous effects
- Memory effects for exogenous influences
- Custom exogenous variable generation

**🛠️ Parameters:**
- `rng`: Random seed for reproducibility (default: 42)
- `n_vars`: Number of system variables influenced by exogenous factors (default: 3)
- `n_exogs`: Number of exogenous variables (default: 2)
- `link_proba`: Probability of exogenous effects on system variables (default: 0.5)
- `param_range`: Parameter range for exogenous effects (default: [0.3, 0.5])
- `mirror_range`: Whether to allow negative coefficients (default: True)
- `nonlinear_proba`: Probability of nonlinear exogenous effects (default: 0.0)
- `nl_sampler`: Nonlinear function sampler object (default: None)

### 5. InstantaneousEffects
Handles contemporaneous (within-timestep) relationships.

**Features:**
- Instantaneous causal relationships
- Lower-triangular structure for causal ordering
- Linear and nonlinear instantaneous effects
- Structural change support
- Acyclicity enforcement

**🛠️ Parameters:**
- `rng`: Random seed for reproducibility (default: 42)
- `n_vars`: Number of variables in the system (default: 3)
- `link_proba`: Probability of instantaneous relationships (default: 0.5)
- `param_range`: Parameter range for instantaneous effects (default: [0.3, 0.5])
- `mirror_range`: Whether to allow negative coefficients (default: True)
- `sample_tries`: Number of attempts to sample acyclic effect matrix (default: 50)
- `nonlinear_proba`: Probability of nonlinear instantaneous effects (default: 0.0)
- `nl_sampler`: Nonlinear function sampler object (default: None)
- `nonstationary_change`: Maximum magnitude of random perturbations for nonstationarity (default: 0.1)

### 6. NL_function_generator
Generates various types of nonlinear functions.

**Function Types:**
- **Power functions**: x^α transformations
- **Spline**: Smooth spline interpolations
- **RBF**: Radial basis function transformations
- **Symbolic**: Complex compositions of multiple functions

**🛠️ Parameters:**
- `rng`: Random seed for reproducibility (default: 42)
- `nl_mode`: Type of nonlinear functions to sample ("power_set", "splines", "rbf", "symbolic", "dummy_linear") (default: "power_set")
- `limit_startx`: Range for input values where no output limiting is applied (default: [-1.0, 1.0])
- `hard_limitx`: Hard limit for output values beyond input range (default: 1.1)
- `limit_starty`: Range for output values where no output limiting is applied (default: [-1.0, 1.0])
- `hard_limity`: Hard limits for output values (default: [-1.1, 1.1])
- `power_dist`: Parameters for sampling exponents in power functions (default: [0.5, 0.6, 1.9, 2])
- `which_power_dist`: Distribution type for power functions (default: "all")
- `rbf_length_scale`: Length scale parameter for RBF kernel (default: 1)
- `spline_samples`: Number of spline control points (default: 4)
- `n_nl_stacking`: Number of nonlinear functions to stack in symbolic mode (default: 2)
- `n_nl_operators`: Number of operators to combine functions in symbolic mode (default: 2)
- `n_options`: Number of nonlinear function options to sample from (default: 10) 



## 🔦 Quality Assurance


### During Generation

The generator includes multiple validation layers:
- **Stability checking**: VAR process stability verification
- **Divergence detection**: Automatic resampling if process diverges
- **Data validation**: NaN/Inf detection and variance checks
- **Acyclicity**: Instantanous effects are garantueed to be acyclic



### Unit testing  and Dataset validity: 

The synthetic data generator includes comprehensive unit testing to ensure reproducibility, determinism, and data quality. All tests live in the repository at tools_and_examples/component_tests/


#### Available Test Scripts


**1. Dataset Determinism Testing (`test_ds_determinism.py`)**
Validates that generated datasets are identical across different runs by comparing file hashes. This is particularly useful for verifying experiment reproducibility.

```bash
cd component_tests
python test_ds_determinism.py <folder_path> --use_saved
# or compare two folders directly:
python test_ds_determinism.py <folder1> <folder2>
```

**Key Features:**
- Compares datasets using SHA256 file hashing
- Supports both single folder validation and cross-folder comparison
- Can save hash signatures for future comparisons
- Handles various file types including numpy arrays with numerical tolerance
- Provides detailed reporting on which datasets match/differ


**2. Component Determinism Testing (`test_determinism.py`)**
Tests whether individual components produce identical results when initialized with the same random seed. This ensures that each component is fully deterministic and reproducible.

```bash
cd component_tests
python test_determinism.py
```

**Key Features:**
- Tests all major components: NoiseGenerator, LaggedEffects, InstantaneousEffects, ExogenousInfluences, NL_function_generator
- Runs multiple independent tests for each component (default: 3 tests per component)
- Compares outputs using array equality checks and numerical tolerances
- Provides detailed pass/fail reporting for each component

**3. Full Generator Determinism Testing (`test_determinism_generator.py`)**
Tests the complete `SyntheticDataGenerator` class to ensure end-to-end determinism across the entire data generation pipeline.

```bash
cd component_tests
python test_determinism_generator.py
```

**Key Features:**
- Tests the complete data generation pipeline with all components integrated
- Verifies determinism across multiple configurations (different variable counts, lag structures, etc.)
- Tests with and without various features (nonlinear effects, exogenous variables, change points)
- Ensures that complex interactions between components remain deterministic


**4. Component Seeding Independence Testing (`test_component_seeding.py`)**
Tests that individual components can be seeded independently, ensuring that changing only one component's seed affects only that component's output while keeping all others deterministic.

```bash
cd component_tests
python test_component_seeding.py
# or test specific component:
python test_component_seeding.py --component struc
```

**Key Features:**
- Tests selective component variation while maintaining determinism in other components
- Verifies expected change propagation (e.g., structural changes affect time series but not observation noise)
- Enables controlled experiments where specific aspects can be varied independently
- Tests all major components: innovation noise, observation noise, structural effects, instantaneous effects, exogenous influences, and nonlinear function sampling
- Configurable seeds and numerical tolerance for flexible testing




## 🐸  Upcoming feature list: 


- Nl functionals should be reworked to be more flexible and consistent. Currently everything is centered arround 0 and often runs in the boundaries.
- More multi violation cases should be properly tested (We only consider a few violations currently.)
- Mixed observational and innovation noise (Currently only 1 can be selected for both)
- Integration of real-world time-series should be generalized

## 📚 Additional Tools and Examples available

The `tools_and_examples/` directory contains comprehensive resources for working with the synthetic data generator:

### 📝 Component Tutorials (`component_tutorials/`)
Interactive Jupyter notebooks demonstrating each component in detail:
- **`data_generator.ipynb`**: Complete walkthrough of the main generator functionality
- **`structural_equation.ipynb`**: Causal structure generation examples and VAR process demonstrations  
- **`noise_generator.ipynb`**: Comprehensive noise type demonstrations and parameter effects
- **`exogenous_effects.ipynb`**: Exogenous variable modeling and influence patterns
- **`instantanous_effects.ipynb`**: Contemporaneous relationships and acyclicity constraints
- **`nl_sampler.ipynb`**: Nonlinear function sampling and composition examples

### 🎛️ Violation Calibration (`calibrate_violations/`)
Tools for fine-tuning violation intensity and parameter scaling:
- **`calibrate_general.ipynb`**: General violation parameter calibration across all types
- **`calibrate_nl_levels.ipynb`**: Nonlinear function intensity calibration and scaling
- **`calibrate_noise_levels.ipynb`**: Noise level calibration for desired SNR and characteristics
- **`calibration_tools.py`**: Utility functions for systematic parameter exploration

### 🗂️ Validation Hashes (`data_hashes/`)
- **`ds_hashes.p`**: Pickled dictionary containing SHA256 hashes of all original TCD-Arena datasets for validation

### 🎭 Structural Masks (`masks/`)
Pre-generated masks for enforcing specific structural constraints:
- **5-variable systems**: `masks_5_*_faith_lagged_*/` for different lag-based faithfulness violations
- **7-variable systems**: `masks_7_*_faith_lagged_*/` for larger system faithfulness violations  
- **Instantaneous effects**: `masks_*_faith_instant_*/` for contemporaneous faithfulness constraints
- Various intensity levels (0-4) and configurations for systematic violation studies

### 📊 Semi-synthetic Base Data (`semi_synthetic_bases/`)
Real-world time series for semi-synthetic noise generation:
- **`gold.npy`**: Gold price time series for financial noise patterns
- **`nvidia.npy`**: NVIDIA stock data for technology sector volatility
- **`s_p.npy`**: S&P 500 index for broad market dynamics
- **`rivers_ts_flood.csv`**: River flow/flood data for environmental time series patterns
- **`real_ts/`**: Additional real-world time series from various domains

### 📈 Violation Visualizations (`violation_depictions/`)
Notebooks for understanding and visualizing different violation types:
- **`general.ipynb`**: General violation examples across all categories
- **`nl.ipynb`**: Nonlinear relationship visualizations and function galleries  
- **`noise.ipynb`**: Comprehensive noise pattern demonstrations and effects

### 🔧 Additional Resources (`additional_violation_resources/`)
Extended tools for violation analysis and data preparation:
- **`extract_semi_synthetic_noise.ipynb`**: Tools for extracting and processing real-world noise patterns
- **`generate_masks.ipynb`**: Utilities for creating custom structural constraint masks
- **`nonlinearity_assessment.ipynb`**: Quantitative tools for assessing nonlinearity strength

### 🖼️ Generated Graphics (`img/`)
Comprehensive collection of violation illustrations and method demonstrations:
- **Nonlinear violations**: `RBF_violation.pdf`, `power_violation.pdf`, `splines_violation.pdf`, `compose_violation.pdf`
- **Noise patterns**: `*_obs.pdf` files showing various observation noise types
- **Faithfulness violations**: `faith_*.pdf` files illustrating lagged and instantaneous faithfulness breaks
- **Missing data**: `missing_structure.pdf`, `m*ar.pdf` files for different missingness mechanisms
- **General violations**: Various PDFs covering all violation categories and intensities

### 🎨 Visualization Tools (`graphic_tools.py`)
Comprehensive plotting and visualization utilities for:
- Time series plotting with causal structure overlays
- Violation intensity visualization
- Comparative analysis plots
- Interactive exploration tools


## Maintainers

Main: [@GideonStein](https://github.com/Gideon-Stein)




