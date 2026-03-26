# Welcome to Time Series Causal Discovery Arena (TCD-Arena)


[![arXiv](https://img.shields.io/badge/arXiv-paper-red)](https://openreview.net/forum?id=MtdrOCLAGY) 
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)]([LICENSE](https://opensource.org/license/mit))

A comprehensive benchmarking framework for evaluating the robustness of causal discovery algorithms on time series data under various assumption violations.

## 🎯 Overview

TCD Arena is a research platform designed to systematically evaluate how causal discovery methods perform when key assumptions are violated. The framework provides:

- **🧪 Modular Synthetic Data Generation**: Advanced time series generator with 33 configurable assumption violations
- **🧪 Scalable violation intensities**: All 33 violations can be controlled in their intensity
- **🔧 Unified Method Interface**: Standardized evaluation of 15 causal discovery algorithms 
- **📊 Comprehensive Analysis Pipeline**: Automated robustness analysis and performance degradation tracking
- **🤖 Ensemble Learning**: Meta-learning approaches for improved robustness

## 🚀 Quick Start

```bash
# Clone the repository

# Clone the repository
git clone https://github.com/your-repo/tcd_arena.git
cd TCD-Arena


# Set up environment for synthetic data generation
conda env create -f tcd_arena.yml
conda activate tcd_arena
```

### Your first experiment: 

#### 1. Data 


Depending on your needs, you can choose one of three ways to access data:

1. **Download example data** (single violation, quick test):

```bash

wget https://github.com/TCD-Arena/TCD-Arena/releases/download/data/sample_datasets.zip
unzip sample_datasets
rm sample_datasets.zip
```


The example files hold the data for a single assumption violation (MCAR). All scripts in the repo can be executed with this dataset. If you want to test functionality of TCD-Arena, this is the best option.



2. **Download the full dataset** from HuggingFace (all 33 violations):

https://huggingface.co/datasets/GideonStein/TCD-Arena/tree/main

This is the full dataset we use in our publication, containing all 33 violations.
If you want to score your own method on some (or all violations) this is your best option.





3. **Regenerate the full dataset** using the synthetic data generator:

Use the scripts in `synthetic_ds_generator/` to fully regenerate the data from scratch (fully seeded). 
Hashing functionality is included to verify exact reproduction. This is the best option if you want to extend datasets, generate your own violations, or alter existing ones.






#### 2. Benchmarking


You can run a simple scoring for the GVAR approach via: 

```bash

./cd_zoo/scripts/run_var_on_mcar.sh
```

# The python entry point can be found here:
```bash
python cd_zoo/benchmark.py
```

Note: The repository includes a placeholder `cd_zoo`. To install the full `cd_zoo` with all methods, run:

```bash
rm -r cd_zoo
git clone https://github.com/TCD-Arena/cd_zoo
```



## 📁 Project Structure

```
main_repo/
├── README.md                      # This overview
├── tcd_arena.yml                  # Main environment file
├── config/                        # Pipeline configuration files
│   ├── 1_extract_results.yaml
│   ├── 2_generate_curve_analysis.yaml
│   └── 3_export_summary_tables.yaml
├── 1_extract_results.py           # Step 1: Results extraction
├── 2_generate_violation_curves.py # Step 2: Curve analysis
├── 3_export_summary_tables.py     # Step 3: Summary tables
├── plot.py                        # Plotting utilities
├── cd_zoo/                        # 🔧 Causal Discovery Methods Zoo
│   ├── benchmark.py               # Main benchmarking script
│   ├── methods/                   # Method implementations
│   ├── tools/                     # Scoring and utility functions
│   ├── config/                    # Method configurations
├── synthetic_ds_generator/        # 🧪 Synthetic Data Generation Suite
│   ├── generate_dataset.py        # Main data generation script
│   ├── create_all_violation_datasets.sh
│   ├── components/                # Modular generation components
│   ├── config/                    # Hydra configuration files
│   └── data_scripts/              # Data scripts
├── robustness_experiments/        # 📊 Robustness analysis pipeline
│   ├── README.md
├── example_saves/                 # Example outputs (pre-generated)
├── sample_datasets/               # Sample datasets for quick tests
├── sample_summary/                # Example summary outputs
```


## 🧪 1. Synthetic Data Generator (`synthetic_ds_generator/`)

Advanced time series generation with configurable violations and fully deterministic seeding.

### Key Features
- **33 Assumption Violations**: Comprehensive coverage of causal discovery assumptions
- **Modular Architecture**: Configurable structural equations, noise components, and functional relationships
- **Hydra Integration**: Full parameter control and reproducibility
- **Scalable Violation Strength**: Continuous violation severity levels


### Supported Violation Categories (examples)

| **Category**         | **Example Violation** | **Description**                       |
|----------------------|----------------------|---------------------------------------|
| Confounding          | V_conf₁              | External/internal common causes        |
| Measurement Noise    | V_obs                | Various observational noise types      |
| Faithfulness         | V_faith              | Path cancellation, near-zero coeffs    |
| Functional Form      | V_func               | Nonlinear effects                     |
| Innovation Noise     | V_inno               | Innovation noise variations            |
| Stationarity         | V_stat, V_coef       | Time-varying causal relationships      |
| Data Quality         | V_length, V_mar, ... | Data quality issues                   |

### Usage
```bash
cd synthetic_ds_generator
conda activate synth_gen_env

# Generate single dataset
python generate_dataset.py -m name=my_dataset

# Generate all violation types (paper datasets)
./create_all_violations.sh

```
See the README in `synthetic_ds_generator/` for more details.



## 🔧 2. Causal Discovery Zoo (`cd_zoo/`)

Unified interface for 15+ causal discovery algorithms with standardized evaluation protocols.



### Hyperparameter Search Space

The framework includes extensive hyperparameter grids for robust evaluation:

| Method | Key Parameters | Search Space |
|--------|----------------|--------------|
| **PCMCI** | max_lag, ci_test | {1,3,5} × {ParCorr, RobustParCorr} |
| **VAR** | max_lag, base_on | {1,3,5} × {coefficients, p_values} |
| **Dynotears** | max_lag, λ_w, λ_a, max_iter | 3×2×2×2×2 = 48 combinations |
| **NTS-Notears** | max_lag, h_tol, ρ_max, λ₁, λ₂ | 3×2×2×2×2 = 48 combinations |


### Usage
```bash
cd cd_zoo
conda activate cd_zoo_env

# Single method evaluation
python benchmark.py method=pcmci \
    data_path=../example_saves/mcar_big/var/0_2026-03-26_15-39-43-848373 \
    method.max_lag=5

# Full hyperparameter sweep on violation
python run_degradation_experiment.py \
    method=var \
    base_path=../example_saves/mcar_big/var \
    ds_name=conf_violation

# Execute all methods on dataset
# (If available)
./execute_all_methods.sh my_dataset
```


## 📊 3. Robustness Experiments Pipeline (`robustness_experiments/`)

Comprehensive analysis pipeline for evaluating method performance under assumption violations.

### Experimental Protocol

The robustness evaluation follows a systematic 4-step pipeline with quality protocols:

#### **Step 1: Result Extraction & Validation**
```bash
cd .. # from cd_zoo to main_repo
cd robustness_experiments
python ../1_extract_results.py ignore_passed=False
```
- Validates experimental results and data consistency
- Aggregates raw method outputs into summary tables  
- Documents failed runs and quality issues
- Exports structured results by violation type and graph structure

#### **Step 2: Performance Degradation Analysis**
```bash
python ../2_generate_violation_curves.py \
    what=INST \
    method_selection="[varlingam]" \
    specific_selection=faith_inst
```
- Generates violation-specific performance curves
- Tracks method degradation across violation strengths
- Creates comparative visualizations

#### **Step 3: Summary Statistics Generation**
```bash
python ../3_export_summary_tables.py \
    which_table=WCG \
    select_max_lag="[5,6]" \
    what=mean \
    performance_score="AUROC Joint"
```
- Computes aggregated performance metrics
- Generates statistical summaries across violations
- Exports tables for publication and analysis

#### **Step 4: Visualization & Graphics**
```bash
# (If available)
python 4_generate_main_graphics.py export_as=pdf
```
- Creates publication-ready figures
- Generates robustness heatmaps and degradation plots
- Exports comparative method analysis


### Data Regimes

Experiments use systematic parameter combinations for comprehensive coverage:

| Parameter | Values | Purpose |
|-----------|---------|---------|
| **Length** | 250, 1000 | Short vs. long time series |
| **Link Probability** | 0.075, 0.15 | Sparse vs. dense graphs |
| **Structure Size** | Big, Small | Scalability analysis |
| **Instantaneous Probability** | 0.0, 0.1 | With/without instant links |
| **Violation Levels** | 5 levels | Severity graduation |
| **Total Datasets** | **80 per violation** | Statistical robustness |

## 🤖 4. Ensemble Learning Suite (`ensembling_experiments/`)

Advanced meta-learning approaches for robust causal discovery.

### Ensemble Strategies

1. **Traditional Ensembles** (`5_train_ensembles.py`)
   - Simple voting and averaging schemes
   - Method combination based on performance

2. **Best Method Selection** (`7_predict_with_best_methods.py`)  
   - Adaptive method selection per dataset
   - Performance-based weighting

3. **Deep Learning Ensembles** (`8_train_deep_ensembles.py`)
   - Neural network meta-learners
   - Feature extraction from method predictions

### CausalRivers Integration

The suite includes complete integration with the CausalRivers benchmark:

```bash
cd ensembling_experiments/causalrivers

# Install CausalRivers environment
./install.sh
conda activate causalrivers

# Generate benchmark datasets
python 0_generate_datasets.py

# Run benchmarking
python benchmark.py method=var \
    data_path=product/rivers_ts_east_germany.csv \
    method.max_lag=5
```

### Real-World Validation
- **East Germany River Network**: 1000+ measurement stations
- **Bavaria River Network**: Regional validation
- **Flood Area Analysis**: Extreme weather robustness
- **Multiple Graph Sampling Strategies**: Random, confounder, close-proximity


## 🔄 Complete Workflow


### 1. Data Generation
```bash
cd synthetic_ds_generator
# Generate all paper violations
./create_all_violation_datasets.sh
# Or generate custom dataset
python generate_dataset.py -m name=custom_experiment
```


### 2. Method Evaluation
```bash
cd ../cd_zoo
# Single method
python benchmark.py method=pcmci data_path=../example_saves/mcar_big/var/0_2026-03-26_15-39-43-848373
# All methods with hyperparameter search (if available)
./execute_all_methods.sh my_dataset_name
```


### 3. Robustness Analysis
```bash
cd ../robustness_experiments
# Extract and validate results
python ../1_extract_results.py path=../cd_zoo/outputs/
# Generate performance summaries
python ../3_export_summary_tables.py which_table=WCG
# Create visualizations (if available)
python 4_generate_main_graphics.py
```


### 4. Ensemble Training (Optional)
```bash
# (If ensembling_experiments/ is present)
cd ../ensembling_experiments
# Prepare training data
python 5_transform_to_training_set.py
# Train ensemble meta-learners
python 5_train_ensembles.py
python 8_train_deep_ensembles.py
```

## 🛠️ Advanced Usage Examples


### Custom Violation Generation
```bash
cd synthetic_ds_generator
python generate_dataset.py -m \
    name=custom_conf_violation \
    generator.n_samples=100 \
    generator.length=2000 \
    generator.conf.conf_n.conf_proba=0.3 \
    generator.conf.conf_n.conf_strength=1.5
```


### Method Comparison Study
```bash
cd ../cd_zoo
# Compare methods on specific violation
python run_degradation_experiment.py \
    method=pcmci,var,dynotears \
    base_path=../example_saves/mcar_big/var \
    ds_name=nonlinear_violation \
    multirun=True
```


### Large-Scale Robustness Study
```bash
cd ../robustness_experiments
# Process all results with custom metrics
python ../1_extract_results.py \
    path=large_experiment/ \
    performance_metrics="[AUROC,F1_max,Accuracy]" \
    automated_testing=True
```


## 🔧 Configuration Management

All components use Hydra for configuration:

- **Reproducibility**: Complete parameter tracking
- **Flexibility**: Command-line overrides for any parameter
- **Modularity**: Separate configs for each component
- **Scalability**: Easy parameter sweeps and multirun support




### Tutorial: Your First Robustness Experiment

```bash
# 1. Generate dataset with confounding violation
cd synthetic_ds_generator
conda activate synth_gen_env
python generate_dataset.py -m name=conf_tutorial \
    generator.conf.conf_n.conf_proba=0.2

# 2. Evaluate methods
cd ../cd_zoo
conda activate cd_zoo_env
python benchmark.py method=pcmci \
    data_path=../rename_after_generation/conf_tutorial
python benchmark.py method=var \
    data_path=../rename_after_generation/conf_tutorial

# 3. Analyze results
cd ../robustness_experiments
python 1_extract_results.py path=../cd_zoo/outputs/
```


## 📚 Documentation

Detailed documentation is available for each component:

- **[Synthetic Data Generator](synthetic_ds_generator/README.md)**: Data generation setup and configuration
- **[CD Zoo](cd_zoo/README.md)**: Method implementations and benchmarking
- **[Robustness Experiments](robustness_experiments/README.md)**: Analysis pipeline details


## 🏆 Key Features & Benefits


### ✅ Comprehensive Coverage
- **33 Assumption Violations**: Most extensive coverage in literature
- **15+ Methods**: State-of-the-art and classical approaches
- **Multiple Graph Types**: Summary, lagged, and instantaneous causal graphs

### ✅ Research-Ready
- **Publication Workflow**: End-to-end pipeline for research papers
- **Statistical Rigor**: Built-in validation and quality protocols
- **Reproducible Results**: Complete parameter tracking and seeding

### ✅ Practical Impact  
- **Real-World Validation**: CausalRivers integration
- **Ensemble Learning**: Meta-approaches for improved robustness
- **Method Development**: Framework for testing new algorithms

### ✅ User-Friendly
- **Modular Design**: Use individual components independently  
- **Hydra Configuration**: Flexible parameter management
- **Extensive Examples**: Tutorials and usage patterns


## 🔬 Research Applications

TCD Arena supports various research directions:

- **Method Robustness Analysis**: Systematic evaluation under violations
- **Algorithm Development**: Framework for testing new methods
- **Comparative Studies**: Head-to-head method comparison
- **Ensemble Research**: Meta-learning for causal discovery
- **Real-World Validation**: Application to practical datasets


## 📊 Reproducibility & Quality

### Validation Protocols
- **Statistical Testing**: Automated result validation
- **Visual Inspection**: Graphical violation verification  
- **Cross-Validation**: Multiple data regimes and parameters
- **Error Tracking**: Comprehensive failure documentation

### Version Control
- **Complete Provenance**: All parameters and configurations tracked
- **Deterministic Seeding**: Reproducible random processes
- **Environment Isolation**: Conda-based dependency management
- **Result Archiving**: Structured output storage

## 🤝 Contributing

We welcome contributions to TCD Arena! Areas for contribution:

### New Methods
Add causal discovery algorithms to the CD Zoo:
1. Implement method wrapper in `cd_zoo/methods/`
2. Add configuration in `cd_zoo/config/method/`
3. Test with validation protocol
4. Submit pull request

### New Violations  
Extend the synthetic data generator:
1. Implement violation in `synthetic_ds_generator/components/`
2. Add configuration options
3. Validate violation effects
4. Document thoroughly

### Analysis Tools
Enhance the analysis pipeline:
1. Add scripts to `robustness_experiments/`
2. Include quality protocols
3. Provide usage examples
4. Update documentation

## 📄 Citation

If you use TCD Arena in your research, please cite:

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

## 🙋‍♀️ Support & Questions

- **Issues**: [GitHub Issues](https://github.com/your-repo/tcd_arena/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-repo/tcd_arena/discussions)  
- **Documentation**: Component-specific READMEs
- **Examples**: Tutorial notebooks and scripts

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🏗️ Development Status

TCD Arena is actively developed and maintained. Current focus areas:

- **Method Expansion**: Adding more causal discovery algorithms
- **Violation Coverage**: Extending assumption violation types
- **Performance Optimization**: Improving computational efficiency
- **Documentation**: Enhanced tutorials and examples

---

**TCD Arena**: *Advancing causal discovery through systematic robustness evaluation* 🚀 

