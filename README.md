# Time Series Causal Discovery Arena (TCD Arena)


!UNDER CONSTRUCTION!

[![arXiv](https://img.shields.io/badge/arXiv-paper-red)](https://arxiv.org/) 
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

A comprehensive benchmarking framework for evaluating the robustness of causal discovery algorithms on time series data under various assumption violations.

## 🎯 Overview

TCD Arena is a research platform designed to systematically evaluate how causal discovery methods perform when their key assumptions are violated. The framework provides:

- **🧪 Modular Synthetic Data Generation**: Advanced time series generator with 32+ configurable assumption violations
- **🔧 Unified Method Interface**: Standardized evaluation of 15+ causal discovery algorithms  
- **📊 Comprehensive Analysis Pipeline**: Automated robustness analysis and performance degradation tracking
- **🤖 Ensemble Learning**: Meta-learning approaches for improved robustness

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/your-repo/tcd_arena.git
cd tcd_arena

# Set up environment for synthetic data generation
conda env create -f synthetic_ds_generator/synth_gen_env.yml
conda activate synth_gen_env

# Generate example synthetic dataset
cd synthetic_ds_generator
python generate_dataset.py -m name=example_dataset

# Set up causal discovery environment
conda env create -f cd_zoo/cd_zoo_env.yml  
conda activate cd_zoo_env

# Run causal discovery benchmark
cd ../cd_zoo
python benchmark.py method=pcmci data_path=../rename_after_generation/example_dataset
```

## 📁 Project Structure

```
tcd_arena/
├── README.md                           # This comprehensive overview
├── synthetic_ds_generator/             # 🧪 Synthetic Data Generation Suite
│   ├── generate_dataset.py            # Main data generation script
│   ├── components/                     # Modular generation components
│   ├── config/                         # Hydra configuration files
│   └── synth_gen_env.yml              # Conda environment
├── cd_zoo/                            # 🔧 Causal Discovery Methods Zoo
│   ├── benchmark.py                   # Main benchmarking script
│   ├── methods/                       # 15+ method implementations
│   ├── tools/                         # Scoring and utility functions
│   ├── config/                        # Method configurations
│   └── cd_zoo_env.yml                 # Conda environment
├── robustness_experiments/            # 📊 Robustness Analysis Pipeline
│   ├── 1_extract_results.py           # Result extraction & validation
│   ├── 2_generate_curve_analysis.py   # Performance degradation analysis
│   ├── 3_export_summary_tables.py     # Summary statistics generation
│   ├── 4_generate_main_graphics.py    # Visualization generation
│   └── config/                        # Experiment configurations
├── ensembling_experiments/            # 🤖 Ensemble Learning Suite
│   ├── 5_train_ensembles.py           # Meta-learner training
│   ├── 7_predict_with_best_methods.py # Best method ensemble
│   ├── 8_train_deep_ensembles.py      # Deep learning ensembles
│   ├── causalrivers/                  # CausalRivers integration
│   └── dl_components/                 # Deep learning components
├── rename_after_generation/           # 📂 Generated dataset storage
└── legacy_content/                    # 📜 Legacy experimental code
```

## 🧪 1. Synthetic Data Generator (`synthetic_ds_generator/`)

Advanced time series generation with configurable violations and fully deterministic seeding.

### Key Features
- **27+ Assumption Violations**: Comprehensive coverage of causal discovery assumptions
- **Modular Architecture**: Configurable structural equations, noise components, and functional relationships
- **Hydra Integration**: Full parameter control and reproducibility
- **Scalable Violation Strength**: Continuous violation severity levels

### Supported Violations

| **Category** | **Description** |
|--------------|----------------|-----------------|
| **Confounding** | V_conf₁, V_conf₂ | External/internal common causes |
| **Measurement Noise** | V_obs | Various observational noise types |
| **Faithfulness** | V_faith| Unfaithfulness through path cancellation or near-zero coefficients|
| **Functional Form** | V_func| Nonlinear effects|
| **Innovation Noise** | V_inno | Innovation noise variations |
| **Stationarity** | V_stat,V_coef | Time-varying causal relationships |
| **Data Quality** | V_length, V_mar, ... | Data quality issues |

### Usage
```bash
cd synthetic_ds_generator
conda activate synth_gen_env

# Generate single dataset
python generate_dataset.py -m name=my_dataset

# Generate all violation types (paper datasets)
./create_all_violations.sh

```
See the README in the synthetic_ds_generator for more details.


## 🔧 2. Causal Discovery Zoo (`cd_zoo/`)

Unified interface for 15+ causal discovery algorithms with standardized evaluation protocols.

### Implemented Methods

| Method | Type | Outputs | Deep Learning | Status |
|--------|------|---------|---------------|--------|
| **PCMCI/PCMCI+** | Constraint | Summary + Window | ❌ | ✅ Validated |
| **VAR** | Granger | Summary + Window | ❌ | ✅ Validated |
| **Dynotears** | Score | Summary + Window + Instant | ❌ | ✅ Validated |
| **NTS-Notears** | Score | Summary + Window + Instant | ✅ | ✅ Validated |
| **VarLiNGAM** | Noise+Granger | Summary + Window + Instant | ❌ | ✅ Validated |
| **CDMI** | Information | Summary | ✅ | ✅ Validated |
| **Causal Pretraining** | End-to-end | Summary + Window + Instant | ✅ | ✅ Validated |
| **TCDF** | Deep Learning | Summary + Window | ✅ | ✅ Validated |
| **CUTS/CUTS+** | Deep Learning | Summary + Window | ✅ | ✅ Validated |

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
    data_path=../rename_after_generation/my_dataset \
    method.max_lag=5

# Full hyperparameter sweep on violation
python run_degradation_experiment.py \
    method=var \
    base_path=../rename_after_generation \
    ds_name=conf_violation

# Execute all methods on dataset
./execute_all_methods.sh my_dataset
```

## 📊 3. Robustness Experiments Pipeline (`robustness_experiments/`)

Comprehensive analysis pipeline for evaluating method performance under assumption violations.

### Experimental Protocol

The robustness evaluation follows a systematic 4-step pipeline with quality protocols:

#### **Step 1: Result Extraction & Validation** 
```bash
cd robustness_experiments
python 1_extract_results.py ignore_passed=False
```
- Validates experimental results and data consistency
- Aggregates raw method outputs into summary tables  
- Documents failed runs and quality issues
- Exports structured results by violation type and graph structure

#### **Step 2: Performance Degradation Analysis**
```bash
python 2_generate_curve_analysis.py \
    what=INST \
    method_selection="[varlingam]" \
    specific_selection=faith_inst
```
- Generates violation-specific performance curves
- Tracks method degradation across violation strengths
- Creates comparative visualizations

#### **Step 3: Summary Statistics Generation**
```bash
python 3_export_summary_tables.py \
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
./create_all_violations.sh
# Or generate custom dataset
python generate_dataset.py -m name=custom_experiment
```

### 2. Method Evaluation  
```bash
cd cd_zoo
# Single method
python benchmark.py method=pcmci data_path=../rename_after_generation/my_data
# All methods with hyperparameter search
./execute_all_methods.sh my_dataset_name
```

### 3. Robustness Analysis
```bash
cd robustness_experiments
# Extract and validate results
python 1_extract_results.py path=../results/
# Generate performance summaries  
python 3_export_summary_tables.py which_table=WCG
# Create visualizations
python 4_generate_main_graphics.py
```

### 4. Ensemble Training (Optional)
```bash
cd ensembling_experiments
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
cd cd_zoo
# Compare methods on specific violation
python run_degradation_experiment.py \
    method=pcmci,var,dynotears \
    base_path=../rename_after_generation \
    ds_name=nonlinear_violation \
    multirun=True
```

### Large-Scale Robustness Study
```bash
cd robustness_experiments
# Process all results with custom metrics
python 1_extract_results.py \
    path=large_experiment/ \
    performance_metrics="[AUROC,F1_max,Accuracy]" \
    automated_testing=True
```

## 📊 Performance Metrics

TCD Arena provides comprehensive evaluation metrics:

- **AUROC**: Area under ROC curve (primary metric)
- **F1-Max**: Maximum F1 score across thresholds  
- **Accuracy**: Maximum accuracy across thresholds
- **Individual vs Joint**: Per-sample and aggregated metrics
- **Null Model Comparison**: Baseline performance validation

## 🔧 Configuration Management

All components use Hydra for configuration:

- **Reproducibility**: Complete parameter tracking
- **Flexibility**: Command-line overrides for any parameter
- **Modularity**: Separate configs for each component
- **Scalability**: Easy parameter sweeps and multirun support
## 🎓 Getting Started Guide

### Prerequisites
- Python 3.8+
- Conda package manager
- Git

### Installation

1. **Clone Repository**
```bash
git clone https://github.com/your-repo/tcd_arena.git
cd tcd_arena
```

2. **Set Up Environments**
```bash
# Synthetic data generation environment
conda env create -f synthetic_ds_generator/synth_gen_env.yml

# Causal discovery methods environment  
conda env create -f cd_zoo/cd_zoo_env.yml

# Optional: CausalRivers integration
cd ensembling_experiments/causalrivers
./install.sh
```

3. **Test Installation**
```bash
# Generate test dataset
cd synthetic_ds_generator
conda activate synth_gen_env
python generate_dataset.py -m name=test_install

# Run test benchmark
cd ../cd_zoo  
conda activate cd_zoo_env
python benchmark.py method=var data_path=../rename_after_generation/test_install
```

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
- **[Ensemble Learning](ensembling_experiments/README.md)**: Meta-learning approaches

## 🏆 Key Features & Benefits

### ✅ Comprehensive Coverage
- **27+ Assumption Violations**: Most extensive coverage in literature
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


# Now we use