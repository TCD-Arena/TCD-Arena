# Welcome to Time Series Causal Discovery Arena (TCD-Arena)


[![arXiv](https://img.shields.io/badge/arXiv-paper-red)](https://openreview.net/forum?id=MtdrOCLAGY) 
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)]([LICENSE](https://opensource.org/license/mit))

A comprehensive benchmarking framework for evaluating the robustness of causal discovery algorithms on time series data under gradually more severe assumption violations.


## 🎯 Overview

TCD-Arena contains the following core features:

- **🧪 Modular Synthetic Data Generation**: Advanced time series generator with 33 configurable assumption violations (May be repurposed as a prior distribution). See the [Generator README](synthetic_ds_generator/README.md) for details.
- **🧪 Scalable violation intensities**: All 33 violations can be controlled in their intensity
- **🔧 Unified Method Interface**: Standardized evaluation of 15 causal discovery methods. 
- **📊 Comprehensive Analysis Pipeline**: Streamlined experimental scripts to evaluate assumption violation robustness


All components use Hydra for configuration allowing for Hyperparameter sweeps and large scale experiments. 





### Supported Violation Categories (examples)

| **Category**         | **Example Violation** | **Description**                       |
|----------------------|----------------------|---------------------------------------|
| Confounding          | V_conf              | External/internal common causes        |
| Measurement Noise    | V_obs                | Various observational noise types      |
| Faithfulness         | V_faith              | Path cancellation, near-zero coeffs    |
| Functional Form      | V_func               | Nonlinear effects                     |
| Innovation Noise     | V_inno               | Innovation noise variations            |
| Stationarity         | V_stat, V_coef       | Time-varying causal relationships      |
| Data Quality         | V_length, V_mar, ... | Data quality issues                   |



## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/your-repo/tcd_arena.git
cd TCD-Arena


# Set up environment for synthetic data generation
conda env create -f tcd_arena.yml
conda activate tcd_arena
```

## 👩‍🔬 Your first experiment: 

### 1. Data 


Depending on your needs, you can choose one of three ways to access data:

1. **Download example data** (single violation, quick test):

```bash

wget https://github.com/TCD-Arena/TCD-Arena/releases/download/data/sample_datasets.zip
unzip sample_datasets
rm sample_datasets.zip
```


The example files hold the data for a single assumption violation (MCAR). All scripts in the repo can be executed with this dataset.
If you want to test functionality of TCD-Arena, this is the best option.



2. **Download the full dataset (~7.3GB** from HuggingFace (all 33 violations):

```bash
wget https://huggingface.co/datasets/GideonStein/TCD-Arena/resolve/main/data_release.zip
unzip data_release
rm data_release.zip
```


This is the full dataset we use in our publication, containing all 33 violations.
If you want to score your own method on some (or all violations) this is your best option.



3. **Regenerate the full dataset** using the synthetic data generator:
Use the scripts in `synthetic_ds_generator/` to fully regenerate the data from scratch (fully seeded). 
Hashing functionality is included to verify exact reproduction.
 This is the best option if you want to extend datasets, generate your own violations, or alter existing ones.
See the [Generator README](synthetic_ds_generator/README.md) for details.





### 2. Benchmarking (Prediction):

You can run a simple execution of the the GVAR approach for a violation like this: 

```bash
# For the test example
./cd_zoo/scripts/run_var_on_mcar.sh
```

```bash
# For the full dataset. Note, to make processing easier, you have 2 command line arguments 
# 1. only process the first n runs 
# 2. provide a string to filter violation names. 
./cd_zoo/scripts/run_var_on_data_release.sh 20 conf
```

The python entry point can be found here:
```bash
python cd_zoo/benchmark.py
```

Note: The repository includes a placeholder `cd_zoo`. To install the full `cd_zoo` with all methods, run:

```bash
rm -r cd_zoo
git clone https://github.com/TCD-Arena/cd_zoo
```


### 3. Evaluation (Extract statistics):

The results of the benchmarking can be analyzed by running the following scripts:

- **1_extract_results.py**: Extracts and validates raw experimental results, aggregates outputs from all methods, and organizes them into structured summary tables. Handles quality checks and documents failed runs.
- **2_generate_violation_curves.py**: Processes the extracted results to generate performance degradation curves for each method and violation type, visualizing how method performance changes as violation severity increases.
- **3_export_summary_tables.py**: Computes and exports summary statistics (such as optimal, average, and hyperparameter-dependent performance) across all methods and violations, producing tables suitable for publication and further analysis.


```bash
# Extract and validate results
python 1_extract_results.py
# Generate performance summaries
python 2_generate_violation_curves.py
# Create visualizations (if available)
python 3_export_summary_tables.py
```

With the this functionality, you can recreate the performance of GVAR on the MCAR violation (NSHD: 0.81, See p. 52 in the main paper)
If you are interested in recreating any other results please download the Full CD_zoo first: [Full CD ZOO](synthetic_ds_generator/README.md) for details.




## 🛠️ Details



###  📁 Project Structure

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


### 📊 Generating custom datasets:


You can generate custom data by running:

### Custom Violation Generation
```bash
cd synthetic_ds_generator
python generate_dataset.py -m \
    name=custom_violation_ds \
    generator.n_samples=100 \
    generator.length=2000 \
    generator.obs_n.common=True \
    generator.obs_n.snr=0.5 \
    generator.interpolate=0.25,0.5 \
    generator.missingness_type=MAR \
```

This would generate 2 datasets with different interpolation levels, 100 samples each
See the [Generator README](synthetic_ds_generator/README.md) for details.



### 📚 Documentation

Detailed documentation is available for each component:

- **[Synthetic Data Generator](synthetic_ds_generator/README.md)**: Data generation setup and configuration
- **[CD Zoo](cd_zoo/README.md)**: Method implementations and benchmarking
- **[Robustness Experiments](robustness_experiments/README.md)**: Analysis pipeline details




## 🤝 Contributing

We welcome contributions to TCD Arena! Areas for contribution:

### New Methods
Add causal discovery algorithms to the CD Zoo:
1. Implement method wrapper in `cd_zoo/methods/`
2. Add configuration in `cd_zoo/config/method/`
3. Test with validation protocol
4. Submit pull request  full CD_ZOO)

### New Violations  
Extend the synthetic data generator:
1. Implement violation in `synthetic_ds_generator/components/`
2. Add configuration options
3. Validate violation effects

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

