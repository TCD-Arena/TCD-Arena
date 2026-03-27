# Causal Discovery Zoo (cd_zoo)

This module provides a unified interface for running and benchmarking causal discovery algorithms on time series data.

⚠️Important⚠️. This is a placeholder for the full cd_zoo. It only holds the a single Causal Discovery method to display the functionality. 
If you want to use other CD methods, feel free to clown the full repo by cloning:

   ```bash 
   cd ..
   rm cd_zoo -r
   git clone https://github.com/TCD-Arena/cd_zoo.git

   ```

# If you want to integrate your own method into the framework to score robustness please follow these steps: 








## Typical Workflow
1. **Prepare Data**: Place your generated datasets in the appropriate folder (e.g., `../sample_datasets`).
2. **Run a Method**:
   ```bash
   python benchmark.py # Runs var approach (default on specified data folder )
   ```
4. **Batch Execution**:
   ```bash
   ./execute_all_methods.sh my_dataset
   ```
