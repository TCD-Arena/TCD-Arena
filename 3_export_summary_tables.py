import pandas as pd
import hydra
from omegaconf import DictConfig
from plot import get_small_big_sets
from os import listdir
from os.path import isfile, join
from pathlib import Path
import numpy as np
from plot import performance_scores




@hydra.main(
    version_base="1.3",
    config_path="config",
    config_name="3_export_summary_tables.yaml",
)
def main(cfg: DictConfig):
    """
    We currently export the following statistics into smaller files: 
    1. Optimal performance: The best performance for each method if we select the best hyperparameters per violation
    2. Highest HP performance: The best single hyperparameter combination that has the highest total performance over all violations.
    3. Average and STD HP performance: Mean and STD performance of different HP runs
    4. HP-dependent performance table: For each HP combination, a single performance value per violation (wide table)
    If more statistics are needed later we can easily add them here.
    """

    # Build folder structure for output if necessary
    out_path = prepare_folder_structure(cfg)

    # Load all paths available
    path = Path(cfg.path) / cfg.which_table
    res_files = [str(path / f) for f in listdir(path) if isfile(join(path, f))]

    # Group the small and big regimes together in one table.
    if cfg.no_size_grouping:
        grouped_paths = [[x] for x in res_files if cfg.filter_datasets in x] # filter by size if no grouping is activated.
    else:
        grouped_paths = get_small_big_sets(res_files)
    # Load the raw files for all violations for later processing.
    raw_stack = load_grouped_paths(grouped_paths, cfg)
    # Now we create a dict with the structure violation -> method -> performance table (HPs as index).
    hp_split_stack = {}
    for key in raw_stack.keys():
        # Drop other performance scores
        a = raw_stack[key].reset_index(drop=True)
        to_drop = [
            col
            for col in a.columns
            if any(score in col for score in performance_scores)
        ]
        to_drop = [x for x in to_drop if x != cfg.performance_score]
        a.drop(columns=to_drop, inplace=True)
        # Drop the max_lag columns as we we want to mean over the sizes with different lags
        # (not a real method HP as it is determined by the process sizing in our experiments)
        if "method.max_lag" in a.columns:
            # Remove max lag as it is otherwise keeping the differently sizes sets apart.
            a.drop(columns=["method.max_lag"], inplace=True)
        # Create the performance table for this violation for each HP combo.
        # We calculate the statistic over all data regimes and violation intensities
        hp_split_stack[key] = extract_performance(
            a, cfg.performance_score, what=cfg.what
        )

    # Quick check if we got the same methods in all violations (This should always be the case):
    test_stack = []
    for violation in hp_split_stack.keys():
        test_stack.append(tuple(hp_split_stack[violation].keys()))

    assert len(set(test_stack)) == 1, str(set(test_stack)) + " Not all violations have the same methods!"
    methods_to_process = test_stack[0]
    print("Methods to process:", methods_to_process)
    print("Exporting to:", out_path)


    
    # 1. Optimal performance:
    extract_optimal_performance(
        hp_split_stack,
        methods_to_process,
        out_path,
        performance_minimize=(cfg.performance_score in ["SHD individual", "SHD mean","SHD Joint"]),
    )
    

    # 2: Highest HP performance:
    highest_hp_performance(
        hp_split_stack,
        methods_to_process,
        out_path,
        performance_minimize=(cfg.performance_score in ["SHD individual", "SHD mean","SHD Joint"]),
    )
    
    # 3: Average and Std HP performance:
    calc_mean_std_performance(
        hp_split_stack,
        methods_to_process,
        out_path
    )
    
    
    # 4: HP-dependent performance table
    export_hp_dependent_performance_table(
        hp_split_stack,
        methods_to_process,
        out_path
    )
    print("Done.")
    
    
    


### HELPER FUNCTIONS ###


def export_hp_dependent_performance_table(hp_split_stack, methods_to_process, out_path):
    """
    Export a table where each row is a unique HP combination (per method),
    each column is a violation, and the value is the performance for that HP combo on that violation.
    Saves one CSV per method for clarity.
    """
    for method in methods_to_process:
        # Collect all HP combos across violations
        violations = list(hp_split_stack.keys())
        hp_tables = [hp_split_stack[v][method] for v in violations]
        # Get union of all HP indices
        all_hp_index = hp_tables[0].index
        for tbl in hp_tables[1:]:
            all_hp_index = all_hp_index.union(tbl.index)
        # Build a DataFrame: rows=HP combos, cols=violations
        perf_df = pd.DataFrame(index=all_hp_index, columns=violations)
        for v, tbl in zip(violations, hp_tables):
            perf_df.loc[tbl.index, v] = tbl.values
        # Save to CSV
        perf_df.to_csv(out_path / f"hp_dependent_performance_{method}.csv")
        print(f"Exported HP-dependent performance table for {method} to", out_path / f"hp_dependent_performance_{method}.csv")



def prepare_folder_structure(cfg):
    if not Path(cfg.out_path).exists():
        Path(cfg.out_path).mkdir(parents=True)

    if not (Path(cfg.out_path) / cfg.what).exists():
        (Path(cfg.out_path) / cfg.what).mkdir(parents=True)

    if cfg.select_max_lag == [3, 4]:
        name = "correct_lag"
    elif cfg.select_max_lag == [5, 6]:
        name = "high_lag"
    elif cfg.select_max_lag == [1, 2]:
        name = "low_lag"

    out_path = (
        Path(cfg.out_path) / cfg.what / name / cfg.which_table / cfg.performance_score
    )
    print(out_path)
    if not out_path.exists():
        out_path.mkdir(parents=True)
    return out_path


def load_grouped_paths(grouped_paths, cfg):
    """
    Load the grouped paths and return a list of DataFrames.
    """
    raw_stack = {}
    for p in grouped_paths:
        stack = []
        for item in p:
            stack.append(pd.read_csv(item, index_col=0, low_memory=False))
        stack = pd.concat(stack)
        print(len(stack), "entries loaded for", p[0])
        if not cfg.no_size_grouping:
            key = "_".join(p[0].split("/")[-1].split("_")[:-1])
        else: # keep sizing
            key = p[0].split("/")[-1][:-4]

        # CP is limited to 5 vars so we drop results for higher vars to not skew the results.
        # We note this in the paper
        stack = stack.drop( # drop the 7 var + cases (6 for the confounding example is still fine)
            stack[(stack["method.name"] == "cp") & (stack["n_vars"] > 6)].index
        )
        # Drop methods that we want to ignore. Not used in the main paper but for extended analysis.
        stack = stack.drop((stack[stack["method.name"].isin(cfg.drop_methods)]).index)
        # Filter by select_max_lag for small and big
        # We always keep cp results as they do not use a max lag at all.
        condition1 = stack["method.max_lag"].isnull()
        condition2 = stack["method.max_lag"].isin(cfg.select_max_lag)
        stack = stack[condition1 | condition2]
        # save into dict for later processing.
        raw_stack[key] = stack
    print(len(raw_stack), "raw stacks exported.")
    print("Length of tables:", [(x, len(raw_stack[x])) for x in raw_stack.keys()])
    return raw_stack


def extract_performance(data, performance_score, what="mean"):
    """
    Extracts the "what" performance for each method,
    returns hp stacks and ultimate performance.
    """
    
    if "runtime" in data.columns:
        # Transfer runtime to seconds.
        data["runtime"] = pd.to_timedelta(data["runtime"])
        data["runtime"] = data["runtime"].dt.total_seconds()

    full_hp_dict = {}
    for method_name in data["method.name"].unique():
        method = data[data["method.name"] == method_name]

        # drop all columns that are irrelevant but keep the performance score in any case.
        rel = method.nunique() > 1
        if not rel[performance_score]:
            rel[performance_score] = True
        method = method.loc[:, rel]

        method_hps = [x for x in method.columns if "method" in x]
        # we need this for ci based methods. Test is placed in the general config
        if "ci_test._target_" in method.columns:
            method_hps += ["ci_test._target_"]
        if len(method_hps) == 0:  # no HP methods (direct cross corr or tsfci)
            if what == "mean":
                performance = method.mean(numeric_only=True)[performance_score]
            elif what == "min":
                performance = method.min(numeric_only=True)[performance_score]
            elif what == "max":
                performance = method.max(numeric_only=True)[performance_score]
            # Make a table out of it
            performance = pd.Series([performance], index=[method_name])
        else:
            if what == "mean":
                performance = method.groupby(method_hps).mean(numeric_only=True)[
                    performance_score
                ]
            elif what == "min":
                performance = method.groupby(method_hps).min(numeric_only=True)[
                    performance_score
                ]
            elif what == "max":
                performance = method.groupby(method_hps).max(numeric_only=True)[
                    performance_score
                ]
        full_hp_dict[method_name] = performance.sort_values(ascending=False)
    return full_hp_dict



def highest_hp_performance(hp_split_stack, methods_to_process, out_path, performance_minimize=False):
    # 2. Highest HP performance and the selected HPS:
    highest_individual_performance = []
    keep_optimal_hps = {}
    for method in methods_to_process:
        # Concatenate HP performance tables for each violation for this method
        violations = hp_split_stack.keys()
        hp_performances = pd.concat(
            [hp_split_stack[v][method] for v in violations], axis=1
        )
        hp_performances.columns = violations
        # Find the best HP combination based on mean performance across violations
        best_hp_combo = (
            hp_performances.mean(axis=1).sort_values(ascending=performance_minimize).index[0]
        )
        print(f"Best HP combo for {method}: {best_hp_combo}")
        # Store the optimal HPs for each method individually
        keep_optimal_hps[method] = pd.DataFrame(
            [best_hp_combo], columns=hp_performances.index.names
        )

        # Extract the performance for the best HP combination
        highest_hp_selection = hp_performances.loc[best_hp_combo]
        highest_hp_selection.name = method
        highest_individual_performance.append(highest_hp_selection)
    # Export the selected HPs for each method to CSV
    # Combine all HPs with a method column
    combined_hps = pd.concat(
        [df.assign(method=key) for key, df in keep_optimal_hps.items()],
        ignore_index=True
    )
    combined_hps.set_index("method", inplace=True)
    combined_hps.to_csv(out_path / "selected_hps.csv", index=True)
        
    # Concatenate all methods' highest HP performances into one table and export
    highest_individual_performance = pd.concat(highest_individual_performance, axis=1)
    print("Highest performance table:")
    print(highest_individual_performance)
    highest_individual_performance.to_csv(out_path / "highest_hps.csv")
    
    
    
def extract_optimal_performance(
    hp_split_stack, methods_to_process, out_path, performance_minimize=False
):
    optimal_performance = []
    # Loop over each method to process its optimal performance per violation
    for method in methods_to_process:
        opti_per_violation = []
        # For each violation, get the best (highest) performance for the method
        for violation in hp_split_stack.keys():
            # Sort HP results for this method/violation, take the top value
            opti = (
                hp_split_stack[violation][method]
                .sort_values(ascending=performance_minimize)
                .values[0]
            )
            opti_per_violation.append([opti, violation])
                
        # Convert results to DataFrame for this method
        opti_per_violation = pd.DataFrame(opti_per_violation)
        opti_per_violation.set_index(1, inplace=True)
        opti_per_violation.index.name = "Violation"
        opti_per_violation.columns = [method]
        optimal_performance.append(opti_per_violation)
    # Concatenate all methods' DataFrames into one table
    optimal_performance = pd.concat(optimal_performance, axis=1)
    # Export the optimal performance table to CSV
    print("Optimal performance table:")
    print(optimal_performance)
    optimal_performance.to_csv(out_path / "optimal_performance.csv")
    
    
def calc_mean_std_performance(hp_split_stack, methods_to_process, out_path):
    # 3: Average and Std HP performance:
    mean_std = []
    for method in methods_to_process:
        # Concatenate HP performance tables for each violation for this method
        violations = hp_split_stack.keys()
        hp_performances = pd.concat(
            [hp_split_stack[v][method] for v in violations], axis=1
        )
        hp_performances.columns = violations
        # mean over all violations per HPs to get the average performance per HP
        # Note we report the effect of different HPs on the general performance over all violations. 
        # Thats why we calculate the std over these means.
        hp_performances = hp_performances.mean(axis=1)
        # Then calculate the mean and std of the HP performances to get the mean and std performance for each HP.
        mean_std.append((method, hp_performances.mean(), hp_performances.std()))
    mean_std = pd.DataFrame(mean_std, columns=["Method", "Mean", "Std"])
    print("Mean STD table:")
    print(mean_std)
    mean_std.to_csv(out_path / "mean_std_hps.csv")



if __name__ == "__main__":
    main()
