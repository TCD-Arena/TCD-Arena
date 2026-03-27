import pandas as pd
from os import listdir
from os.path import isfile, join
from yaml import safe_load
import hydra
from omegaconf import DictConfig
import datetime
import os
import shutil
from pathlib import Path
import numpy as np


@hydra.main(
    version_base="1.3", config_path="config", config_name="1_extract_results.yaml"
)
def main(cfg: DictConfig):
    """
    Main function to extract and export experiment results based on a configuration.

    Args:
        cfg (DictConfig): Configuration object containing the following attributes:
            - path (str): Base path to the experiment files.
            - out_path (str): Relative path for output directories.
            - res_path (str): Relative path to the results directory.
            - method_list (list): List of methods to process.
            - load_data_hps (Any): Hyperparameters for data loading.
            - passed_test_path (str): Path to the file tracking cached data.
            - restrict_to (int): Number of experiments to process.

    Workflow:
        2. Ensures output directories ('SG_mean', 'SG_max', 'WCG', 'INST') exist.
        3. Iterates over the specified number of experiments, calling `export_experiment`
           for each, and handles exceptions by printing errors.
    """
    # Collect and filter dataset names that will be parsed based on config
    out_path = create_folder_structure(cfg)
    ready_to_parse = get_what_to_parse(cfg)

    # Build structure for output folders
    print("Ready to parse:", ready_to_parse)
    for f in ready_to_parse:
        passed_all_tests = []
        print("extracting ", f, "...")
        run_stack = export_experiment(f, cfg)
        for key in run_stack.keys():
            if cfg.automated_testing:
                passed, reason = format_testing(run_stack[key], cfg, mode=key, ds=f)
                if passed:
                    print(f"Automated tests passed for {key} in {f}")
                    run_stack[key].to_csv(
                        out_path + "extracted/" + key + "/" + f + ".csv"
                    )
                    passed_all_tests.append(f)
                else:
                    print(f"Automated tests failed for {key} in {f}: {reason}")
                    run_stack[key].to_csv(out_path + "failed/" + key + "/" + f + ".csv")
                    with open(
                        out_path + "/failed_test.txt", "a", encoding="utf-8"
                    ) as file_handler:
                        file_handler.write(
                            f"{f} ,{key}: {reason}" + "\n"
                        )  # Save the list of passed / failed experiments
            else:
                run_stack[key].to_csv(out_path + "extracted/" + key + "/" + f + ".csv")

        if len(passed_all_tests) == len(run_stack.keys()):
            print(f"All tests passed for {f}. Adding to passed_all_tests.txt")
            with open(
                out_path + "passed_all_tests.txt", "a", encoding="utf-8"
            ) as file_handler:
                file_handler.write(f + "\n")


### Helper functions for main workflow ###
def export_experiment(f, cfg):
    """
    returns dicts with pandas tables.

    """
    # pathing
    exp_path = Path(cfg.base_path) / cfg.res_path / f
    method = [method for method in listdir(exp_path)]
    # should never fail here as we filtered before

    method = [exp_path / x for x in method]

    # result dicts
    run_stack = {}
    run_stack["SG_mean"] = []
    run_stack["SG_max"] = []
    run_stack["WCG"] = []
    run_stack["INST"] = []
    data_hps_cache = {}

    for m in method:
        start = datetime.datetime.now()
        print("Start: ", m)
        runs = [m / x for x in listdir(m)]
        for run in runs:
            performance = pd.read_csv(run / "scoring.csv")
            performance.set_index("Metric", inplace=True)

            with open(run / "config.yaml", "r") as f:
                hps = pd.json_normalize(safe_load(f)).T
            if cfg.load_data_hps:
                # Get data path from performance table
                dp = Path(performance.loc["path"].values[-1])
                if dp in data_hps_cache:
                    data_hps = data_hps_cache[dp]
                else:
                    with open(dp / "config.yaml", "r") as f:
                        data_hps = pd.json_normalize(safe_load(f)).T
                        data_hps_cache[dp] = data_hps
            # seperate the columns and process them individually:
            graphs_to_export = performance.columns
            if cfg.specific_graphs is not None:
                graphs_to_export = [
                    x for x in graphs_to_export if x in cfg.specific_graphs
                ]
            for column in graphs_to_export:
                performance_column = performance.drop(
                    columns=[x for x in performance.columns if x != column]
                )
                # replace hp column name to match for concat (everything is a single column with many rows)
                hps.columns = performance_column.columns
                join = pd.concat([performance_column, hps], axis=0)
                # also add hps if requested
                if cfg.load_data_hps:
                    data_hps.columns = performance_column.columns
                    join = pd.concat([join, data_hps])
                join.columns = [run]
                if "run_name" in join.index:
                    # Drop run_name (hydra param from table as it is duplicated and prevents merge.)
                    join.drop(index="run_name", inplace=True)
                run_stack[column].append(join)
        print("End: ", datetime.datetime.now() - start)
    out_stack = {}
    for key in run_stack.keys():
        if len(run_stack[key]) == 0:
            # no instant runs for selected methods.
            continue
        else:
            experimental_table = pd.concat(run_stack[key], axis=1).T
            if key == "INST":  # Remove NaNs from runs without INST links.
                experimental_table = experimental_table.dropna(subset=["AUROC Joint"])
            out_stack[key] = experimental_table
    return out_stack


def get_what_to_parse(cfg):

    # load ready_to_parse.txt as a list:
    mypath = Path(cfg.base_path) / cfg.res_path
    ready_to_parse = [f for f in listdir(mypath) if not isfile(join(mypath, f))]
    # As this runs a while (we load a lot of files, we can skip already passed ones subfolders.)
    if cfg.ignore_passed:
        if isfile(cfg.base_path + cfg.out_path + "/passed_all_tests.txt"):
            with open(cfg.base_path + cfg.out_path + "/passed_all_tests.txt", "r") as f:
                already_done = [line.strip() for line in f.readlines()]
        else:
            f = open(
                cfg.base_path + cfg.out_path + "passed_all_tests.txt",
                "x",
                encoding="utf-8",
            )
            already_done = []
        print("Before removal of finished: ", len(ready_to_parse))
        ready_to_parse = [x for x in ready_to_parse if x not in already_done]
        print("After removal of finished: ", len(ready_to_parse))
    if cfg.specific_dataset is not None:
        ready_to_parse = [x for x in ready_to_parse if cfg.specific_dataset in x]
        print("After restriction to specific dataset: ", len(ready_to_parse))
    print("Path to parse: ", ready_to_parse)
    return ready_to_parse


def create_folder_structure(cfg):

    # Make for folders in the summarized_results folder
    # We also make a folder for failed results that didnt pass the tests (debug purposes)
    out_path = cfg.base_path + cfg.out_path
    res_path = out_path + "/extracted/"
    fail_path = out_path + "/failed/"
    fail_list = out_path + "failed_test.txt"
    success_list = out_path + "passed_all_tests.txt"

    if not os.path.exists(res_path):
        os.makedirs(res_path)
    if os.path.exists(fail_path):
        # empty the fail_path folder
        shutil.rmtree(fail_path)
    else:
        os.makedirs(fail_path)

    for folder in ["SG_mean", "SG_max", "WCG", "INST"]:
        if not os.path.exists(res_path + "/" + folder):
            os.makedirs(os.path.join(res_path, folder))
        if not os.path.exists(fail_path + "/" + folder):
            os.makedirs(os.path.join(fail_path, folder))

    # Make new failed_list
    with open(fail_list, "w", encoding="utf-8") as file_handler:
        file_handler.write("")
    # Make new failed_list
    if not os.path.exists(success_list):
        with open(success_list, "w", encoding="utf-8") as file_handler:
            file_handler.write("")
    return out_path


def format_testing(a, cfg, mode="WCG", ds="no_violation_small"):
    """
    Tests a number of formatting constraints that should hold for our experimental data.
    Returns True if all tests pass, or (False, reason) if a test fails.
    """
    # check process sizing
    size = a["n_vars"].unique()
    if "method.max_lag" not in a.columns:
        # CP has no max lag param, we set it to 1 if this occurs
        lags = 1
    else:
        lags = a["method.max_lag"].nunique()
    # for conf we got more sizes.
    if len(size) != 1:
        return (False, "Different sizes in the same file")

    if lags != 1:
        print("Warning: Different max lags detected. This can may be due to some runs not finishing.")

    # There are two length in our experiments
    a_length = len(a["generator.time_series_n"].unique()) == 2
    b_length = "length" in ds  # length violation case
    if not (a_length | b_length):
        return (False, "ts length off")

    if len(a["which_dataset"].value_counts().unique()) != 1:
        return (False, "Multiple datasets mixed together.!")

    if a["AUROC Joint"].isnull().sum() > 0:
        return (False, "NANs found in metric!")


    # Check if we dropped samples wrongly
    full_unrestricted = a["restrict_to_n_samples"] == -1
    # less samples currently for ntsnotears because of runtime.
    lower_samples = a["restrict_to_n_samples"] == 33
    if not np.all(full_unrestricted | lower_samples):
        return (False, "Ds was wrongfully restricted")

    # we need 40 samples (and 20 for the instant case as results only exist with labels
    if a["which_dataset"].nunique() not in [10, 20, 40]:
        return (False, "Some ds samples missing")

    # sometimes we use masks that determine the links.
    opt1 = len(a["generator.lagged.link_proba"].unique()) == 2
    if "link_mask_path" in a.columns:  # for link mask path, this changes.
        opt2 = (
            (len(a["link_mask_path"].unique()) == 2)
            or (len(a["link_mask_path"].unique()) == 10)
            or (len(a["link_mask_path"].unique()) == 5)
        )
    else:
        opt2 = False
    if not (opt1 or opt2):
        return (False, "lagged off")

    if mode != "INST":
        # sometimes we use masks that determine the links.
        opt1 = len(a["generator.lagged.link_proba"].unique()) == 2
        opt3 = len(a["generator.lagged.link_proba"].unique()) == 1
        if (
            "instant_link_mask_path" in a.columns
        ):  # added property.earlier runs miss it.
            opt2 = (
                (len(a["instant_link_mask_path"].unique()) == 2)
                or (len(a["instant_link_mask_path"].unique()) == 10)
                or (len(a["instant_link_mask_path"].unique()) == 5)
            )
        else:
            opt2 = False
        if not (opt1 or opt2 or opt3):
            return (False, "instant off")
    else:
        # sometimes we use masks that determine the links.
        opt1 = len(a["generator.instant.link_proba"].unique()) == 1
        if (
            "instant_link_mask_path" in a.columns
        ):  # added property.earlier runs miss it.
            opt2 = (len(a["instant_link_mask_path"].unique()) == 1) or (
                len(a["instant_link_mask_path"].unique()) == 5
            )
        else:
            opt2 = False
        if not (opt1 or opt2):
            return (False, "instant off")

    return True, "Passed"


if __name__ == "__main__":
    main()
