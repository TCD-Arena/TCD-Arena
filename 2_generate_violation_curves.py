import pandas as pd
from os import listdir
from os.path import isfile, join
import numpy as np
import hydra
from omegaconf import DictConfig
import matplotlib.pyplot as plt
import os
from plot import (
    return_violation_property_and_order,
    performance_scores,
    matplot_styling,
    rename_labels,
    color_map,
    ordering,
    renames,
    get_small_big_sets
)
from pathlib import Path

# ignore pandas warnings:
pd.options.mode.chained_assignment = None  # default='warn'



@hydra.main(
    version_base="1.3",
    config_path="config",
    config_name="2_generate_curve_analysis.yaml",
)
def main(cfg: DictConfig):
    """
    This script takes the output of 1_extract_results.py to build violation curves
    for all methods, violations and graph structures.

    """
    # Some styling for matplotlib
    matplot_styling()

    # Prep paths and load folder names
    to_process, in_path, out_path = prep_paths(cfg)

    # Specify how the violation is ordered. This is not always ascending!
    violation_property_dict = return_violation_property_and_order(to_process)

    # Join the small and big violation sets together.
    to_process = get_small_big_sets(to_process)
    print("To process after pairing small and big: ", len(to_process))

    for n, item in enumerate(to_process):
        print("Processing item: ", item)
        out_stacks = []
        to_considers = []

        # Get all the violations curves for all sizes and ordered by method.
        for subsets in item:
            viol_prop, ascending = violation_property_dict[subsets]

            out, to_cons = get_violation_curves(
                p=in_path / subsets,
                ds_name = subsets,
                lag_spec=[3, 4],
                violation_property=viol_prop,
                performance_score=cfg.performance_score,
                ascending=ascending,
            )
            out_stacks.append(out)
            to_considers.append(to_cons)
        # check if all sizes have the same methods in the same order
        assert len(set((str(x) for x in to_considers))) == 1, (
            "Missmatch in methods between sizes."
        )


        # Order methods based on predefined ordering
        fig, axs = build_base_plots(item, to_considers, viol_prop, rename_labels, cfg)

        # plot the full stack
        alphas = [0.9,0.5] # alpha for small and big
        for i in range(len(out_stacks)):
            plot_each_curve(to_considers, axs, out_stacks[i], color_map, renames,alpha=alphas[i])

        # save figure
        title = item[0].replace("_small.csv", "").replace("_big.csv", "")
        fig.savefig(os.path.join(out_path / (title + "." + cfg.format)), bbox_inches="tight", dpi=500)
        plt.close(fig)

        # Remember done files
        with open(
            out_path / "curve_exported.txt", "a", encoding="utf-8"
        ) as file_handler:
            for i in item:
                file_handler.write(i + "\n")
    print("Done.")


### HELPER FUNCTIONS ###


def get_violation_curves(
    p="main_experiment/summarized_results/WCG/obs_mul_n_big.csv",
    ds_name="obs_mul_n",
    performance_score="AUROC individual",
    violation_property="generator.obs_n.snr",
    ascending=False,
    lag_spec=[3, 4],
):
    """
    Inputs:
    p: path to csv file with experiment results
    performance_score: performance score to consider
    violation_property: property to plot that shows the violation severity
    ascending: whether the violation property is ordered ascendingly or descendingly
    method_selection: list of methods to consider
    edge_case: specific edge case handling
    lag_spec: list of lags to consider


    returns:
    out_stack: dict of method names to list of violation curves
    to_consider: list of methods considered
    """
    out_stack = {}

    a = pd.read_csv(p, index_col=0, low_memory=False)

    # remove nonunique columns but keep method.name
    to_drop = (a.nunique() > 1)
    to_drop["method.name"] = True
    a = a.loc[:, to_drop]
    # Drop all columns that contain any string from performance_scores
    to_drop = [
        col for col in a.columns if any(score in col for score in performance_scores)
    ]
    # keep path for masked violations as its dropped by the "path performance score" rule
    to_drop = [x for x in to_drop if (x != performance_score) and ("mask_path" not in x) and ("method.name" not in x)]
    a.drop(columns=to_drop, inplace=True)
    to_consider = a["method.name"].unique()

    # iterate through methods individually
    for method_name in to_consider:
        method = a[a["method.name"] == method_name]
        method = handle_unique_edge_cases(method, violation_property,ds_name)
        # drop all columns that are irrelevant but keep the performance score in any case.
        rel = method.nunique() > 1
        if not rel[performance_score]:
            rel[performance_score] = True
        method = method.loc[:, rel]
        # get all method hps and group by them to get the individual curves
        method_hps = [
            x
            for x in method.columns
            if x not in ["which_dataset", performance_score, violation_property]
        ]
        # group and exract performance score as individual values
        performance = method.groupby(method_hps)
        snr_auroc_groups = [
            group[[violation_property, performance_score]] for _, group in performance
        ]
        violation_curves = [
            x.sort_values(by=violation_property, ascending=ascending)[
                performance_score
            ].values
            for x in snr_auroc_groups
        ]
        # stack everything.
        out_stack[method_name] = violation_curves
    return out_stack, to_consider


def prep_paths(cfg):
    in_path = Path(cfg.path) / cfg.what
    out_path = Path(cfg.out_path) / cfg.what
    print("Input path: ", in_path)
    print("Output path: ", out_path)
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    if not os.path.exists(out_path / "curve_exported.txt"):
        with open(out_path / "curve_exported.txt", "w") as f:
            f.write("")

    to_process = [f for f in listdir(in_path) if isfile(join(in_path, f))]
    # remove already done files
    if cfg.ignore_finished:
        with open(out_path / "curve_exported.txt", "r") as f:
            already_done = [line.strip() for line in f.readlines()]

        print("Before removal of finished: ", len(to_process))
        to_process = [x for x in to_process if x not in already_done]
        print("After removal of finished: ", len(to_process))
    # filter specific dataset if given
    if cfg.specific_dataset:
        to_process = [x for x in to_process if cfg.specific_dataset in x]
        print(
            "Specific selection: ",
            cfg.specific_dataset,
            " found ",
            len(to_process),
            " files.",
        )

    return to_process, in_path, out_path


def handle_unique_edge_cases(method, violation_property, ds_name):
    """
    Handles unique edge cases where the property cannot be directly ordered.

    Inputs:
    method: dataframe of method to process
    violation_property: property to plot that shows the violation severity
    edge_case: specific edge case naming

    Returns:
    method: modified dataframe

    """
    
    

    if "empty" in ds_name   and (
        violation_property == "generator.change_points"
    ):
        mapping = {
            "[24, 488, 512, 976]": 5,
            "[40, 100, 150, 210]": 1,
            "[6, 122, 128, 244]": 5,
            "[56, 468, 532, 944]": 4,
            "[31, 106, 144, 219]": 2,
            "[14, 117, 133, 236]": 4,
            "[23, 111, 139, 227]": 3,
            "[124, 424, 576, 876]": 2,
            "[160, 400, 600, 840]": 1,
            "[92, 444, 556, 908]": 3,
        }

        cp_series = method["generator.change_points"].astype(str)
        mapped = cp_series.map(mapping)
        # If a value wasn't found in mapping, keep the original string; otherwise use the mapped integer.
        method.loc[:, "generator.change_points"] = mapped.where(
            mapped.notnull(), cp_series
        )

    if "stat" in ds_name and (
        violation_property == "generator.change_points"
    ):  # edge case for the faithful case

        mapping = {
            "[50, 100, 150, 200]": 4,
            "[83, 166]": 2,
            "[250, 500, 750]": 3,
            "[166, 333, 500, 666, 833]": 5,
            "[200, 400, 600, 800]": 4,
            "[63, 126, 187]": 3,
            "[41, 82, 122, 163, 205]": 5,
            "[125]": 1,
            "[500]": 1,
            "[333, 666]": 2,
        }

        cp_series = method["generator.change_points"].astype(str)
        mapped = cp_series.map(mapping)
        # If a value wasn't found in mapping, keep the original string; otherwise use the mapped integer.
        method.loc[:, "generator.change_points"] = mapped.where(
            mapped.notnull(), cp_series
        )

    if ("faith_la" in ds_name) and (
        violation_property == "link_mask_path"
    ):  # edge case for the faithful case
        method.loc[:, "full"] = (
            method.copy().loc[:, "link_mask_path"].str.contains("_4_").values
        )
        method.loc[:, "link_mask_path"] = (
            method.copy().loc[:, "link_mask_path"].str[-2:].values
        )
    if ("faith_inst" in ds_name) and (
        violation_property == "instant_link_mask_path"
    ):  # edge case for the faithful case
    
        method.loc[:, "instant_link_mask_path"] = (
            method.copy().loc[:, "instant_link_mask_path"].str[-2:].values
        )
    if ("faith_z" in ds_name):
        method.drop(columns=["generator.instant.param_range"], inplace=True)
        
    return method


def build_base_plots(item, to_considers, viol_prop, rename_labels, cfg):
    # Build the base_plot plots
    length = len(to_considers[0])
    
    # Determine layout orientation from config (default to horizontal)
    vertical_layout = getattr(cfg, 'vertical_layout', False)
    
    if vertical_layout:
        # nx1 vertical stack
        fig, axs = plt.subplots(
            length, 1, figsize=(2, length * 1.25), sharex=True, sharey=True
        )
    else:
        # 1xn horizontal stack (default)
        fig, axs = plt.subplots(
            1, length, figsize=(length * 2, 1.25), sharex=True, sharey=True
        )
    
    # Ensure axs is always iterable (in case of single subplot)
    if length == 1:
        axs = [axs]
    
    for n in range(length):
        # Only set xlabel for the last (bottom-most or right-most) axis
        if vertical_layout:
            if n == length - 1:
                axs[n].set_xlabel("Violation Step", fontsize=12)
        else:
            axs[n].set_xlabel("Violation Step", fontsize=12)
        axs[n].set_title(renames[to_considers[0][n]], fontsize=11)
    title = rename_labels[item[0].replace("_small.csv", "").replace("_big.csv", "")]
    
    # Only add suptitle for horizontal layout
    if not vertical_layout:
        fig.suptitle(title, fontsize=18, y=1.30)
    
    # Add ylabel with downarrow (↓ indicates lower is better for most metrics)
    ylabel_text = cfg.performance_score.split(" ")[0] + " [↓]"
    
    if vertical_layout:
        axs[0].set_ylabel(ylabel_text, fontsize=12)
        axs[1].set_ylabel(ylabel_text, fontsize=12)
        axs[2].set_ylabel(ylabel_text, fontsize=12)
    else:
        axs[0].set_ylabel(ylabel_text, fontsize=12)
    axs[0].set_xticks([0, 1, 2, 3, 4])
    axs[0].set_xticklabels([1, 2, 3, 4, 5])
    if vertical_layout: 
        axs[0].set_yticks([0,0.5,1])
        axs[0].set_yticklabels([0,0.5,1])
    else:
        axs[0].set_yticks([0.5,1])
        axs[0].set_yticklabels([0.5,1])
    axs[0].set_ylim(0, 1)
    for ax in axs:
        ax.tick_params(axis="both", labelsize=12)
    
    if vertical_layout:
        plt.subplots_adjust(hspace=0.35)  # Increased spacing to prevent overlap
    else:
        plt.subplots_adjust(wspace=0.05)

    return fig, axs


def plot_each_curve(to_considers, axs, fuse_for_plotting, c, renames,alpha=0.8):
    
    for n, method in enumerate(to_considers[0]):
        axs[n].plot(
            np.array(fuse_for_plotting[method]).T,
            linestyle="solid",
            color=c[renames[method]],
            linewidth=1,
            alpha=alpha,
        )


if __name__ == "__main__":
    main()
