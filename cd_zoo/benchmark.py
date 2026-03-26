import datetime
from os import listdir

import hydra
import numpy as np
from omegaconf import DictConfig,OmegaConf

import os
from pathlib import Path
import pickle
from tools.scoring_tools import score
from tools.tools import benchmarking, load_numpy_ds
from methods.var import run_var as cd_method


def save_full_out(out, end_time, lagged_preds, instant_preds, path, run_time, cfg):
    
    # Results will be saved in main_folder/violation_scmsize/method_name/job_id_timestamp/actual results.
    # gets violation_name and ds_name from path
    violation, ds = Path(path).parts[-2:]

    # creates violation folder
    main_folder = Path(cfg.save_path) / violation
    if not os.path.exists(main_folder):
        try:
            os.makedirs(main_folder)
        except OSError as e:
            print(e)
            pass  # multirun catch if multiple jobs try to create things simultaneously

    # creates method folder
    p = main_folder / cfg.method.name
    if not os.path.exists(p):
        try:
            os.makedirs(p)
        except OSError as e:
            print(e)
            pass  # multirun catch

    # Gets the run_name to label the subfolder for a specific ds.
    if cfg.run_name is not None:
        inner_p = os.path.join(
            p,
            str(cfg.run_name)
            + datetime.datetime.now().strftime("_%Y-%m-%d_%H-%M-%S-%f"),
        )
    else:
        # If we exexute without multirun there is no job_id which we use for naming
        inner_p = os.path.join(
            p, "0" + datetime.datetime.now().strftime("_%Y-%m-%d_%H-%M-%S-%f")
        )

    # create the folder
    os.makedirs(inner_p, exist_ok=True)
    inner_p = Path(inner_p)
    # add runtime and data_path of the original dataset to the output pd.DataFrame
    out.loc["runtime"] = run_time
    out.loc["path"] = path

    # Save everything into the folder.
    print("Saving results to: ", inner_p)
    out.to_csv(inner_p / "scoring.csv")
    with open(inner_p / "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)



# Example script to benchmark causal discovery methods.
@hydra.main(version_base=None, config_path="config", config_name="benchmark.yaml")
def main(cfg: DictConfig):

    # The path that is specified should contain multiple datasets to benchmark on (data regimes and violation levels)
    # We then select a single one of them to benchmark on. This allows us to run multiple experiments in parallel on different datasets.
    onlyfiles = sorted([cfg.data_path + "/" + f for f in listdir(cfg.data_path)])
    path = onlyfiles[cfg.which_dataset]
    test_data, lagged_labels, instant_labels = load_numpy_ds(path)


    # The datasets that were loaded can be restricted further if necessary
    if cfg.restrict_to_n_samples > 0:
        test_data = test_data[cfg.restriction_start_index : cfg.restriction_start_index + cfg.restrict_to_n_samples]
        lagged_labels = lagged_labels[cfg.restriction_start_index : cfg.restriction_start_index + cfg.restrict_to_n_samples]
        if isinstance(instant_labels, np.ndarray):
            instant_labels = instant_labels[
                cfg.restriction_start_index : cfg.restriction_start_index + cfg.restrict_to_n_samples
            ]

    # run the dataset through the method:
    start = datetime.datetime.now()
    # This returns a lagged (var x var x lag) and instantaneous prediction (var x var) if the method provides one else None
    lagged_preds, instant_preds = benchmarking(test_data, cfg, cd_method)

    # for violation datasets labels have an additional dimension that specifies graph changes during time series generation.
    # In the base example this is not the case and we can just collapse the dimension
    lagged_labels = lagged_labels[:, 0, :, :, :]
    instant_labels = instant_labels[:, 0, :, :]

    out = score(
        lagged_labels,
        lagged_preds,
        instant_labels if isinstance(instant_labels, np.ndarray) else None,
        instant_preds if isinstance(instant_preds, np.ndarray) else None,
        remove_autoregressive_for_lagged=False,
        verbose=cfg.verbose,
        per_sample_metrics=True,
    )
    end_time = datetime.datetime.now()
    run_time = end_time - start

    print(out)

    save_full_out(out, end_time, lagged_preds, instant_preds, path, run_time, cfg)
    print("Done", datetime.datetime.now() - start)


if __name__ == "__main__":
    main()
