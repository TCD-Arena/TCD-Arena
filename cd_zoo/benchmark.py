import datetime
from os import listdir

import hydra
import numpy as np
from omegaconf import DictConfig

from tools.scoring_tools import score
from tools.tools import benchmarking, load_numpy_ds, save_full_out
from tools.method_loader import load_cd_method


# Example script to benchmark causal discovery methods.
@hydra.main(version_base=None, config_path="config", config_name="benchmark.yaml")
def main(cfg: DictConfig):
    
    # Load the specified method based on config
    print(cfg.method)
    cd_method = load_cd_method(cfg.method.name)


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
    # track runtime of only inference
    end_time = datetime.datetime.now()
    run_time = end_time - start
    

    # for violation datasets labels have an additional dimension that specifies graph changes during time series generation.
    if lagged_labels.ndim == 3: # no change case and different data source
        pass
    else:
        # If there are changes we mean over them to collapse the dimension.
        # Note, there is a fringe case where different coefficients cancel each other out exactly to 0. 
        # Practically this is extremely unlikely to happen.
        if lagged_labels.shape[1] == 1: # violation datasets with no changes
            lagged_labels =lagged_labels[:, 0, :, :, :]
        else:
            lagged_labels = lagged_labels.mean(axis=1)

        if instant_labels.shape[1] == 1:
            instant_labels =instant_labels[:, 0, :, :]
        else:   
            instant_labels = instant_labels.mean(axis=1)
       
    # Takes in predictions and labels and computes various metrics
    # Returns a pd.DataFrame with the results
    

    out = score(
        lagged_labels,
        lagged_preds,
        instant_labels
        if isinstance(instant_labels, np.ndarray) else None,  
        instant_preds if isinstance(instant_preds, np.ndarray) else None,
        remove_autoregressive_for_lagged=False,
        verbose=cfg.verbose,
        per_sample_metrics=cfg.per_sample_metrics
    )

    print(out)

    if cfg.save:
        save_full_out(out, end_time, lagged_preds,instant_preds, path, run_time, cfg)
    print("Done", datetime.datetime.now() - start)


if __name__ == "__main__":
    main()
