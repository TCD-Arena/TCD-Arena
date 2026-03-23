import datetime
from os import listdir

import hydra
import numpy as np
from omegaconf import DictConfig

from tools.scoring_tools import score
from tools.tools import benchmarking, load_numpy_ds
from methods.var import run_var as cd_method


# Example script to benchmark causal discovery methods.
@hydra.main(version_base=None, config_path="config", config_name="benchmark.yaml")
def main(cfg: DictConfig):

    # The path that is specified should contain multiple datasets to benchmark on (data regimes and violation levels)
    # We then select a single one of them to benchmark on. This allows us to run multiple experiments in parallel on different datasets.
    onlyfiles = sorted([cfg.data_path + "/" + f for f in listdir(cfg.data_path)])
    path = onlyfiles[cfg.which_dataset]
    test_data, lagged_labels, instant_labels = load_numpy_ds(path)

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
        per_sample_metrics=False,
    )

    print(out)
    print("Done", datetime.datetime.now() - start)


if __name__ == "__main__":
    main()
