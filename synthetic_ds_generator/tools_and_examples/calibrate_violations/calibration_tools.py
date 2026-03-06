import sys
sys.path.append("../..")

sys.path.append("../../..")

import numpy as np

from cd_zoo.methods.baseline_methods import (
    cross_corr_for_window_causal_graph as cross_corr,
)
from cd_zoo.tools.scoring_tools import score
from hydra import initialize, compose
from hydra.utils import instantiate



def eval_cross_corr_performance(
    X,
    Y,
):
    with initialize(version_base=None, config_path="../../../cd_zoo/config/method"):
        cfg = compose(config_name="crosscorr.yaml")
        cfg.max_lag = 3

    pred_stack = []
    for x in X:
        pred_stack.append(cross_corr(x, cfg)[0])

    a = np.stack(pred_stack, 0)
    b = np.array(Y)
    res = score(b, a, None, None, verbose=0)
    return res.loc["AUROC Joint"]["WCG"]




def check_violation_range(cfg, samples=10, verbose=0):
    generator = instantiate(cfg.generator)
    X = []
    Y = []
    fail_stack = []
    for counter in range(10):
        if cfg.link_mask_path:
            link_mask = np.load(cfg.link_mask_path + "/" + str(counter) + ".npy")
        else:
            link_mask = None
        if cfg.instant_link_mask_path:
            inst_link_mask = np.load(
                cfg.instant_link_mask_path + "/" + str(counter) + ".npy"
            )
        else:
            inst_link_mask = None
        # Generate a sample
        (
            sample_data,
            lagged_y,
            inst_y,
            exog_y,
            exog_ts,
            nl_struc,
            nl_inst,
            nl_exog,
            fails,
        ) = generator.get_sample(link_mask=link_mask, instant_link_mask=inst_link_mask)
        X.append(sample_data)
        Y.append(lagged_y[0])
    fail_stack.append(fails)
    if verbose> 0: 
            print(np.sum(np.abs(X)), np.var(X))
    
    return eval_cross_corr_performance(X, Y), fail_stack