import pandas as pd
import numpy as np
from statsmodels.tsa.api import VAR



def run_var(d, cfg):
    """ 
    Simple Granger based strategy that selects based on absolute parameter values.
    """
    # np or pd.
    n_vars = d.values.shape[-1] if isinstance(d, pd.DataFrame) else  d.shape[0] 
    try:
        res = VAR(d).fit(cfg.max_lag) if isinstance(d, pd.DataFrame) else VAR(d.T).fit(cfg.max_lag)
        
        if cfg.base_on == "p_values":
            pred = 1 - res.pvalues[1:]
            
        elif cfg.base_on == "coefficients":
            pred = res.params[1:]
            if cfg.absolute_coefficients:
                pred = np.abs(pred)
        else:
            print("BASE UNKNOWN")
            return False
        # reformat to original caused causing lag:
        # :) einsum needed i guess
        if isinstance(d, pd.DataFrame):
            pred = np.stack(
                [pred.values[:, x].reshape(cfg.max_lag, n_vars).T for x in range(pred.shape[1])]
            )
        else:
            pred = np.stack(
                [pred[:, x].reshape(cfg.max_lag, n_vars).T for x in range(pred.shape[1])]
            )
    except Exception as e:
        print(e)
        pred = np.zeros((n_vars,n_vars,cfg.max_lag))
        
    if np.isnan(pred).sum() > 0:
        print("NAN predicted. replacing with  0")
        pred = np.zeros((n_vars,n_vars,cfg.max_lag))
    return pred, None # No instant preds.