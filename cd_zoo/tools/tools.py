import numpy as np
import pandas as pd
import os
import pickle
from omegaconf import OmegaConf
from pathlib import Path
import datetime

def save_full_out(out,end_time,lagged_preds,instant_preds,path, run_time, cfg): 
    # Results will be saved in main_folder/violation_scmsize/method_name/job_id_timestamp/actual results.
    
    # gets violation_name and ds_name from path
    violation, ds = Path(path).parts[-2:]
    
    # creates violation folder
    main_folder =  Path(cfg.save_path)  / violation
    if not os.path.exists(main_folder):
            try:
                os.makedirs(main_folder)
            except OSError as e: 
                print(e)
                pass # multirun catch if multiple jobs try to create things simultaneously
            
    # creates method folder
    p = main_folder /  cfg.method.name
    if not os.path.exists(p):
            try:
                os.makedirs(p)
            except OSError as e: 
                print(e)
                pass # multirun catch
    
    # Gets the run_name to label the subfolder for a specific ds.
    if cfg.run_name is not None:
        inner_p = os.path.join(p, str(cfg.run_name)+ datetime.datetime.now().strftime("_%Y-%m-%d_%H-%M-%S-%f")) 
    else:
        # If we exexute without multirun there is no job_id which we use for naming
        inner_p = os.path.join(p, "0" + datetime.datetime.now().strftime("_%Y-%m-%d_%H-%M-%S-%f"))

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
    if cfg.save_predictions:
        pickle.dump((lagged_preds, instant_preds), open(inner_p / "preds.p", "wb"))
        
        
def make_human_readable(out, d):
    """
    Transforms matrix to pd.DataFrame format specified by d
    """
    out = pd.DataFrame(out, columns=d.columns, index=d.columns)
    out = pd.concat([pd.concat([out], keys=["Cause"], axis=1)], keys=["Effect"])
    return out


def benchmarking(X, cfg, method_to_test):
    """
    Takes in the output of the data loader and perform the predictions with a specified method.
    If anything else should happen with the data beforehand we should perform this here.
    """
    ll = []
    il = []
    for x, sample in enumerate(X):
        if cfg.verbose > 0:
            print(x, "/", len(X))
        lagged, instant = method_to_test(sample, cfg.method) 
        ll.append(lagged)
        il.append(instant)
    return np.array(ll), np.array(il) if isinstance(instant, np.ndarray) else None

def load_numpy_ds(data_path):
    """
    Loads the data from the specified path.
    """

    X = np.load(data_path + "/X.npy")
    Y = np.load(data_path + "/Y.npy")
    try:
        Z = np.load(data_path + "/instant_links.npy")
    except FileNotFoundError:
        Z = None
    return X, Y, Z