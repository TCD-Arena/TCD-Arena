import numpy as np
    
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