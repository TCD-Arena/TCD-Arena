import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_recall_curve,
    roc_auc_score,
    accuracy_score,
)

np.seterr(divide='ignore', invalid='ignore')


def remove_diagonal(T):
    """
    Removes all autoregressive predictions from the Tensor.
    ONLY implemented for 3D Tensors (var, var (, lag))
    
    """
    out = []
    if T.ndim == 3: 
        for x in T:
            # Remove diagonal elements and reshape
            masked = x[~np.eye(x.shape[0], dtype=bool)]
            reshaped = masked.reshape(x.shape[0], -1)
            out.append(reshaped)
    else:
        raise ValueError("Only implemented for 3D tensors.")
    return np.stack(out)


def max_accuracy(labs, preds):
    # ACCURACY MAX
    preds = preds.astype(float)
    if preds.min() == preds.max():
        a = []
    else:
        a = list(np.linspace(preds.min(), preds.max(), num=100))
        # 100 steps
    possible_thresholds = [0] + a + [preds.max() + 1e-6]
    acc = [accuracy_score(labs, preds > thresh) for thresh in possible_thresholds]
    acc_thresh = possible_thresholds[np.argmax(acc)]
    acc_score = np.nanmax(acc)
    return acc_thresh, acc_score


def f1_max(labs, preds):
    # F1 MAX
    # Note this is generating warnings if target is empty. (e.g. no positive instantanous links in the labels).
    # F1 is 0 in this case and influences the individual F1 scores (pushes them lower)
    precision, recall, thresholds = precision_recall_curve(labs, preds)
    f1_scores = 2 * recall * precision / (recall + precision)
    f1_thresh = thresholds[np.argmax(f1_scores)]
    f1_score = np.nanmax(f1_scores)
    return f1_thresh, f1_score


def min_shd(labs, preds):
    # SHD MIN
    preds = preds.astype(float)
    if preds.min() == preds.max():
        a = []
    else:
        a = list(np.linspace(preds.min(), preds.max(), num=100))
        # 100 steps
        #a = sorted(list(set(a)))  # unique and sorted
    possible_thresholds = [0] + a + [preds.max() + 1e-6]
    shd = [
        np.sum((preds > thresh) != labs) for thresh in possible_thresholds
    ]  # sum of false pos and false neg
    shd_thresh = possible_thresholds[np.argmin(shd)]
    shd_score = np.nanmin(shd)
    # normalize with total number of positive links to make it comparable across datasets with different sparsity and size.
    shd_score = shd_score / sum(labs)
    return shd_thresh, shd_score


def get_tpr_fpr_tnr_fnr_prec_rec_for_threshold(labs, preds,th):
    # TPR / FPR rates
    preds = preds.astype(float)
    tp = np.sum((preds > th) & (labs == 1))
    fn = np.sum((preds <= th) & (labs == 1))
    fp = np.sum((preds > th) & (labs == 0))
    tn = np.sum((preds <= th) & (labs == 0))
    tpr = tp / (tp + fn) if ((tp + fn) > 0 ) else 0 # catch division by zero.
    fpr = fp / (fp + tn) if ((fp + tn) > 0 ) else 0
    tnr = tn / (tn + fp) if ((tn + fp) > 0 ) else 0
    fnr = fn / (fn + tp) if ((fn + tp) > 0 ) else 0
    prec = tp / (tp + fp) if ((tp + fp) > 0 )else 0
    rec = tp / (tp + fn) if ((tp + fn) > 0 ) else 0    
    return tpr, fpr, tnr, fnr, prec, rec


def calc_metrics_for_each_sample(labs, preds, verbose=0):
    stack = []
    for x in range(len(labs)):
        if verbose:
            print(x, "/", len(labs))
        stack.append(calc_metrics(labs[x], preds[x]))

    stack = pd.concat(stack, axis=1).mean(axis=1)
    stack.index = stack.index + " individual"
    return stack


def calc_metrics(labs, preds):
    # Joint calculation
    labs = labs.flatten()
    preds = preds.flatten()
    # AUROC
    if labs.max() == labs.min(): # One one class present.
        auroc = np.nan
    else:
        auroc = roc_auc_score(labs, preds)
    # F1 MAX
    f1_thresh, f1_score = f1_max(labs, preds)
    # ACCURACY MAX
    acc_thresh, acc_score = max_accuracy(labs, preds)
    # SHD MIN
    shd_thresh, shd_score = min_shd(labs, preds)

    f1_tpr, f1_fpr, f1_tnr, f1_fnr, f1_prec, f1_rec = get_tpr_fpr_tnr_fnr_prec_rec_for_threshold(
        labs, preds, f1_thresh
    )
    acc_tpr, acc_fpr, acc_tnr, acc_fnr, acc_prec, acc_rec = get_tpr_fpr_tnr_fnr_prec_rec_for_threshold(
        labs, preds, acc_thresh
    )
    shd_tpr, shd_fpr, shd_tnr, shd_fnr, shd_prec, shd_rec = get_tpr_fpr_tnr_fnr_prec_rec_for_threshold(
        labs, preds, shd_thresh
    )


    # Null Models with no prediction.
    if labs.max() == labs.min(): # One one class present.
        null_model_auroc = np.nan
    else:
        null_model_auroc = roc_auc_score(labs, np.zeros(preds.shape))
    _, null_model_f1 = f1_max(labs, np.zeros(preds.shape))
    _, null_model_acc = max_accuracy(labs, np.zeros(preds.shape))
    _, null_model_shd = min_shd(labs, np.zeros(preds.shape))
    return pd.DataFrame(
        [
            auroc,
            f1_score,
            acc_score,
            shd_score,
            f1_thresh,
            acc_thresh,
            shd_thresh,
            null_model_auroc,
            null_model_f1,
            null_model_acc,
            null_model_shd,
            f1_fpr,
            f1_tpr,
            f1_tnr,
            f1_fnr,
            f1_prec,
            f1_rec,
            acc_fpr,
            acc_tpr,
            acc_tnr,
            acc_fnr,
            acc_prec,
            acc_rec,
            shd_fpr,
            shd_tpr,
            shd_tnr,
            shd_fnr,
            shd_prec,
            shd_rec,
        ],
        index=[
            "AUROC",
            "Max F1",
            "Acc",
            "SHD",
            "Max F1 th",
            "Acc th",
            "SHD th",
            "Null AUROC",
            "NULL F1",
            "NULL ACC",
            "NULL SHD",
            "F1_FPR",
            "F1_TPR",
            "F1_TNR",
            "F1_FNR",
            "F1_PREC",
            "F1_REC",
            "ACC_FPR",
            "ACC_TPR",
            "ACC_TNR",
            "ACC_FNR",
            "ACC_PREC",
            "ACC_REC",
            "SHD_FPR",
            "SHD_TPR",
            "SHD_TNR",
            "SHD_FNR",
            "SHD_PREC",
            "SHD_REC",
        ],
    )


def calc_stat_table(labs, preds, name, per_sample_metrics=True, verbose=1):
    # Calculates a number of metrics and returns a df holding them
    # Joint Calculation
    metrics_joint = calc_metrics(labs, preds)
    metrics_joint.index = metrics_joint.index + " Joint"

    if per_sample_metrics:
        # Individual scoring for each sample.
        metrics_individual = calc_metrics_for_each_sample(labs, preds, verbose=verbose)
        metrics = pd.concat([metrics_joint, metrics_individual], axis=0)
    else:
        metrics = metrics_joint
    metrics.index.name = "Metric"
    metrics.columns = [name]
    return metrics


def handle_padding(labs, preds):
    # IF the max lax is missspecified we need to pad 0 to either the 
    # predictions (if the labels are higher) or the labels (if the predictions are higher)
    if preds.shape[-1] > labs.shape[-1]:
        l = np.pad(
            labs,
            ((0, 0), (0, 0), (0, 0), (0, preds.shape[-1] - labs.shape[-1])),
            constant_values=False,
        )
        p = preds

    elif preds.shape[-1] < labs.shape[-1]:
        p = np.pad(
            preds,
            ((0, 0), (0, 0), (0, 0), (0, labs.shape[-1] - preds.shape[-1])),
            constant_values=False,
        )
        l = labs
    else:
        p, l = preds, labs
    return l, p


def score(
    labs,
    preds,
    instant_labs=None,
    instant_preds=None,
    remove_autoregressive_for_lagged=False,
    verbose=1,
    per_sample_metrics=False,
):
    """
    Calculates a number of metrics given preds and labs.
    
    Score predicted causal adjacency matrices against ground-truth labels and return
    a concatenated table of evaluation metrics.
    This function supports both instantaneous (no lag) and lagged causal graphs. It
    accepts ground-truth labels and model predictions in several common shapes and
    computes a set of summary and windowed metrics by delegating to helper
    functions.
    
    Parameters
    ----------
    labs : np.ndarray or list
            Ground-truth adjacency matrices. Expected shapes:
            - Summary: (samples, n_vars, n_vars)
            - Lagged: (samples, n_vars, n_vars, lags)
            Non-zero entries denote the presence of a link.
    preds : np.ndarray or list
            Predicted adjacency matrices in either summary or lagged format.
    instant_labs : np.ndarray, optional
            Ground-truth instantaneous adjacency matrices to be scored separately.
            If provided, must be a numpy array.
    instant_preds : np.ndarray, optional
            Predicted instantaneous adjacency matrices to be scored separately.
    remove_autoregressive_for_lagged : bool, optional (default=False)
            If True, remove diagonal/autoregressive edges before scoring when lagged
            inputs are present. This is only implemented for 3 dimensional inputs.
    verbose : int, optional (default=1)
            Verbosity flag forwarded to calc_stat_table for display/logging.
    per_sample_metrics : bool, optional (default=False)
            If True, calc_stat_table will return metrics computed per sample (and mean over them)
            additionally to aggregated metrics.
    Returns
    -------
    pd.DataFrame
            A DataFrame formed by concatenating one or more metric tables (columns).
            Typical column groups and their meanings:
            - "SG_max": summary graph using max over lag dimension for predictions and
                labels (always present).
            - "WCG": windowed/lag-aware comparison (present for lagged inputs).
            - "SG_mean": summary graph using mean of predictions across lags and max of
                labels (present for lagged inputs).
            - "INST": instantaneous scoring table (present when instant predictions and labels are given).

    """

    
    # match the dimensions of preds and labs if necessary
    if preds.ndim == 3 and labs.ndim == 4:
        # preds are summary graphs, labs are lagged graphs
        # convert labs to summary graphs by taking max over lag dimension
        labs = labs.sum(axis=3)
    elif preds.ndim == 4 and labs.ndim == 3:
        preds = preds.sum(axis=3)
        
    # to bool for binary classification
    labs = labs != 0
    instant_labs = instant_labs != 0

    # If you want to remove diagonal (autoregressive elements) before calc scoring:
    # This can be useful e.g. if links are garantueed to be autocorrelated.
    if remove_autoregressive_for_lagged:
        # ONLY IMPLEMENTED FOR SUMMARY GRAPHS. #TODO implement for WCG as well.
        preds = remove_diagonal(preds)
        labs = remove_diagonal(labs)
        


    result_tables = []
    # summary graph always possible.
    result_tables.append(
        calc_stat_table(
            labs.max(axis=3).astype(bool) if preds.ndim == 4 else labs.astype(bool),
            preds.max(axis=3) if preds.ndim == 4 else preds,
            name="SG_max",
            verbose=verbose,
            per_sample_metrics=per_sample_metrics,
        )
    )

    # If we have a lag dim, Calc 2 additonal tables: WCG, and SG_mean:
    if preds.ndim == 4:
        l, p = handle_padding(labs, preds)
        result_tables.append(
            calc_stat_table(
                l.astype(bool), p, name="WCG", per_sample_metrics=per_sample_metrics, verbose=verbose
            )
        )

        result_tables.append(
            calc_stat_table(
                labs.max(axis=3).astype(
                    bool),  # We still want to simply decide whether a any link exist
                preds.mean(axis=3),
                name="SG_mean",
                verbose=verbose,
                per_sample_metrics=per_sample_metrics,
            )
        )

    if isinstance(instant_preds, np.ndarray) and isinstance(instant_labs, np.ndarray):
        instant_preds = remove_diagonal(instant_preds) #Always remove as self-cause is not possible.
        instant_labs = remove_diagonal(instant_labs)

        result_tables.append(
            calc_stat_table(
                instant_labs.astype(
                    bool
                ),  # We still want to simply decide whether a link exist
                instant_preds,
                name="INST",
                verbose=verbose,
                per_sample_metrics=per_sample_metrics,
            )
        )

    final = pd.concat(result_tables, axis=1)

    return final
