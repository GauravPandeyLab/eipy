import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
import inspect
from eipy.utils import minority_class, retrieve_X_y
from sklearn.metrics import (
    f1_score,
    roc_auc_score,
)

def try_metric_with_pos_label(y_true, y_pred, metric, pos_label):
    '''
    Compute score for a given metric.
    '''
    try:
        score = metric(y_true, y_pred, pos_label=pos_label)
    except:
        score = metric(y_true, y_pred)
    return score

def metric_scaler_function(arg, y_true, y_pred, metric, pos_label):
        threshold = np.sort(np.unique(y_pred))[int(np.round(arg))]
        y_binary = (y_pred >= threshold).astype(int)
        return try_metric_with_pos_label(y_true, y_binary, metric, pos_label)

def max_min_score(y_true, y_pred, metric, pos_label, max_min):
    '''
    Compute maximized/minimized score for a given metric.
    '''

    if max_min=='max':
        multiplier = -1
    elif max_min=='min':
        multiplier = 1
    
    metric_func = lambda arg: multiplier * metric_scaler_function(arg, y_true, y_pred, metric, pos_label)
    optimized_result = minimize_scalar(metric_func, 
                                                 bounds=(0, len(np.unique(y_pred))-1), 
                                                 method='bounded')

    threshold = np.sort(np.unique(y_pred))[int(np.round(optimized_result.x))]
    score = multiplier * optimized_result.fun

    return score, threshold

# def max_min_score(y_true, y_pred, metric, pos_label, max_min):

#     '''
#     Compute maximized/minimized score for a given metric.
#     '''

#     thresholds = np.unique(y_pred)  # following sklearn approach (see sklearn.metrics.precision_recall_curve)
#     scores = []
#     for threshold in thresholds:
#         y_binary = (y_pred >= threshold).astype(int)
#         scores.append(try_metric_with_pos_label(y_true, y_binary, metric, pos_label))

#     if max_min=='max':
#         index = np.argmax(scores)  # find index where max occurs
#     elif max_min=='min':
#         index = np.argmin(scores)  # find index where min occurs

#     return scores[index], thresholds[index]

def scores(y_true, y_pred, metrics):
    '''
    Compute all metrics for a single set of predictions. Returns a dictionary
    containing metric keys, each paired to a tuple (score, threshold).
    '''

    pos_label = minority_class(y_true)  # gives value 1 or 0

    metric_threshold_dict = {}

    for metric_key, metric in metrics.items():

        string_suffix = metric_key[-3:]

        # maximize/minimize depending on whether metric_key has max/min suffix
        if string_suffix in ('max', 'min'):
            metric_threshold_dict[metric_key] = max_min_score(y_true, y_pred, metric, pos_label, string_suffix)
        else:
            # if y_pred parameter exists in metric function then y should be target prediction vector
            if 'y_pred' in inspect.signature(metric).parameters:
                y = (np.array(y_pred) >= .5).astype(int)  # threshold is 0.5
                metric_threshold_dict[metric_key] = try_metric_with_pos_label(y_true, y, metric, pos_label), 0.5
            # if y_score parameter exists in metric function then y should be probability vector
            elif 'y_score' in inspect.signature(metric).parameters:
                y = y_pred
                metric_threshold_dict[metric_key] = try_metric_with_pos_label(y_true, y, metric, pos_label), np.nan

    return metric_threshold_dict

def scores_matrix(X, labels, metrics):
    '''
    Calculate metrics and threshold (if applicable) for each column (set of predictions) in matrix X
    '''

    scores_dict = {}
    for column in X.columns:
        column_temp = X[column]
        metrics_per_column = scores(labels, column_temp, metrics)
        # metric_names = list(metrics.keys())
        for metric_key in metrics_per_column.keys():
            if not (metric_key in scores_dict):
                scores_dict[metric_key] = [metrics_per_column[metric_key]]
            else:
                scores_dict[metric_key].append(metrics_per_column[metric_key])

    return scores_dict

def create_metric_threshold_dataframes(X, labels, metrics):
    '''
    Create a separate dataframe for metrics and thresholds. thresholds_df contains
    NaN if threshold not applicable.
    '''

    scores_dict = scores_matrix(X, labels, metrics)

    metrics_df = pd.DataFrame(columns=X.columns)
    thresholds_df = pd.DataFrame(columns=X.columns)
    for k, val in scores_dict.items():
        metrics_df.loc[k], thresholds_df.loc[k] = list(zip(*val))
    return metrics_df, thresholds_df

def create_metric_threshold_dict(X, labels, metrics):
    df_dict = {}
    df_dict["metrics"], df_dict["thresholds"] = create_metric_threshold_dataframes(X, labels, metrics)
    return df_dict

def base_summary(meta_test_dataframes, metrics):
    '''
    Create a base predictor performance summary by concatenating data across test folds
    '''
    labels = pd.concat([df["labels"] for df in meta_test_dataframes])
    meta_test_averaged_samples = pd.concat(
        [
            df.drop(columns=["labels"], level=0).groupby(level=(0, 1), axis=1).mean()
            for df in meta_test_dataframes
        ]
    )
    return create_metric_threshold_dict(meta_test_averaged_samples, labels, metrics)

def meta_summary(meta_predictions, metrics):
    X = meta_predictions.drop(["labels"], axis=1)
    labels = meta_predictions["labels"]
    return create_metric_threshold_dict(X, labels, metrics)

