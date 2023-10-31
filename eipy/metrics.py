import numpy as np
import pandas as pd
import inspect
from eipy.utils import minority_class
from sklearn.metrics import roc_auc_score, precision_recall_curve


def fmax_score(y_test, y_score, beta=1.0, pos_label=1):
    fmax_score, _, _, threshold_fmax = fmax_precision_recall_threshold(
        y_test, y_score, beta=beta, pos_label=pos_label
    )
    return fmax_score, threshold_fmax


def fmax_precision_recall_threshold(labels, y_score, beta=1.0, pos_label=1):
    """
    Radivojac, P. et al. (2013). A Large-Scale Evaluation of Computational Protein
    Function Prediction. Nature Methods, 10(3), 221-227.
    Manning, C. D. et al. (2008). Evaluation in Information Retrieval. In
    Introduction to Information Retrieval. Cambridge University Press.
    """
    if pos_label == 0:
        labels = 1 - np.array(labels)
        y_score = 1 - np.array(y_score)

    precision_scores, recall_scores, thresholds = precision_recall_curve(
        labels, y_score
    )

    np.seterr(divide="ignore", invalid="ignore")
    f_scores = (
        (1 + beta**2)
        * (precision_scores * recall_scores)
        / ((beta**2 * precision_scores) + recall_scores)
    )

    arg_fmax = np.nanargmax(f_scores)

    fmax_score = f_scores[arg_fmax]
    precision_fmax = precision_scores[arg_fmax]
    recall_fmax = recall_scores[arg_fmax]
    threshold_fmax = thresholds[arg_fmax]

    return fmax_score, precision_fmax, recall_fmax, threshold_fmax


def try_metric_with_pos_label(y_true, y_pred, metric, pos_label):
    """
    Compute score for a given metric.
    """
    if "pos_label" in inspect.signature(metric).parameters:
        score = metric(y_true, y_pred, pos_label=pos_label)
    else:
        score = metric(y_true, y_pred)
    return score


def scores(y_true, y_pred, metrics):
    """
    Compute all metrics for a single set of predictions. Returns a dictionary
    containing metric keys, each paired to a tuple (score, threshold).
    """

    # default metrics to calculate
    if metrics is None:
        metrics = {"fmax (minority)": fmax_score, "auc": roc_auc_score}

    pos_label = minority_class(y_true)  # gives value 1 or 0

    metric_threshold_dict = {}

    for metric_key, metric in metrics.items():
        # if y_pred parameter exists in metric function then y
        # should be target prediction vector
        if "y_pred" in inspect.signature(metric).parameters:
            # calculate metric for target vector with threshold=0.5
            metric_threshold_dict[metric_key] = (
                try_metric_with_pos_label(
                    y_true, (np.array(y_pred) >= 0.5).astype(int), metric, pos_label
                ),
                0.5,
            )
        # if y_score parameter exists in metric function then y should be probability vector
        elif "y_score" in inspect.signature(metric).parameters:
            metric_results = try_metric_with_pos_label(
                y_true, y_pred, metric, pos_label
            )
            if isinstance(
                metric_results, tuple
            ):  # if metric includes threshold value as tuple
                metric_threshold_dict[metric_key] = metric_results
            else:  # add np.nan threshold if not outputted
                metric_threshold_dict[metric_key] = metric_results, np.nan

    return metric_threshold_dict


def scores_matrix(X, labels, metrics):
    """
    Calculate metrics and threshold (if applicable) for each column
    (set of predictions) in matrix X
    """

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
    """
    Create a separate dataframe for metrics and thresholds. thresholds_df contains
    NaN if threshold not applicable.
    """

    scores_dict = scores_matrix(X, labels, metrics)

    metrics_df = pd.DataFrame(columns=X.columns)
    thresholds_df = pd.DataFrame(columns=X.columns)
    for k, val in scores_dict.items():
        metrics_df.loc[k], thresholds_df.loc[k] = list(zip(*val))
    return metrics_df, thresholds_df


def create_metric_threshold_dict(X, labels, metrics):
    df_dict = {}
    df_dict["metrics"], df_dict["thresholds"] = create_metric_threshold_dataframes(
        X, labels, metrics
    )
    return df_dict


def base_summary(ensemble_test_dataframes, metrics):
    """
    Create a base predictor performance summary by concatenating data across test folds
    """
    labels = pd.concat([df["labels"] for df in ensemble_test_dataframes])
    ensemble_test_averaged_samples = pd.concat(
        [
            df.drop(columns=["labels"], level=0).groupby(level=(0, 1), axis=1).mean()
            for df in ensemble_test_dataframes
        ]
    )
    return create_metric_threshold_dict(ensemble_test_averaged_samples, labels, metrics)


def ensemble_summary(ensemble_predictions, metrics):
    X = ensemble_predictions.drop(["labels"], axis=1)
    labels = ensemble_predictions["labels"]
    return create_metric_threshold_dict(X, labels, metrics)


# These two functions are an attempt at maximizing/minimizing any metric
# def metric_scaler_function(arg, y_true, y_pred, metric, pos_label, multiplier):
#         threshold = np.sort(np.unique(y_pred))[int(np.round(arg))]
#         y_binary = (y_pred >= threshold).astype(int)
#         return multiplier * try_metric_with_pos_label(y_true, y_binary, metric, pos_label)

# def max_min_score(y_true, y_pred, metric, pos_label, max_min):
#     '''
#     Compute maximized/minimized score for a given metric.
#     '''

#     if max_min=='max':
#         multiplier = -1
#     elif max_min=='min':
#         multiplier = 1

#     optimized_result = minimize_scalar(
#                                         metric_scaler_function,
#                                         args=(y_true, y_pred, metric, pos_label, multiplier),
#                                         bounds=(0, len(np.unique(y_pred))-1),
#                                         method='bounded'
#                                         )

#     threshold = np.sort(np.unique(y_pred))[int(np.round(optimized_result.x))]
#     score = multiplier * optimized_result.fun

#     return score, threshold
#
