import pandas as pd
import numpy as np
import random
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_curve,
    matthews_corrcoef,
    precision_recall_fscore_support,
    make_scorer,
)
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

# from tensorflow.keras.backend import clear_session
import warnings
import sklearn
import sklearn.metrics
from sklearn.exceptions import UndefinedMetricWarning

warnings.filterwarnings(action="ignore", category=UndefinedMetricWarning)

bar_format = "{desc}: |{bar}|{percentage:3.0f}%"

class TFWrapper:
    def __init__(self, tf_fun, compile_kwargs, fit_kwargs):
        self.tf_fun = tf_fun
        # self.initial_weights = self.tf_fun().get_weights()
        self.compile_kwargs = compile_kwargs
        self.fit_kwargs = fit_kwargs

        # self.tf_model.compile(**self.compile_kwargs)

    def fit(self, X, y):
        # clear_session()
        self.model = self.tf_fun()
        self.model.compile(**self.compile_kwargs)
        # self.model.set_weights(self.initial_weights)  # re-initialises weights for multiple .fit calls
        self.model.fit(X, y, verbose=0, **self.fit_kwargs)

    def predict_proba(self, X):
        return np.squeeze(self.model.predict(X))

        separator = "#" * 40
        if modality is None:
            text = separator * 2
        else:
            text = f"{separator} {modality} modality {separator}"


def fancy_print(text=None, length=None):
    print("\n")
    print("#" * len(text))
    print(f"{separator} {text} {separator}")
    print("#" * len(text), "\n")


class dummy_cv:
    def __init__(self, n_splits=1):
        self.n_splits = n_splits

    def split(self, X, y, groups=None):
        indices = np.arange(0, len(X), 1)
        yield indices, []

    def get_n_splits(self, X, y, groups=None):
        return self.n_splits


def create_base_summary(meta_test_dataframe):
    labels = pd.concat([df["labels"] for df in meta_test_dataframe])
    meta_test_averaged_samples = pd.concat(
        [
            df.drop(columns=["labels"], level=0).groupby(level=(0, 1), axis=1).mean()
            for df in meta_test_dataframe
        ]
    )
    meta_test_averaged_samples["labels"] = labels
    return metric_threshold_dataframes(meta_test_averaged_samples)


def safe_predict_proba(model, X):  # uses predict_proba method where possible
    if hasattr(model, "predict_proba"):
        y_pred = model.predict_proba(X)[:, 1]
    else:
        y_pred = model.predict(X)
    return y_pred


def score_threshold_vectors(df, labels):
    scores_dict = {}
    for column in df.columns:
        column_temp = df[column]
        metrics = scores(labels, column_temp)
        metric_names = list(metrics.keys())
        for m in metric_names:
            if not (m in scores_dict):
                scores_dict[m] = [metrics[m]]
            else:
                scores_dict[m].append(metrics[m])
    return scores_dict


# def score_vector_split(list_of_tuples): # don't think this is needed anymore
#     score = []
#     threshold = []
#     for score_tuple in list_of_tuples:
#         score.append(score_tuple[0])
#         threshold.append(score_tuple[1])
#     return score, threshold


def metrics_per_fold(df, labels):
    scores_dict = score_threshold_vectors(df, labels)

    metrics_df = pd.DataFrame(columns=df.columns)
    thresholds_df = pd.DataFrame(columns=df.columns)
    for k, val in scores_dict.items():
        metrics_df.loc[k], thresholds_df.loc[k] = list(zip(*val))
    return metrics_df, thresholds_df


def metric_threshold_dataframes(df):
    data = df.drop(["labels"], axis=1)
    labels = df["labels"]
    df_dict = {}
    df_dict["metrics"], df_dict["thresholds"] = metrics_per_fold(data, labels)
    return df_dict


def format_input_datatype(X, modality_name):
    if type(X) == pd.core.frame.DataFrame:
        """if the data input is dataframe, store the feature name"""
        feature_names = list(X.columns)
        X_np = X.values
        # print(modal_name, modality.shape)
    elif type(X) == np.ndarray:
        """If there is no feature name in input/feature name dictionary"""
        feature_names = [f'{modality_name}_{i}' for i in range(X.shape[1])]
        X_np = X
    else:
        print('Input X can only be either numpy array or pandas dataframe object.')
        return None, None

    return X_np, feature_names


def fmax_score(y_true, y_pred, beta=1):
    # beta = 0 for precision, beta -> infinity for recall, beta=1 for harmonic mean
    np.seterr(divide="ignore", invalid="ignore")
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    fmeasure = (
        (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)
    )
    argmax = np.nanargmax(fmeasure)

    fmax = fmeasure[argmax]
    max_fmax_threshold = thresholds[argmax]

    return fmax, max_fmax_threshold


def fmeasure_score(
    labels, predictions, thres=None, beta=1.0, pos_label=1, thres_same_cls=False
):
    """
    Radivojac, P. et al. (2013). A Large-Scale Evaluation of Computational Protein Function Prediction. Nature Methods, 10(3), 221-227.
    Manning, C. D. et al. (2008). Evaluation in Information Retrieval. In Introduction to Information Retrieval. Cambridge University Press.
    """
    np.seterr(divide="ignore", invalid="ignore")
    if pos_label == 0:
        labels = 1 - np.array(labels)
        predictions = 1 - np.array(predictions)
        # if not(thres is None):
        #     thres = 1-thres
    # else:

    if thres is None:  # calculate fmax here
        np.seterr(divide="ignore", invalid="ignore")
        precision, recall, threshold = sklearn.metrics.precision_recall_curve(
            labels,
            predictions,
            #   pos_label=pos_label
        )

        fs = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)
        fmax_point = np.where(fs == np.nanmax(fs))[0]
        p_maxes = precision[fmax_point]
        r_maxes = recall[fmax_point]
        pr_diff = np.abs(p_maxes - r_maxes)
        balance_fmax_point = np.where(pr_diff == min(pr_diff))[0]
        p_max = p_maxes[balance_fmax_point[0]]
        r_max = r_maxes[balance_fmax_point[0]]
        opt_threshold = threshold[fmax_point][balance_fmax_point[0]]

        return {
            "F": np.nanmax(fs),
            "thres": opt_threshold,
            "P": p_max,
            "R": r_max,
            "PR-curve": [precision, recall],
        }

    else:  # calculate fmeasure for specific threshold
        binary_predictions = np.array(predictions)
        if thres_same_cls:
            binary_predictions[binary_predictions >= thres] = 1.0
            binary_predictions[binary_predictions < thres] = 0.0
        else:
            binary_predictions[binary_predictions > thres] = 1.0
            binary_predictions[binary_predictions <= thres] = 0.0
        precision, recall, fmeasure, _ = precision_recall_fscore_support(
            labels,
            binary_predictions,
            average="binary",
            zero_division=0.0,
            # pos_label=pos_label
        )
        return {"P": precision, "R": recall, "F": fmeasure}


def matthews_max_score(y_true, y_pred):
    thresholds = np.arange(0, 1, 0.01)
    coeffs = []

    for threshold in thresholds:
        y_pred_round = np.copy(y_pred)
        y_pred_round[y_pred_round >= threshold] = 1
        y_pred_round[y_pred_round < threshold] = 0
        coeffs.append(matthews_corrcoef(y_true, y_pred_round))

    max_index = np.argmax(coeffs)
    max_mcc_threshold = thresholds[max_index]
    max_mcc = coeffs[max_index]

    return max_mcc, max_mcc_threshold


def scores(y_true, y_pred, beta=1, metric_to_maximise="fscore", verbose=0):
    if np.bincount(y_true)[0] < np.bincount(y_true)[1]:
        minor_class = 0
        major_class = 1
    else:
        minor_class = 1
        major_class = 0

    # fmax = fmax_score(y_true, y_pred, beta=1)
    f_measure_minor = fmeasure_score(y_true, y_pred, pos_label=minor_class)
    # f_measure_minor_0 = fmeasure_score(y_true, y_pred, pos_label=minor_class,
    # thres=f_measure_minor['thres'])
    # print('without thres', f_measure_minor)
    # print('with thres', f_measure_minor_0)
    f_measure_major = fmeasure_score(
        y_true, y_pred, pos_label=major_class, thres=1 - f_measure_minor["thres"]
    )

    max_mcc = matthews_max_score(y_true, y_pred)

    auc = roc_auc_score(y_true, y_pred)

    scores_threshold_dict = {
        "fmax (minority)": (f_measure_minor["F"], f_measure_minor["thres"]),
        "f (majority)": (f_measure_major["F"], f_measure_minor["thres"]),
        "AUC": (auc, np.nan),
        "max MCC": max_mcc,
    }  # dictionary of (score, threshold)

    if verbose > 0:
        for metric_name, score in scores_threshold_dict.items():
            print(metric_name + ": ", score[0])

    return scores_threshold_dict


def read_arff_to_pandas_df(arff_path):
    df = pd.read_csv(arff_path, comment="@", header=None)
    columns = []
    file = open(arff_path, "r")
    lines = file.readlines()

    # Strips the newline character
    for line_idx, line in enumerate(lines):
        # if line_idx > num_col
        if "@attribute" in line.lower():
            columns.append(line.strip().split(" ")[1])

    df.columns = columns
    return df


def set_seed(random_state=1):
    random.seed(random_state)


def random_integers(n_integers=1, seed=42):
    random.seed(seed)
    return random.sample(range(0, 10000), n_integers)


def sample(X, y, strategy, random_state):
    if strategy is None:
        X_resampled, y_resampled = X, y
    elif strategy == "undersampling":  # define sampler
        sampler = RandomUnderSampler(random_state=random_state)
    elif strategy == "oversampling":
        sampler = RandomOverSampler(random_state=random_state)
    elif strategy == "hybrid":
        y_pos = float(sum(y == 1))
        y_total = y.shape[0]
        if (y_pos / y_total) < 0.5:
            y_min_count = y_pos
            y_maj_count = y_total - y_pos
            maj_class = 0
        else:
            y_maj_count = y_pos
            y_min_count = y_total - y_pos
            maj_class = 1
        rus = RandomUnderSampler(
            random_state=random_state, sampling_strategy=y_min_count / (y_total / 2)
        )
        ros = RandomOverSampler(
            random_state=random_state, sampling_strategy=(y_total / 2) / y_maj_count
        )
        X_maj, y_maj = rus.fit_resample(X=X, y=y)
        X_maj = X_maj[y_maj == maj_class]
        y_maj = y_maj[y_maj == maj_class]
        X_min, y_min = ros.fit_resample(X=X, y=y)
        X_min = X_min[y_min != maj_class]
        y_min = y_min[y_min != maj_class]
        X_resampled = np.concatenate([X_maj, X_min])
        y_resampled = np.concatenate([y_maj, y_min])

    if (strategy == "undersampling") or (strategy == "oversampling"):
        X_resampled, y_resampled = sampler.fit_resample(X=X, y=y)
    return X_resampled, y_resampled


def retrieve_X_y(labelled_data):
    X = labelled_data.drop(columns=["labels"], level=0)
    y = np.ravel(labelled_data["labels"])
    return X, y


def append_modality(current_data, modality_data, model_building=False):
    if current_data is None:
        combined_dataframe = modality_data
    else:
        combined_dataframe = []
        for fold, dataframe in enumerate(current_data):
            if not model_building:
                if (
                    dataframe.iloc[:, -1].to_numpy()
                    != modality_data[fold].iloc[:, -1].to_numpy()
                ).all():
                    print(
                        "Error: something is wrong. Labels do not match across modalities"
                    )
                    break
                combined_dataframe.append(
                    pd.concat((dataframe.iloc[:, :-1], modality_data[fold]), axis=1)
                )
            else:
                combined_dataframe.append(
                    pd.concat((dataframe.iloc[:, :], modality_data[fold]), axis=1)
                )
    return combined_dataframe


def auprc(y_true, y_scores):
    return sklearn.metrics.average_precision_score(y_true, y_scores)


auprc_sklearn = make_scorer(auprc, greater_is_better=True, needs_proba=True)


def f_minority_score(y_true, y_pred):
    # if (len(y_pred.shape) > 2):
    #     y_pred_pos = y_pred[:, -1]
    # else:
    #     y_pred_pos = y_pred
    if np.bincount(y_true)[0] < np.bincount(y_true)[1]:
        minor_class = 0
    else:
        minor_class = 1
    # return fmeasure_score(y_true, y_pred_pos, pos_label=minor_class)['F']
    return fmeasure_score(y_true, y_pred, pos_label=minor_class)["F"]


def generate_scorer_by_model(score_func, model, greater_is_better):
    needs_proba = False
    if hasattr(model, "predict_proba"):
        needs_proba = True
    print(model, needs_proba)
    new_scorer = make_scorer(
        score_func=score_func, greater_is_better=greater_is_better, needs_proba=True
    )
    return new_scorer


f_minor_sklearn = make_scorer(
    f_minority_score, greater_is_better=True, needs_proba=True
)
f_minor_sklearn_bin_only = make_scorer(
    f_minority_score, greater_is_better=True, needs_proba=False
)


