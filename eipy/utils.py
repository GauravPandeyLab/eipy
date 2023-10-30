import pandas as pd
import numpy as np
import random

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

# from tensorflow.keras.backend import clear_session
import warnings
from sklearn.pipeline import Pipeline
from sklearn.exceptions import UndefinedMetricWarning

warnings.filterwarnings(action="ignore", category=UndefinedMetricWarning)

bar_format = "{desc}: |{bar}|{percentage:3.0f}%"


def minority_class(y_true):
    if np.bincount(y_true)[0] < np.bincount(y_true)[1]:
        minority_class = 0
    else:
        minority_class = 1
    return minority_class


def set_predictor_seeds(base_predictors, random_state):
    for _, v in base_predictors.items():
        if type(v) == Pipeline:
            est_ = list(v.named_steps)[-1]
            if hasattr(v[est_], "random_state") and hasattr(v[est_], "set_params"):
                v.set_params(**{"{}__random_state".format(est_): random_state})
        if hasattr(v, "random_state") and hasattr(v, "set_params"):
            v.set_params(**{"random_state": random_state})


def X_is_dict(X):
    if isinstance(X, dict):
        return True
    else:
        return False


def X_dict_to_numpy(X_dict):
    """
    Retrieve feature names and convert arrays to numpy.
    """
    X_dict_numpy = {}
    feature_names = {}
    for key, X in X_dict.items():
        X_dict_numpy[key], feature_names[key] = X_to_numpy(X)
    return X_dict_numpy, feature_names


def X_to_numpy(X):
    """
    Return X as a numpy array, with feature names if applicable.
    """
    if isinstance(X, np.ndarray):
        return X, []
    elif isinstance(X, pd.DataFrame):
        return X.to_numpy(), X.columns.to_list()
    else:
        raise TypeError(
            """Object must be a numpy array, a pandas dataframe
            or a dictionary containing either."""
        )


def y_to_numpy(y):
    """
    Check y is numpy array and convert if not.
    """
    _y = None
    if isinstance(y, np.ndarray):
        _y = y
    elif isinstance(y, list):
        _y = np.array(y)
    elif isinstance(y, (pd.Series)):
        _y = y.to_numpy()
    elif isinstance(y, (pd.DataFrame)):
        _y = np.squeeze(y.to_numpy())
    else:
        raise TypeError(
            """Object must be a numpy array, list
            or pandas Series."""
        )

    if not is_binary_array(_y):
        raise ValueError("y must contain binary values.")

    return _y


def is_binary_array(arr):
    if all(x == 0 or x == 1 or x == 0.0 or x == 1.0 for x in arr):
        return True
    else:
        return False


class dummy_cv:
    def __init__(self, n_splits=1):
        self.n_splits = n_splits

    def split(self, X, y, groups=None):
        indices = np.arange(0, len(X), 1)
        yield indices, []

    def get_n_splits(self, X, y, groups=None):
        return self.n_splits


def safe_predict_proba(model, X):  # uses predict_proba method where possible
    if hasattr(model, "predict_proba"):
        y_pred = model.predict_proba(X)[:, 1]
    else:
        y_pred = model.predict(X)
    return y_pred


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
