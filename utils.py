import pandas as pd
import numpy as np
import random
import pickle
import os
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import StratifiedKFold
from joblib import Parallel, delayed
from imblearn.under_sampling import RandomUnderSampler
from sklearn.calibration import CalibratedClassifierCV
import warnings

def fmax_score(y_true, y_pred, beta=1, display=False):
    # beta = 0 for precision, beta -> infinity for recall, beta=1 for harmonic mean
    np.seterr(divide='ignore', invalid='ignore')
    precision, recall, threshold = precision_recall_curve(y_true, y_pred)
    fmeasure = (1 + beta ** 2) * (precision * recall) / ((beta ** 2 * precision) + recall)
    argmax = np.nanargmax(fmeasure)

    f1score = fmeasure[argmax]
    pscore = precision[argmax]
    rscore = recall[argmax]

    if display:
        print("f1 score: ", f1score,
              "\nprecision score:", pscore,
              "\nrecall score:", rscore)
    return f1score, pscore, rscore


def read_arff_to_pandas_df(arff_path):
    df = pd.read_csv(arff_path, comment='@', header=None)
    columns = []
    file = open(arff_path, 'r')
    lines = file.readlines()

    # Strips the newline character
    for line_idx, line in enumerate(lines):
        # if line_idx > num_col
        if '@attribute' in line.lower():
            columns.append(line.strip().split(' ')[1])

    df.columns = columns
    return df


def set_seed(random_state=1):
    random.seed(random_state)


def random_integers(n_integers=1):
    return random.sample(range(0, 10000), n_integers)


def undersample(X, y, random_state):
    RUS = RandomUnderSampler(random_state=random_state)
    X_resampled, y_resampled = RUS.fit_resample(X=X, y=y)
    return X_resampled, y_resampled


def retrieve_X_y(labelled_data):
    X = labelled_data.drop(columns=["labels"], level=0)
    y = np.ravel(labelled_data["labels"])
    return X, y


def update_keys(dictionary, string):
    return {f"{k}_" + string: v for k, v in dictionary.items()}


def append_modality(current_data, modality_data):
    combined_dataframe = []
    for fold, dataframe in enumerate(current_data):
        if (dataframe.iloc[:, -1].to_numpy() != modality_data[fold].iloc[:, -1].to_numpy()).all():
            print("Error: labels do not match across modalities")
            break
        combined_dataframe.append(pd.concat((dataframe.iloc[:, :-1],
                                             modality_data[fold]), axis=1))
    return combined_dataframe

