import pandas as pd
import numpy as np
import random
from sklearn.metrics import precision_recall_curve, matthews_corrcoef
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler


def fmax_score(y_true, y_pred, beta=1):
    # beta = 0 for precision, beta -> infinity for recall, beta=1 for harmonic mean
    np.seterr(divide='ignore', invalid='ignore')
    precision, recall, threshold = precision_recall_curve(y_true, y_pred)
    fmeasure = (1 + beta ** 2) * (precision * recall) / ((beta ** 2 * precision) + recall)
    argmax = np.nanargmax(fmeasure)

    f1score = fmeasure[argmax]
    pscore = precision[argmax]
    rscore = recall[argmax]

    return f1score, pscore, rscore
def matthews_max_score(y_true, y_pred):
    thresholds = np.arange(0, 1, 0.01)
    coeffs = []

    for threshold in thresholds:
        y_pred_round = np.copy(y_pred)
        y_pred_round[y_pred_round >= threshold] = 1
        y_pred_round[y_pred_round < threshold] = 0
        coeffs.append(matthews_corrcoef(y_true, y_pred_round))

    max_index = np.argmax(coeffs)
    max_threshold = thresholds[max_index]
    max_coeff = coeffs[max_index]

    return max_coeff, max_threshold

def scores(y_true, y_pred, beta=1, display=False):

    fmax = fmax_score(y_true, y_pred, beta=1)

    matthews_score, _ = matthews_max_score(y_true, y_pred)

    scores_dict = {"fmax score (positive class)": fmax,
                   "Matthew's correlation coefficient": matthews_score
                   }

    if display:
        for metric_name, score in scores_dict.items():
            print(metric_name + ": ", score)

    return scores_dict


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


def sample(X, y, random_state, strategy="undersampling"):
    if strategy == "undersampling":
        sampler = RandomUnderSampler(random_state=random_state)
    if strategy == "oversampling":
        sampler = RandomOverSampler(random_state=random_state)
    X_resampled, y_resampled = sampler.fit_resample(X=X, y=y)
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
