import pandas as pd
import numpy as np
import random
from sklearn.metrics import roc_auc_score, precision_recall_curve, matthews_corrcoef
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from tensorflow.keras.models import clone_model

class TFWrapper:
    def __init__(self, tf_model, compile_kwargs, fit_kwargs):
        self.tf_model = tf_model
        self.initial_weights = self.tf_model.get_weights()
        self.compile_kwargs = compile_kwargs
        self.fit_kwargs = fit_kwargs

        self.tf_model.compile(**self.compile_kwargs)

    def fit(self, X, y):
        self.tf_model_new = clone_model(self.tf_model)
        self.tf_model_new.set_weights(self.initial_weights)  # re-initialises weights for multiple .fit calls
        self.tf_model_new.fit(X, y, verbose=0, **self.fit_kwargs)

    def predict_proba(self, X):
        y_pred = np.squeeze(self.tf_model_new.predict(X))
        return y_pred


def score_threshold_vectors(df, labels):
    fmax = []
    auc = []
    mcc = []
    for column in df.columns:
        column_temp = df[column]
        metrics = scores(labels, column_temp)
        metric_names = list(metrics.keys())
        fmax.append(metrics[metric_names[0]])
        auc.append(metrics[metric_names[1]])
        mcc.append(metrics[metric_names[2]])
    return fmax, auc, mcc


# def score_vector_split(list_of_tuples): # don't think this is needed anymore
#     score = []
#     threshold = []
#     for score_tuple in list_of_tuples:
#         score.append(score_tuple[0])
#         threshold.append(score_tuple[1])
#     return score, threshold


def metrics_per_fold(df, labels):
    fmax, auc, mcc = score_threshold_vectors(df, labels)

    metrics_df = pd.DataFrame(columns=df.columns)
    thresholds_df = pd.DataFrame(columns=df.columns)

    metrics_df.loc["fmax score"], thresholds_df.loc["fmax score"] = list(zip(*fmax))
    metrics_df.loc["AUC score"], thresholds_df.loc["AUC score"] = list(zip(*auc))
    metrics_df.loc["MCC score"], thresholds_df.loc["MCC score"] = list(zip(*mcc))

    return metrics_df, thresholds_df


def metric_threshold_dataframes(df):
    data = df.drop(["labels"], axis=1)
    labels = df["labels"]
    df_dict = {}
    df_dict["metrics"], df_dict["thresholds"] = metrics_per_fold(data, labels)
    return df_dict


def fmax_score(y_true, y_pred, beta=1):
    # beta = 0 for precision, beta -> infinity for recall, beta=1 for harmonic mean
    np.seterr(divide='ignore', invalid='ignore')
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    fmeasure = (1 + beta ** 2) * (precision * recall) / ((beta ** 2 * precision) + recall)
    argmax = np.nanargmax(fmeasure)

    fmax = fmeasure[argmax]
    max_fmax_threshold = thresholds[argmax]

    return fmax, max_fmax_threshold


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


def scores(y_true, y_pred, beta=1, metric_to_maximise="fscore", display=False):
    fmax = fmax_score(y_true, y_pred, beta=1)

    max_mmc = matthews_max_score(y_true, y_pred)

    auc = roc_auc_score(y_true, y_pred)

    scores_threshold_dict = {"fmax score (positive class)": fmax,
                             "AUC score": (auc, np.nan),
                             "Matthew's correlation coefficient": max_mmc
                             }  # dictionary of (score, threshold)

    if display:
        for metric_name, score in scores_threshold_dict.items():
            print(metric_name + ": ", score[0])

    return scores_threshold_dict


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
    if strategy == 'hybrid':
        y_pos = float(sum(y==1))
        y_total = y.shape[0]
        if (y_pos/y_total) < 0.5:
            y_min_count = y_pos
            y_maj_count = (y_total - y_pos)
            maj_class = 0
        else:
            y_maj_count = y_pos
            y_min_count = (y_total - y_pos)
            maj_class = 1
        rus = RandomUnderSampler(random_state=random_state, 
                                sampling_strategy=y_min_count/(y_total/2))
        ros = RandomOverSampler(random_state=random_state,   
                                sampling_strategy=(y_total/2)/y_maj_count)
        X_maj, y_maj = rus.fit_resample(X=X, y=y)
        X_maj = X_maj[y_maj==maj_class]
        y_maj = y_maj[y_maj==maj_class]
        X_min, y_min = ros.fit_resample(X=X, y=y)
        X_min = X_min[y_min!=maj_class]
        y_min = y_min[y_min!=maj_class]
        X_resampled = np.concatenate([X_maj, X_min])
        y_resampled = np.concatenate([y_maj, y_min])
    else:
        X_resampled, y_resampled = sampler.fit_resample(X=X, y=y)
    return X_resampled, y_resampled


def retrieve_X_y(labelled_data):
    X = labelled_data.drop(columns=["labels"], level=0)
    y = np.ravel(labelled_data["labels"])
    return X, y


def append_modality(current_data, modality_data):
    if current_data is None:
        combined_dataframe = modality_data
    else:
        combined_dataframe = []
        for fold, dataframe in enumerate(current_data):
            if (dataframe.iloc[:, -1].to_numpy() != modality_data[fold].iloc[:, -1].to_numpy()).all():
                print("Error: something is wrong. Labels do not match across modalities")
                break
            combined_dataframe.append(pd.concat((dataframe.iloc[:, :-1],
                                                 modality_data[fold]), axis=1))
    return combined_dataframe
