from numpy import argmax, argmin, argsort, corrcoef, mean, nanmax, sqrt, triu_indices_from, where
from pandas import DataFrame, concat, read_csv
from scipy.io.arff import loadarff
import sklearn.metrics
import numpy as np
import os
from os.path import exists,abspath,isdir,dirname
from sys import argv
from os import listdir,environ
import pandas as pd
import argparse


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def argsortbest(x):
    return argsort(x) if greater_is_better else argsort(x)[::-1]


def average_pearson_score(x):
    if isinstance(x, DataFrame):
        x = x.values
    rho = corrcoef(x, rowvar = 0)
    return mean(abs(rho[triu_indices_from(rho, 1)]))


def get_best_performer(df, one_se = False, _greater_is_better=True):
    if not one_se:
        return df[df.score == best(df.score)].head(1)
    se = df.score.std() / sqrt(df.shape[0] - 1)
    if _greater_is_better:
        return df[df.score >= best(df.score) - se].head(1)
    return df[df.score <= best(df.score) + se].head(1)


def confusion_matrix_fpr(labels, predictions, false_discovery_rate = 0.1):
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(labels, predictions)
    max_fpr_index = where(fpr >= false_discovery_rate)[0][0]
    print(sklearn.metrics.confusion_matrix(labels, predictions > thresholds[max_fpr_index]))


def fmeasure_score(labels, predictions, thres=None, beta = 1.0, pos_label = 1):
    """
        Radivojac, P. et al. (2013). A Large-Scale Evaluation of Computational Protein Function Prediction. Nature Methods, 10(3), 221-227.
        Manning, C. D. et al. (2008). Evaluation in Information Retrieval. In Introduction to Information Retrieval. Cambridge University Press.
    """
    if thres is None:
        precision, recall, threshold = sklearn.metrics.precision_recall_curve(labels, predictions,
                                                                              pos_label=pos_label)
        # f1 = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)
        f1 = 2 * (precision * recall) / (precision + recall)
        # print(threshold)
        # if len(threshold[where(f1==nanmax(f1))]) > 1:
        fmax_point = where(f1==nanmax(f1))[0]
        # if len(fmax_point) > 0:
        p_maxes = precision[fmax_point]
        r_maxes = recall[fmax_point]
        pr_diff = np.abs(p_maxes - r_maxes)
        balance_fmax_point = where(pr_diff == min(pr_diff))[0]
        p_max = p_maxes[balance_fmax_point][0]
        r_max = r_maxes[balance_fmax_point][0]

        opt_threshold = threshold[fmax_point][balance_fmax_point][0]

        # r_max = recall[where(f1==nanmax(f1))[0]][0]
        # p_max = precision[where(f1 == nanmax(f1))[0]][0]
        # else:
        #     opt_threshold = threshold[where(f1 == nanmax(f1))][0]
        #     r_max = recall[where(f1 == nanmax(f1))][0]
        #     p_max = precision[where(f1 == nanmax(f1))][0]
        return {'F':nanmax(f1), 'thres':opt_threshold, 'P':p_max, 'R':r_max, 'PR-curve': [precision, recall]}

    else:
        predictions[predictions > thres] = 1
        predictions[predictions <= thres] = 0
        precision, recall, fmeasure, _ = sklearn.metrics.precision_recall_fscore_support(labels,
                                                                                      predictions, average='binary')
        return {'P':precision, 'R':recall, 'F':fmeasure}

def auprc(y_true, y_scores):
    # precision, recall, thresholds = sklearn.metrics.precision_recall_curve(y_true, y_scores, pos_label=1)
    # return sklearn.metrics.auc(recall, precision)
    return sklearn.metrics.average_precision_score(y_true, y_scores)

def f_max(labels, predictions, thres=None, beta = 1.0, pos_label = 1):
    """
        Radivojac, P. et al. (2013). A Large-Scale Evaluation of Computational Protein Function Prediction. Nature Methods, 10(3), 221-227.
        Manning, C. D. et al. (2008). Evaluation in Information Retrieval. In Introduction to Information Retrieval. Cambridge University Press.
    """
    precision, recall, threshold = sklearn.metrics.precision_recall_curve(labels, predictions,
                                                                          pos_label=pos_label)
    f1 = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)
    # print(threshold)
    if len(threshold[where(f1==nanmax(f1))]) > 1:
        opt_threshold = threshold[where(f1==nanmax(f1))][0]
    else:
        opt_threshold = threshold[where(f1 == nanmax(f1))]
    return nanmax(f1)

# def fmeasure(labels, predictions)

def load_cmd_option(cmd, option):
    return cmd.split('--{} '.format(option))[-1].split(' --')[0]

def load_arff(filename):
    return DataFrame.from_records(loadarff(filename)[0])


def load_arff_headers(filename):
    dtypes = {}
    for line in open(filename):
        if line.startswith('@data'):
            break
        if line.startswith('@attribute'):
            _, name, dtype = line.split()
            if dtype.startswith('{'):
                dtype = dtype[1:-1]
            dtypes[name] = set(dtype.split(','))
    return dtypes


def load_properties(dirname):
    properties = [_.split('=') for _ in open(dirname + '/weka.properties').readlines()]
    d = {}
    for key, value in properties:
        d[key.strip()] = value.strip()
    return d


def read_fold(path, fold):
    train_df        = read_csv('%s/validation-%s.csv.gz' % (path, fold), index_col = [0, 1], compression = 'gzip')
    # print(train_df[train_df.isna().any(axis=1)])
    test_df         = read_csv('%s/predictions-%s.csv.gz' % (path, fold), index_col = [0, 1], compression = 'gzip')
    train_labels    = train_df.index.get_level_values('label').values
    test_labels     = test_df.index.get_level_values('label').values
    # train_labels = train_df.index.get_level_values('label')
    # test_labels = test_df.index.get_level_values('label')

    return train_df, train_labels, test_df, test_labels


def rmse_score(a, b):
    return sqrt(mean((a - b)**2))


def unbag(df, bag_count, remove_label=False):
    cols = []
    bag_start_indices = range(0, df.shape[1], bag_count)
    # names = [_.split('.')[0] for _ in df.columns.values[bag_start_indices]]
    names = [_.rsplit('.',1)[0] for _ in df.columns.values[bag_start_indices]]
    for i in bag_start_indices:
        cols.append(df.iloc[:, i:i+bag_count].mean(axis = 1))
    df = concat(cols, axis = 1)
    df.columns = names
    if remove_label:
        df.reset_index(inplace=True)
        df.set_index('id', inplace=True)
    return df

def check_dir_n_mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)

def data_dir_list(data_path, excluding_folder = ['analysis', 'feature_rank', 'model_built']):
    fns = listdir(data_path)
    fns = [fn for fn in fns if not fn in excluding_folder]
    fns = [fn for fn in fns if not 'tcca' in fn]
    fns = [fn for fn in fns if not 'pca' in fn]
    fns = [fn for fn in fns if not 'weka.classifiers' in fn]
    fns = [data_path + '/' + fn for fn in fns]
    feature_folders = [fn for fn in fns if isdir(fn)]
    return feature_folders

def read_arff_to_pandas_df(arff_path):
    # loadarff doesn't support string attribute
    # data = arff.loadarff(arff_path)
    df = pd.read_csv(arff_path, comment='@', header=None)
    num_col = df.shape[1]
    columns = []
    file1 = open(arff_path, 'r')
    Lines = file1.readlines()

    count = 0
    # Strips the newline character
    for line_idx, line in enumerate(Lines):
        # if line_idx > num_col
        if '@attribute' in line:
            columns.append(line.strip().split(' ')[1])

    df.columns = columns
    return df

diversity_score = average_pearson_score
# score = sklearn.metrics.roc_auc_score
score = sklearn.metrics.roc_auc_score
greater_is_better = True
best = max if greater_is_better else min
argbest = argmax if greater_is_better else argmin
fmax_scorer = sklearn.metrics.make_scorer(fmeasure_score, greater_is_better = True, needs_threshold = True)
