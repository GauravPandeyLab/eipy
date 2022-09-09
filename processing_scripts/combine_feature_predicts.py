'''
Combine predictions of models using different feature sets.
Author: Linhua (Alex) Wang
Date:  12/27/2018
'''
from os.path import exists, abspath, isdir, dirname
from sys import argv
from os import listdir, environ
from processing_scripts.common import load_properties, load_arff_headers, data_dir_list, read_arff_to_pandas_df, str2bool
import pandas as pd
import numpy as np

data_folder = abspath(argv[1])
attr_imp_bool = str2bool(argv[2])
test_model = str2bool(argv[3])
#
# fns = listdir(data_folder)
# fns = [fn for fn in fns if fn != 'analysis']
# fns = [data_folder  + '/' + fn for fn in fns]
# feature_folders = [fn for fn in fns if isdir(fn)]

feature_folders = data_dir_list(data_folder)

# foldValues = range(int(argv[2]))
p = load_properties(data_folder)
# fold_count = int(p['foldCount'])
if 'foldAttribute' in p:
    # input_fn = '%s/%s' % (feature_folders[0], 'data.arff')
    # assert exists(input_fn)
    # headers = load_arff_headers(input_fn)
    # fold_values = headers[p['foldAttribute']]
    # fold_values = ['67890']
    df = read_arff_to_pandas_df(feature_folders[0] + '/data.arff')
    fold_values = list(df[p['foldAttribute']].unique())

else:
    fold_values = range(int(p['foldCount']))

if test_model:
    fold_values = ['test']

pca_fold_values = ['pca_{}'.format(fv) for fv in fold_values]
prediction_dfs = []
validation_dfs = []


# for value in foldValues:


def merge_base_feat_preds_by_fold(f_list):
    print(f_list)
    for value in f_list:
        prediction_dfs = []
        validation_dfs = []
        attribute_imp_dfs = []
        # Combine
        for folder in feature_folders:
            feature_name = folder.split('/')[-1]

            prediction_df = pd.read_csv(folder + '/predictions-%s.csv.gz' % value, compression='gzip')
            print(feature_name)
            if test_model:
                prediction_df.set_index(['id'], inplace=True)
            else:
                prediction_df.set_index(['id', 'label'], inplace=True)
            prediction_df.columns = ['%s.%s' % (feature_name, col) for col in prediction_df.columns]

            prediction_dfs.append(prediction_df)
            if not test_model:
                validation_df = pd.read_csv(folder + '/validation-%s.csv.gz' % value, compression='gzip')
                validation_df.set_index(['id', 'label'], inplace=True)
                validation_df.columns = ['%s.%s' % (feature_name, col) for col in validation_df.columns]
                validation_dfs.append(validation_df)

            if attr_imp_bool:
                attribute_imp_df = pd.read_csv(folder + '/attribute_imp-%s.csv.gz' % value, compression='gzip')
                attribute_imp_df['modality'] = feature_name
                attribute_imp_dfs.append(attribute_imp_df)

        print(prediction_dfs)
        prediction_dfs = pd.concat(prediction_dfs, axis=1).dropna()
        prediction_dfs.to_csv(data_folder + '/predictions-%s.csv.gz' % value, compression='gzip')
        if not test_model:
            validation_dfs = pd.concat(validation_dfs, axis=1).dropna()
            validation_dfs.to_csv(data_folder + '/validation-%s.csv.gz' % value, compression='gzip')
        if attr_imp_bool:
            # print('running attribute imp2')
            attribute_imp_dfs = pd.concat(attribute_imp_dfs)
            attribute_imp_dfs.to_csv(data_folder + '/attribute_imp-%s.csv.gz' % value, compression='gzip')


merge_base_feat_preds_by_fold(fold_values)
# merge_base_feat_preds_by_fold(pca_fold_values)
