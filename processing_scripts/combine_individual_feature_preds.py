'''
Comine predictions from training base classifiers based on one feature set.
Author: Linhua (Alex) Wang
Date:  1/02/2019
'''

from glob import glob
import gzip
from os.path import abspath, exists, isdir
from os import listdir
from sys import argv
from processing_scripts.common import load_arff_headers, load_properties, data_dir_list, read_arff_to_pandas_df, str2bool
from pandas import concat, read_csv

def merged_base_innerCV_by_outerfold(f_list, path):
    dirnames = sorted(filter(isdir, glob('%s/weka.classifiers.*' % path)))
    print(f_list)
    if not test_model:
        for fold in f_list:
            dirname_dfs = []
            for dirname in dirnames:
                classifier = dirname.split('.')[-1]
                nested_fold_dfs = []
                for nested_fold in range(nested_fold_count):
                    bag_dfs = []
                    for bag in range(bag_count):
                        filename = '%s/validation-%s-%02i-%02i.csv.gz' % (dirname, fold, nested_fold, bag)
                        try:
                            if test_model:
                                df = read_csv(filename, skiprows=1, index_col=0, compression='gzip',
                                              engine='python')
                            else:
                                df = read_csv(filename, skiprows=1, index_col=[0, 1], compression='gzip', engine='python')
                            # df = read_csv(filename, skiprows=1, index_col=[0, 1], compression='gzip', engine='python')
                            df = df[['prediction']]
                            df.rename(columns={'prediction': '%s.%s' % (classifier, bag)}, inplace=True)
                            bag_dfs.append(df)
                        except:
                            print('file not existed or crashed %s' % filename)
                    nested_fold_dfs.append(concat(bag_dfs, axis=1))

                dirname_dfs.append(concat(nested_fold_dfs, axis=0))
            # concat(dirname_dfs, axis=1).sort_index().to_csv('%s/validation-%s.csv.gz' % (path, fold), compression='gzip')
            concat(dirname_dfs, axis=1).dropna().sort_index().to_csv('%s/validation-%s.csv.gz' % (path, fold), compression='gzip')


    # for fold in range(fold_count):
    for fold in f_list:
        dirname_dfs = []
        dirname_attribute_imp_dfs = []
        for dirname in dirnames:
            classifier = dirname.split('.')[-1]
            bag_dfs = []
            attribute_imp_dfs = []
            for bag in range(bag_count):
                if attr_imp_bool:
                    # print('running attribute imp1')
                    attribute_imp_filename = '%s/attribute_imp-%s-%02i.csv.gz' % (dirname, fold, bag)
                filename = '%s/predictions-%s-%02i.csv.gz' % (dirname, fold, bag)
                print(filename)
                try:
                    if test_model:
                        df = read_csv(filename, skiprows=1, index_col=0, compression='gzip',
                                      engine='python')
                    else:
                        df = read_csv(filename, skiprows=1, index_col=[0, 1], compression='gzip', engine='python')
                    df = df[['prediction']]
                    df.rename(columns={'prediction': '%s.%s' % (classifier, bag)}, inplace=True)
                    bag_dfs.append(df)
                    if attr_imp_bool:
                        attribute_imp_df = read_csv(attribute_imp_filename, compression='gzip', engine='python')
                        attribute_imp_dfs.append(attribute_imp_df)
                except:
                    print('file not existed or crashed %s' % filename)
            dirname_dfs.append(concat(bag_dfs, axis=1))
            if attr_imp_bool:
                dirname_attribute_imp_dfs.append(concat(attribute_imp_dfs, ignore_index=True))

        # concat(dirname_dfs, axis=1).sort_index().to_csv('%s/predictions-%s.csv.gz' % (path, fold), compression='gzip')
        concat(dirname_dfs, axis=1).dropna().sort_index().to_csv('%s/predictions-%s.csv.gz' % (path, fold), compression='gzip')
        if attr_imp_bool:
            concat(dirname_attribute_imp_dfs, ignore_index=True).to_csv('%s/attribute_imp-%s.csv.gz' % (path, fold), compression='gzip')

def combine_individual(f_values, path):
    merged_base_innerCV_by_outerfold(f_values, path)
    # merged_base_innerCV_by_outerfold(pca_fold_values, path)



data_folder = abspath(argv[1])
attr_imp_bool = str2bool(argv[2])
test_model = str2bool(argv[3])

# print(attr_imp_bool)
feature_folders = data_dir_list(data_folder)
# data_name = data_folder.split('/')[-1]
# fns = listdir(data_folder)
# fns = [fn for fn in fns if fn != 'analysis']
# fns = [data_folder + '/' + fn for fn in fns]
# feature_folders = [fn for fn in fns if isdir(fn)]



p = load_properties(data_folder)
# fold_count = int(p['foldCount'])
if 'foldAttribute' in p:
    # input_fn = '%s/%s' % (feature_folders[0], 'data.arff')
    # assert exists(input_fn)
    # headers = load_arff_headers(input_fn)
    # fold_values = headers[p['foldAttribute']]
    df = read_arff_to_pandas_df(feature_folders[0] + '/data.arff')
    fold_values = df[p['foldAttribute']].unique()

else:
    fold_values = range(int(p['foldCount']))

if test_model:
    fold_values = ['test']

pca_fold_values = ['pca_{}'.format(fv) for fv in fold_values]

nested_fold_count = int(p['nestedFoldCount'])
bag_count = max(1, int(p['bagCount']))
for path_f in feature_folders:
    combine_individual(fold_values, path_f)
