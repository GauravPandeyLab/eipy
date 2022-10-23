import pandas as pd
import datetime
import numpy as np
import os
from os.path import exists, abspath, isdir
from os import mkdir
from sys import argv
from itertools import product
from sklearn.model_selection import KFold, StratifiedKFold
# import common

def convert_to_arff(df, path):
    fn = open(path, 'w')
    fn.write('@relation yanchakli\n')
    col_counter = 0
    for col in df.columns:
        if col not in ['cls', 'seqID', 'fold']:
            fn.write('@attribute %s numeric\n' % col)
            col_counter += 1
        elif col == 'cls':
            fn.write('@attribute cls {pos,neg}\n')
            col_counter += 1
        elif col == 'seqID':
            fn.write('@attribute seqID string\n')
            col_counter += 1
        elif col == 'fold':
            fold_values = list(df['fold'].astype(str).unique())
            fold_str = ','.join(fold_values)
            fn.write('@attribute fold {' + fold_str + '}\n')


    print('col counter:', col_counter)

    fn.write('@data\n')
    print(path, 'start to write df')
    cont = df.to_csv(index=False, header=None)
    fn.write(cont)
    print(path, 'finished writing df')
    fn.close()


def processTermArff(param, impute, fold=5):
    term, feature, go_hpo_df, csv_filepath = param
    if impute:
        feature_df = pd.read_csv('{}rwrImputed_{}.csv'.format(csv_filepath, feature), index_col=0)
        print('using imputed data')
    else:
        feature_df = pd.read_csv('{}{}.csv'.format(csv_filepath, feature), index_col=0)

    before_shape = feature_df.shape
    go_hpo_df.fillna(0, inplace=True)
    go_hpo_df = go_hpo_df[go_hpo_df != 0]
    go_hpo_df.replace(-1, 'neg', inplace=True)
    go_hpo_df.replace(1, 'pos', inplace=True)

    filled_df = feature_df.fillna(0)
    cols = (filled_df == 0).all(axis=0)
    cols = cols.loc[cols == False].index.tolist()
    filled_df = filled_df.loc[:,cols]
    filled_df = filled_df.round(5)
    # filled_df = filled_df.iloc[:, 0:100]
    # merged_df = pd.merge(filled_df, go_hpo_df, how='inner')
    merged_df = pd.concat([filled_df, go_hpo_df], axis=1, join='inner')
    merged_df.rename(columns={term: 'cls'}, inplace=True)
    merged_df['seqID'] = merged_df.index
    print('before', merged_df.shape)
    merged_df.dropna(inplace=True)
    print(merged_df['cls'].value_counts())
    # merged_df = merged_df.iloc[0:500,:]
    kf_split = StratifiedKFold(n_splits=fold, shuffle=True, random_state=142)
    # kf_split = KFold(n_splits=fold, shuffle=True, random_state=64)
    kf_idx_list = kf_split.split(merged_df, y=merged_df['cls'])
    merged_df.reset_index(inplace=True, drop=True)
    merged_df['fold'] = 0
    for fold_attr, (kf_train_idx, kf_test_idx) in enumerate(kf_idx_list):
        # f_bool = np.zeros(merged_df.shape[0], int)
        # f_bool[kf_test_idx] = 1
        # merged_df.loc[f_bool,'fold'] = fold_attr
        merged_df.loc[kf_test_idx,'fold'] = fold_attr+10000

    print('after', merged_df.shape)
    # del merged_df.index.name
    # print(term, 'merged df')
    p = os.path.join(scratch_data_dir, feature)
    print(p)
    if not exists(p):
        mkdir(p)
    path = os.path.join(p, 'data.arff')
    convert_to_arff(merged_df, path)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Feed some bsub parameters')
    parser.add_argument('--outcome', type=str, required=True, help='data path')
    parser.add_argument('--output_dir', type=str, default='./', help='attribute importance')
    parser.add_argument('--method', type=str, default='EI', help='attribute importance')
    parser.add_argument('--feature_csv_path', type=str, required=True, help='attribute importance')
    parser.add_argument('--outcome_tsv_path', type=str, required=True, help='attribute importance')
    # parser.add_argument('--outcome_tsv_name', type=str, required=True, help='attribute importance')
    args = parser.parse_args()
    # scratch_data_dir = '/sc/arion/scratch/liy42/'
    # group_number_goterm = argv[2]

    # TODO: predefine foldAttribute
    # if len(argv) > 4:
    #     fold = int(argv[4])
    # else:
    fold = 5
    #
    #
    #
    # if 'Impute' in group_number_goterm:
    #     impute_graph = True
    # else:
    impute_graph = False

    csv_dir = args.feature_csv_path
    tsv_dir = args.outcome_tsv_path
    # if 'go' in args.outcome.lower():
    #     go_to_hpo_file = args.outcome_tsv_name
    if args.method != 'EI':
        features = [args.method]
    else:
        features = ['coexpression', 'cooccurence', 'database',
                    'experimental', 'fusion', 'neighborhood',]

    term = args.outcome

    t = term.split(':')[0] + term.split(':')[1]
    scratch_data_dir = args.output_dir

    if not exists(scratch_data_dir):
        mkdir(scratch_data_dir)

    scratch_data_dir = os.path.join(scratch_data_dir, t+'/')


    if not exists(scratch_data_dir):
        mkdir(scratch_data_dir)

    os.system('cp sample_data/classifiers.txt {}'.format(scratch_data_dir[:-1]))
    os.system('cp sample_data/weka.properties {}'.format(scratch_data_dir[:-1]))

    # for f in features:
    #     dest_dir = scratch_data_dir + f
    #     os.system('cp sample_data/classifiers.txt {}'.format(dest_dir))
    #     os.system('cp sample_data/weka.properties {}'.format(dest_dir))
    #
    # if len(features) > 0:
    for feature in features:
        f_dir = os.path.join(scratch_data_dir, feature+'/')
        if not exists(f_dir):
            mkdir(f_dir)
        if len(features) > 1:
            dest_dir = scratch_data_dir + feature
            os.system('cp sample_data/classifiers.txt {}'.format(dest_dir))
            os.system('cp sample_data/weka.properties {}'.format(dest_dir))


    # deepNF_net = pd.read_csv('/sc/hydra/scratch/liy42/deepNF/%s/%s.arff' %(t,t), header=None,comment='@')
    # seqs = deepNF_net.iloc[:,-1].tolist()
    # labels = deepNF_net.iloc[:,-2].tolist()
    # go_to_hpo_df = pd.read_csv(os.path.join(tsv_dir, go_to_hpo_file), sep='\t', index_col=0)
    go_to_hpo_df = pd.read_csv(tsv_dir, sep='\t', index_col=0)
    print(term)
    go_to_hpo_df_with_specific_term = go_to_hpo_df[[term]]


    params = list(product([term], features, [go_to_hpo_df_with_specific_term], [csv_dir]))
    print(
        '[STARTED: %s] start multithreads computing to generate feature files for GO term: %s...............................' % (
        str(datetime.datetime.now()), term))
    for param in params:
        cols = processTermArff(param, impute=impute_graph, fold=fold)

    # print(col_intersect.shape)
    # print(col_intersect)
    #
    # for param in params:
    #     preprocessingTermFeat(param, impute=impute_graph, index_list=col_intersect, fold=fold)



    # p = Pool(6)

    # p.map(processTermFeature_3, params)
    print(
        '[FINISHED: %s] completed the generation of feature files for GO term: %s...........................................' % (
        str(datetime.datetime.now()), term))
