from sklearn.cluster import k_means
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedKFold
from utils import scores, set_seed, random_integers, sample, \
    retrieve_X_y, append_modality
from joblib import Parallel, delayed
import pandas as pd
import numpy as np
from ei import MedianAggregation, MeanAggregation
from ens_selection import CES
import copy
import sklearn.metrics
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.pipeline import Pipeline
import shap
import pickle
from itertools import groupby
from operator import itemgetter
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import LabelEncoder

import warnings
warnings.filterwarnings('ignore')

class EI_interpreter:
    """
    EI_object: Initialized EI object
    base_predictors: List of base predictors
    meta_models: List of meta models
    modalities: dictionary of multimodal dataset sorted by modality name
    y: label of dataset

    Returns
    -------
    feature_ranking: Dictionary of the overall feature ranking of multimodal
                    data to the target label by each meta model

    """
    def __init__(self, 
                 EI, 
                 metric,
                 n_repeats=10,
                 ensemble_methods='all'  # can be "all" or a list of keys for ensemble methods
                 ):
        
        self.EI = EI
        self.metric = metric
        self.n_repeats = n_repeats
        self.ensemble_methods = ensemble_methods

        self.LFR = None

    def local_feature_rank(self, X_dict, y):
        """
        Compute Local Feature Ranks (LFRs) of base predictors
        Parameters
        ----------
        X: data matrix of features of a modality
        modality: modality name
        feature_names: feature name of X
        """

        importance_list = []

        for modality_index, modality_name in enumerate(self.EI.modality_names):

            X = X_dict[modality_name]
            
            base_models = copy.deepcopy(self.EI.final_models["base models"][modality_name])

            base_models = sorted(base_models, key = itemgetter('model name'))

            for key, base_models_per_sample in groupby(base_models, key = itemgetter('model name')):
                
                list_of_base_models = []

                for base_model_dict in base_models_per_sample:
                    base_model = pickle.loads(base_model_dict["pickled model"])
                    list_of_base_models.append((str(base_model_dict["sample id"]), base_model,))  # list of tuples for VotingClassifier

                if len(list_of_base_models) > 1:  # create mean ensemble with base predictors with different sample ids

                    ###################################################################################################### 
                    #  This code is a work around and may be fragile. We use VotingClassifier to combine models trained on 
                    #  different samples (taking a mean of model output). The current sklearn implementation of 
                    #  VotingClassifier does not accept pretrained models, so we set parameters ourselves to allow it. In
                    #  the future it may be possible to use VotingClassifier alone without additional code.

                    model = VotingClassifier(estimators=list_of_base_models, 
                                                           voting='soft',
                                                           weights=np.ones(len(list_of_base_models))) # average predictions of models built on different data samples

                    model.estimators_ = [j for _, j in list_of_base_models]
                    model.le_ = LabelEncoder().fit(y)
                    model.classes_ = model.le_.classes_

                    ######################################################################################################
                
                else:

                    model = list_of_base_models[0]

                pi = permutation_importance(estimator=model,
                                            X=X,
                                            y=y,
                                            n_repeats=self.n_repeats,
                                            n_jobs=self.EI.n_jobs,
                                            random_state=self.EI.random_state,
                                            scoring=self.metric)

                pi_df = pd.DataFrame({"local_importance_mean": pi.importances_mean, 
                                    "local_importance_std": pi.importances_std, 
                                    "local_feature_id": range(X.shape[1])})
            
                pi_df['base predictor'] = base_model_dict["model name"]
                pi_df['modality'] = modality_name
                pi_df['LFR'] = pi_df["local_importance_mean"].rank(pct=True, ascending=False)
                # pi_df['sample'] = base_model_dict["sample id"]
                importance_list.append(pi_df)

        self.LFR = pd.concat(importance_list)


    def local_model_rank(self, meta_models_interested):
        X_train_list = []
        y_train_list = []
        # print(self.EI.meta_training_data)
        for fold_id in range(self.EI.k_outer):
            X_train, y_train = retrieve_X_y(labelled_data=self.meta_test_int[fold_id])
            X_train_list.append(X_train)
            y_train_list.append(y_train)

        meta_X_train = pd.concat(X_train_list)
        if self.EI.sampling_aggregation == "mean":
            meta_X_train = meta_X_train.groupby(level=[0, 1], axis=1).mean()
        meta_y_train = np.concatenate(y_train_list)
        # print(meta_X_train.shape, meta_y_train)
        lm_pi_list = []
        for model_name, model in meta_models_interested.items():
            if ('Mean' in model_name) or ('Median' in model_name):
                lm_pi = np.ones(len(meta_X_train.columns))
                # print(model_name, X_train.columns)
            elif model_name=='CES':
                model.fit(meta_X_train, meta_y_train)
                model.selected_ensemble
                model_selected_freq = []
                for bp in meta_X_train.columns:
                    model_selected_freq.append(model.selected_ensemble.count(bp))
                lm_pi = model_selected_freq
            else:
                if type(model)==Pipeline:
                    est_ = list(model.named_steps)[-1]
                    if hasattr(model[est_], 'random_state') and hasattr(model[est_], 'set_params'):
                        model.set_params(**{'{}__random_state'.format(est_):self.random_state})
                if hasattr(model, 'random_state') and hasattr(model, 'set_params'):
                    model.set_params(**{'random_state': self.random_state})
                model.fit(meta_X_train, meta_y_train)
                # model.fit()
                if self.shap_val:
                    # shap_exp = shap.Explainer(model)
                    # shap_vals = shap_exp.shap_values(meta_X_train)
                    # print(shap_vals)
                    lm_pi = self.shap_val_mean(model, meta_X_train)
                else:
                    lm_pi = permutation_importance(estimator=model,
                                                X=meta_X_train,
                                                y=meta_y_train,
                                                n_repeats=self.n_repeats,
                                                n_jobs=-1,
                                                random_state=self.random_state,
                                                scoring=self.metric)
                lm_pi = lm_pi.importances_mean

            pi_df = pd.DataFrame({'local_model_PI': lm_pi,
                                    'base predictor': [i[1] for i in meta_X_train.columns],
                                    'modality': [i[0] for i in meta_X_train.columns],
                                    # 'sample': [i[2] for i in meta_X_train.columns]
                                    })
            
            pi_df['ensemble_method'] = model_name
            pi_df['LMR'] = pi_df['local_model_PI'].rank(pct=True, ascending=False)
            lm_pi_list.append(pi_df)
        self.LMRs = pd.concat(lm_pi_list)
        # print(self.LMRs)

    def shap_val_mean(self, m, x):
        if hasattr(m, "predict_proba"):
            shap_exp = shap.Explainer(m.predict_proba, x)
        else:
            shap_exp = shap.Explainer(m.predict, x)
        
        shap_vals = shap_exp(x)
        print(shap_vals.values.shape)
        return np.mean(shap_vals, axis=1)

    def rank_product_score(self, X_dict, y):

        self.local_feature_rank(X_dict, y)

        meta_models = copy.deepcopy(self.EI.final_models["meta models"])

        breakpoint()

        self.local_model_rank(meta_models_interested=self.ensemble_methods)

        """Calculate the Rank percentile & their products here"""
        # return feature_ranking
        feature_ranking_list = {}
        self.merged_lmr_lfr = {}
        for model_name in ens_list:
            lmr_interest = self.LMRs[self.LMRs['ensemble_method']==model_name].copy()
            self.merged_lmr_lfr[model_name] = pd.merge(lmr_interest, self.LFRs,  
                                        how='right', left_on=['base predictor','modality'], 
                                        right_on = ['base predictor','modality'])
            # print(merged_lmr_lfr)
            self.merged_lmr_lfr[model_name]['LMR_LFR_product'] = self.merged_lmr_lfr[model_name]['LMR']*self.merged_lmr_lfr[model_name]['LFR']
            """ take mean of LMR*LFR for each feature """
            RPS_list = {'modality':[],
                        'feature': [],
                        'RPS': []}
            print(self.merged_lmr_lfr[model_name])
            for modal in self.merged_lmr_lfr[model_name]['modality'].unique():
                merged_lmr_lfr_modal = self.merged_lmr_lfr[model_name].loc[self.merged_lmr_lfr[model_name]['modality']==modal]
                for feat in merged_lmr_lfr_modal['local_feat_name'].unique():
                    RPS_list['modality'].append(modal)
                    RPS_list['feature'].append(feat)
                    RPS_list['RPS'].append(merged_lmr_lfr_modal.loc[merged_lmr_lfr_modal['local_feat_name']==feat, 
                                            'LMR_LFR_product'].mean())
            RPS_df = pd.DataFrame(RPS_list)
            RPS_df['feature rank'] = RPS_df['RPS'].rank(ascending=True)
            RPS_df['ensemble method'] = model_name
            RPS_df.sort_values(by='feature rank',inplace=True)
            feature_ranking_list[model_name] = RPS_df
        self.ensemble_feature_ranking = feature_ranking_list
        print('Finished feature ranking of ensemble model(s)!')