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
    def __init__(self, EI_object, base_predictors,
                 meta_models, modalities, y, metric,
                 feature_dict=None,
                 n_jobs=-1,
                 n_repeats=10,
                 random_state=42,
                 ensemble_of_interest='ALL'):
        self.EI = copy.copy(EI_object)
        self.k_outer = self.EI.k_outer
        self.base_predictors = copy.copy(base_predictors)
        self.meta_models = copy.copy(meta_models)
        self.meta_test_int = None

        self.y = y
        self.metric = metric
        if feature_dict is None:
            self.feature_dict = {}
            self.modalities = {}
            for modal_name, modality in modalities.items():
                if type(modality) == pd.core.frame.DataFrame:
                    """if the data input is dataframe, store the feature name"""
                    self.feature_dict[modal_name] = list(modality.columns)
                    self.modalities[modal_name] = modality.values
                    # print(modal_name, modality.shape)
                
                else:
                    """If there is no feature name in input/feature name dictionary"""
                    self.feature_dict[modal_name] = ['{}_{}'.format(modal_name, i) for i in range(modality.shape[1])]
                    self.modalities[modal_name] = modality
        else:
            self.feature_dict = feature_dict
            self.modalities = modalities
        self.n_jobs = n_jobs
        self.n_repeats = n_repeats
        self.ensemble_of_interest = ensemble_of_interest
        self.random_state = random_state
        self.LFRs = []
        self.LMRs = None
        self.ensemble_feature_ranking = None

    def local_feature_rank(self, X, modality):
        """
        Compute Local Feature Ranks (LFRs) of base predictors
        Parameters
        ----------
        X: data matrix of features of a modality
        modality: modality name
        feature_names: feature name of X
        """
        if self.base_predictors is not None:
            self.base_predictors = self.EI.base_predictors  # update base predictors

        if modality is not None:
            print(f"\n Working on {modality} data... \n")
            # EI_obj.base_predictors = update_keys(dictionary=EI_obj.base_predictors,
            #                                      string=modality)  # include modality in model name
        
        meta_test_temp = self.EI.train_base_outer(X, self.y, self.EI.cv_outer, self.EI.base_predictors, modality)
        
        self.meta_test_int = append_modality(self.meta_test_int, meta_test_temp)

        lf_pi_list = []
        for model_name, model in self.base_predictors.items():
            if self.EI.calibration_model is not None:
                self.EI.calibration_model.base_estimator = model
                model = self.EI.calibration_model
            model.fit(X, self.y)
            lf_pi = permutation_importance(estimator=model,
                                           X=X,
                                           y=self.y,
                                           n_repeats=self.n_repeats,
                                           n_jobs=-1,
                                           random_state=self.random_state,
                                           scoring=self.metric)
            # pi_df = pd.DataFrame(data=[lf_pi.importances_mean], 
                                # columns=self.feature_dict[modality], index=[0])
            pi_df = pd.DataFrame({'local_feat_PI': lf_pi.importances_mean, 
                                'local_feat_name': self.feature_dict[modality]})
            pi_df['base predictor'] = model_name
            pi_df['modality'] = modality
            pi_df['LFR'] = pi_df['local_feat_PI'].rank(pct=True, ascending=False)
            lf_pi_list.append(pi_df)

        self.LFRs.append(pd.concat(lf_pi_list))
        # print(self.LFRs)

    def local_model_rank(self, meta_models_interested):
        X_train_list = []
        y_train_list = []
        # print(self.EI.meta_training_data)
        for fold_id in range(self.EI.k_outer):
            X_train, y_train = retrieve_X_y(labelled_data=self.meta_test_int[fold_id])
            X_train_list.append(X_train)
            y_train_list.append(y_train)

        meta_X_train = pd.concat(X_train_list)
        meta_y_train = np.concatenate(y_train_list)
        print(meta_X_train.shape, meta_y_train)
        lm_pi_list = []
        for model_name, model in meta_models_interested.items():
            if ('Mean' in model_name) or ('Median' in model_name):
                lm_pi = np.ones(len(meta_X_train.columns))
                # print(model_name, X_train.columns)
            
            elif 'CES' == model_name:
                model.fit(meta_X_train, meta_y_train)
                """TODO"""
            else:
                model.fit(meta_X_train, meta_y_train)
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
                                    'sample': [i[2] for i in meta_X_train.columns]})
            
            pi_df['ensemble_method'] = model_name
            pi_df['LMR'] = pi_df['local_model_PI'].rank(pct=True, ascending=False)
            lm_pi_list.append(pi_df)
        self.LMRs = pd.concat(lm_pi_list)
        # print(self.LMRs)


    def rank_product_score(self):
        for modal_name, modality_data in self.modalities.items():
            self.local_feature_rank(modality_data, modality=modal_name)
        self.LFRs = pd.concat(self.LFRs)

        """Add mean/median aggregation here"""
        meta_models = {"S." + k: v for k, v in self.meta_models.items() if not (k in ["Mean", "Median"])}
        if self.ensemble_of_interest == "ALL":
            if not ("Mean" in meta_models):
                meta_models["Mean"] = MeanAggregation()
            elif not ("Median" in meta_models):
                meta_models["Median"] = MedianAggregation()
            # elif not ("CES" in meta_models):
            #     meta_models["CES"] = CES()
        self.meta_models = meta_models


        if self.ensemble_of_interest == 'ALL':
            self.local_model_rank(meta_models_interested=self.meta_models)
            ens_list = [k for k, v in self.meta_models.items()]
        else:
            self.local_model_rank(meta_models_interested=self.ensemble_of_interest)
            ens_list = self.ensemble_of_interest
        """Calculate the Rank percentile & their products here"""
        # return feature_ranking
        feature_ranking_list = {}
        for model_name in ens_list:
            lmr_interest = self.LMRs[self.LMRs['ensemble_method']==model_name].copy()
            merged_lmr_lfr = pd.merge(lmr_interest, self.LFRs,  
                                        how='right', left_on=['base predictor','modality'], 
                                        right_on = ['base predictor','modality'])
            # print(merged_lmr_lfr)
            merged_lmr_lfr['LMR_LFR_product'] = merged_lmr_lfr['LMR']*merged_lmr_lfr['LFR']
            """ take mean of LMR*LFR for each feature """
            RPS_list = {'modality':[],
                        'feature': [],
                        'RPS': []}
            print(merged_lmr_lfr)
            for modal in merged_lmr_lfr['modality'].unique():
                merged_lmr_lfr_modal = merged_lmr_lfr.loc[merged_lmr_lfr['modality']==modal]
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
