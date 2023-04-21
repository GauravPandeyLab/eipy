from sklearn.inspection import permutation_importance
from eipy.utils import scores, set_seed, random_integers, sample, \
    retrieve_X_y, append_modality, generate_scorer_by_model
import pandas as pd
import numpy as np
import copy
from sklearn.metrics import fbeta_score, make_scorer
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
                 ensemble_methods='all',
                 metric_greater_is_better = True  # can be "all" or a list of keys for ensemble methods
                 ):
        
        self.EI = EI
        self.metric = metric
        self.n_repeats = n_repeats
        self.ensemble_methods = ensemble_methods
        self.metric_greater_is_better = metric_greater_is_better

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

        print("     Calculating local feature ranks...", end=" ")

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
                    #  the future it may be possible to use VotingClassifier alone without additional code. An 
                    #  sklearn-like model is needed to be passed to permutation_importance

                    model = VotingClassifier(estimators=list_of_base_models, 
                                             voting='soft',
                                             weights=np.ones(len(list_of_base_models))) # average predictions of models built on different data samples

                    model.estimators_ = [j for _, j in list_of_base_models]
                    model.le_ = LabelEncoder().fit(y)
                    model.classes_ = model.le_.classes_

                    ######################################################################################################
                
                else:
                    model = list_of_base_models[0][1]

                needs_proba = hasattr(base_model, "predict_proba")
                scorer_ = make_scorer(self.metric, 
                                        greater_is_better=self.metric_greater_is_better,
                                        needs_proba=needs_proba)

                pi = permutation_importance(estimator=model,
                                            X=X,
                                            y=y,
                                            n_repeats=self.n_repeats,
                                            n_jobs=self.EI.n_jobs,
                                            random_state=self.EI.random_state,
                                            scoring=scorer_)

                pi_df = pd.DataFrame({"local_importance_mean": pi.importances_mean, 
                                      "local_importance_std": pi.importances_std, 
                                      "local_feature_id": range(X.shape[1])
                                      })
            
                pi_df['base predictor'] = base_model_dict["model name"]
                pi_df['modality'] = modality_name
                pi_df['LFR'] = pi_df["local_importance_mean"].rank(pct=True, ascending=False)
                importance_list.append(pi_df)

        self.LFR = pd.concat(importance_list)

        print("complete!")


    def local_model_rank(self, ensemble_model_keys):

        print("     Calculating local model ranks...", end=" ")

        #  load meta training data from EI training

        meta_X_train, meta_y_train = retrieve_X_y(labelled_data=self.EI.meta_training_data_final[0])

        if self.EI.sampling_aggregation == "mean":
            meta_X_train = meta_X_train.groupby(level=[0, 1], axis=1).mean()

        #  calculate importance for ensemble models of interest

        lm_pi_list = []

        ensemble_models = copy.deepcopy(self.EI.final_models["meta models"])

        ensemble_models = itemgetter(*ensemble_model_keys)(ensemble_models)  

        ensemble_models = dict(zip(ensemble_model_keys, ensemble_models)) 

        for model_name, model in ensemble_models.items():

            meta_model = pickle.loads(model)

            if ('Mean' in model_name) or ('Median' in model_name):

                importances_mean = np.ones(len(meta_X_train.columns))
                importances_std = np.zeros(len(meta_X_train.columns))

            elif model_name=='CES':

                model_selected_freq = []
                for bp in meta_X_train.columns:
                    model_selected_freq.append(meta_model.selected_ensemble.count(bp))
                importances_mean = model_selected_freq
                importances_std = np.ones(len(meta_X_train.columns)) * np.nan

            else:
                needs_proba = hasattr(model, "predict_proba")
                scorer_ = make_scorer(self.metric, 
                                    greater_is_better=self.metric_greater_is_better,
                                    needs_proba=needs_proba)
                pi = permutation_importance(estimator=meta_model,
                                            X=meta_X_train,
                                            y=meta_y_train,
                                            n_repeats=self.n_repeats,
                                            n_jobs=-1,
                                            random_state=self.EI.random_state,
                                            scoring=scorer_)
            
                importances_mean = pi.importances_mean
                importances_std = pi.importances_std

            pi_df = pd.DataFrame({"local_importance_mean": importances_mean, 
                                  "local_importance_std": importances_std, 
                                  "base predictor": [column_name[1] for column_name in meta_X_train.columns],
                                  "modality": [column_name[0] for column_name in meta_X_train.columns]
                                  })
            
            pi_df['ensemble_method'] = model_name
            pi_df['LMR'] = pi_df["local_importance_mean"].rank(pct=True, ascending=False)
            lm_pi_list.append(pi_df)
        self.LMR = pd.concat(lm_pi_list)

        print("complete!")

    def shap_val_mean(self, m, x):
        if hasattr(m, "predict_proba"):
            shap_exp = shap.Explainer(m.predict_proba, x)
        else:
            shap_exp = shap.Explainer(m.predict, x)
        
        shap_vals = shap_exp(x)

        return np.mean(shap_vals, axis=1)

    def rank_product_score(self, X_dict, y):
    
        print("\nInterpreting ensembles...")

        if self.ensemble_methods == "all":
            ensemble_methods = self.EI.meta_models.keys()
        else:
            ensemble_methods = self.ensemble_methods

        self.local_feature_rank(X_dict, y)

        self.local_model_rank(ensemble_model_keys=ensemble_methods)

        """Calculate the Rank percentile & their products here"""
        # return feature_ranking
        feature_ranking_list = {}
        self.merged_lmr_lfr = {}
        for model_name in ensemble_methods:
            lmr_interest = self.LMR[self.LMR['ensemble_method']==model_name].copy()
            self.merged_lmr_lfr[model_name] = pd.merge(lmr_interest, self.LFR,  
                                        how='right', left_on=['base predictor', 'modality'], 
                                        right_on = ['base predictor', 'modality'])

            self.merged_lmr_lfr[model_name]['LMR_LFR_product'] = self.merged_lmr_lfr[model_name]['LMR']*self.merged_lmr_lfr[model_name]['LFR']
            """ take mean of LMR*LFR for each feature """
            RPS_list = {'modality':[],
                        'feature': [],
                        'RPS': []}

            for modal in self.merged_lmr_lfr[model_name]['modality'].unique():
                merged_lmr_lfr_modal = self.merged_lmr_lfr[model_name].loc[self.merged_lmr_lfr[model_name]['modality']==modal]
                for feat in merged_lmr_lfr_modal['local_feature_id'].unique():
                    RPS_list['modality'].append(modal)
                    RPS_list['feature'].append(feat)
                    RPS_list['RPS'].append(merged_lmr_lfr_modal.loc[merged_lmr_lfr_modal['local_feature_id']==feat, 
                                            'LMR_LFR_product'].mean())
            RPS_df = pd.DataFrame(RPS_list)
            RPS_df['feature rank'] = RPS_df['RPS'].rank(ascending=True)
            RPS_df['ensemble method'] = model_name
            RPS_df.sort_values(by='feature rank',inplace=True)
            feature_ranking_list[model_name] = RPS_df
        self.ensemble_feature_ranking = feature_ranking_list
        print('... complete!')