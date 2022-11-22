from sklearn.cluster import k_means
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedKFold
from utils import scores, set_seed, random_integers, sample, \
    retrieve_X_y, append_modality, create_base_summary
from joblib import Parallel, delayed
import pandas as pd
import numpy as np
from ei import MedianAggregation, MeanAggregation
from ens_selection import CES
import copy
import sklearn.metrics
from sklearn.metrics import fbeta_score, make_scorer
from utils import safe_predict_proba

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
                 k_outer=5,
                 k_inner=5,
                 feature_dict=None,
                 n_jobs=-1,
                 n_repeats=10,
                 random_state=42,
                 ensemble_of_interest='ALL'):
        self.EI = copy.copy(EI_object)
        self.k_outer = self.EI.k_outer
        self.base_predictors = copy.copy(base_predictors)
        self.meta_models = copy.copy(meta_models)
        
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
        self.k_outer = k_outer
        self.k_inner = k_inner

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
        
        """For the ensemble"""
        self.EI.train_base_inner(X, self.y, self.EI.base_predictors, modality)
        self.importance_outer_base(X, self.y, self.EI.base_predictors, modality)
        

    def local_model_rank(self, meta_models_interested):
        X_train_list = []
        y_train_list = []
        lm_pi_list = []
        # print(self.EI.meta_training_data)
        for model_name, model in meta_models_interested.items():
            for fold_id in range(self.EI.k_outer):
                X_train, y_train = retrieve_X_y(labelled_data=self.EI.meta_training_data[fold_id])
                X_test, y_test = retrieve_X_y(labelled_data=self.EI.meta_test_data[fold_id])
                # X_train_list.append(X_train)
                # y_train_list.append(y_train)
                if (model_name == "Mean") or (model_name == "Median"):
                    lm_pi = np.ones((len(X_train.columns), self.n_repeats))
                    # print(model_name, X_train.columns)
                
                elif 'CES' == model_name:
                    model.fit(X_train, y_train)
                    """TODO"""
                    lm_pi = np.zeros((len(X_train.columns), self.n_repeats))
                    # model.best_ensemble
                else:
                    model.fit(X_train, y_train)
                    lm_pi_ = permutation_importance(estimator=model,
                                                    X=X_test,
                                                    y=y_test,
                                                    n_repeats=self.n_repeats,
                                                    n_jobs=-1,
                                                    random_state=self.random_state,
                                                    scoring=self.metric)
                    lm_pi = lm_pi_.importances

                # pi_df = pd.DataFrame({'local_model_PI': lm_pi,})
                pi_df = pd.DataFrame(data=lm_pi,
                            #  index=X_train.columns,
                             columns=range(lm_pi.shape[1])).reset_index()
                # pi_df.rename(columns={'index':'local_feat_name'}, inplace=True)
                pi_df['base predictor'] = [i[1] for i in X_train.columns]
                pi_df['fold'] = fold_id
                pi_df['modality'] = [i[0] for i in X_train.columns]
                pi_df['sample_id'] = [i[2] for i in X_train.columns]
                pi_df['ensemble_method'] = model_name
                pi_df = pd.melt(pi_df, id_vars=['base predictor', 'fold',
                                                'modality', 'sample_id', 'ensemble_method'],
                                    value_vars=range(lm_pi.shape[1]),
                                    var_name='iteration', value_name='local_model_PI')
                # print(pi_df.shape)

            pi_mean_df = pi_df.groupby(['modality', 
                        'base predictor', 
                        'ensemble_method'])['local_model_PI'].mean().reset_index()
            pi_mean_df['LMR'] = pi_mean_df['local_model_PI'].rank(pct=True, ascending=False)
            lm_pi_list.append(pi_mean_df)
        self.LMRs = pd.concat(lm_pi_list)
        # print(self.LMRs)

    def importance_outer_base(self, X, y, base_predictors=None, modality=None):
        if base_predictors is not None:
            self.EI.base_predictors = base_predictors  # update base predictors

        if modality is not None:
            print(f"\n{modality} modality: training base predictors on outer training sets...")
        else:
            print("Training base predictors on outer training sets...")

        # define joblib Parallel function
        parallel = Parallel(n_jobs=self.n_jobs, verbose=10, backend=self.EI.parallel_backend)
        # spawn job for each sample, outer_fold and model
        output = parallel(delayed(self.interpret_model_fold_sample)(X=X,
                                                                y=y,
                                                                model_params=model_params,
                                                                fold_params=outer_fold_params,
                                                                sample_state=sample_state,
                                                                modality=modality)
                          for model_params in self.EI.base_predictors.items()
                          for outer_fold_params in enumerate(self.EI.cv_outer.split(X, y))
                          for sample_state in enumerate(self.EI.random_numbers_for_samples))
        # meta_test_data = self.combine_data_outer(output, modality)
        pred, interpret = self.combine_data_outer(output, modality)
        self.EI.meta_test_data = append_modality(self.EI.meta_test_data, pred)
        self.EI.base_summary = create_base_summary(self.EI.meta_test_data)
        lfr_df_list = []
        for k, v in self.EI.base_predictors.items():
            interpret_k = interpret.loc[interpret['base predictor']==k]
            interpret_k = interpret_k.groupby(['modality', 
                                        'base predictor', 
                                        'local_feat_name'])['local_feat_PI'].mean().reset_index()
            interpret_k['LFR'] = interpret_k['local_feat_PI'].rank(pct=True, 
                                                                ascending=False)
            # """ rescale """
            # interpret_k['sigmoid_LFR'] = 1/(1+np.exp(-(interpret_k['LFR']-0.5)))
            lfr_df_list.append(interpret_k)

        self.LFRs.append(pd.concat(lfr_df_list))

        return self

    def interpret_model_fold_sample(self, X, y, model_params, 
                                    fold_params, sample_state, modality):
        model_name, model = model_params
        # model.set_params(**{'random_state': self.random_state})

        fold_id, (train_index, test_index) = fold_params
        sample_id, sample_random_state = sample_state

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        X_sample, y_sample = sample(X_train, y_train, strategy=self.EI.sampling_strategy, 
                                    random_state=sample_random_state)

        model.fit(X_sample, y_sample)

        y_pred = safe_predict_proba(model, X_test)

        lf_pi = permutation_importance(estimator=model,
                                                X=X_test,
                                                y=y_test,
                                                n_repeats=self.n_repeats,
                                                n_jobs=1,
                                                random_state=self.random_state,
                                                scoring=self.metric)

        # pi_df = pd.DataFrame({'local_feat_PI': lf_pi.importances_mean, 
                                # 'local_feat_name': self.feature_dict[modality]})        

        pi_df = pd.DataFrame(data=lf_pi.importances, 
                             index=self.feature_dict[modality],
                             columns=range(self.n_repeats)).reset_index()
        pi_df.rename(columns={'index':'local_feat_name'}, inplace=True)
        pi_df = pd.melt(pi_df, id_vars=['local_feat_name'],
                               value_vars=range(self.n_repeats),
                               var_name='iteration', value_name='local_feat_PI')
        pi_df['base predictor'] = model_name
        pi_df['fold'] = fold_id
        pi_df['sample_id'] = sample_id

        metrics = scores(y_test, y_pred)

        results_dict = {"model_name": model_name,
                        "sample_id": sample_id,
                        "fold_id": fold_id,
                        "scores": metrics,
                        "model": model,
                        "y_pred": y_pred,
                        "labels": y_test,
                        "lf_pi": pi_df}

        return results_dict
    
    def combine_data_outer(self, list_of_dicts, modality):
        combined_predictions = []

        for fold_id in range(self.EI.k_outer):
            predictions = {}
            # interpretations = []
            for model_name in self.EI.base_predictors.keys():
                for sample_id in range(self.EI.n_samples):
                    model_predictions = list([d["y_pred"], d["model"]] for d in list_of_dicts if
                                             d["fold_id"] == fold_id and d["model_name"] == model_name and d[
                                                 "sample_id"] == sample_id)
                    predictions[modality, model_name, sample_id] = model_predictions[0][0]
                    # combined_interpretations = combined_interpretations + d['pi_df']
                    # interpretations[modality, model_name, sample_id] = model_interpretation[0][0]
                    # self.trained_base_predictors[(model_name, sample_id)] = model_predictions[0][1] # need to write to file to avoid memory issues
            labels = [d["labels"] for d in list_of_dicts if
                      d["fold_id"] == fold_id and d["model_name"] == list(self.EI.base_predictors.keys())[0] and d[
                          "sample_id"] == 0]
            predictions = pd.DataFrame(predictions)
            predictions["labels"] = labels[0]
            combined_predictions.append(predictions.rename_axis(["modality", "base predictor", "sample"], axis=1))
        # TODO: add interpretation here:
        combined_interpretations = pd.concat([d['lf_pi'] for d in list_of_dicts])
        combined_interpretations['modality'] = modality
        
        return combined_predictions, combined_interpretations


    def rank_product_score(self):
        for modal_name, modality_data in self.modalities.items():
            self.local_feature_rank(modality_data, modality=modal_name)
        self.LFRs = pd.concat(self.LFRs)

        """Add mean/median aggregation here"""
        meta_models = {"S." + k: v for k, v in self.meta_models.items() if not (k in ["Mean", "Median"])}
        if self.ensemble_of_interest == "ALL":
            if not ("Mean" in meta_models.keys()):
                meta_models["Mean"] = MeanAggregation()
            elif not ("Median" in meta_models.keys()):
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
        return self
