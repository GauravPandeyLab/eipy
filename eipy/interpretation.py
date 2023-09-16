from sklearn.inspection import permutation_importance
from eipy.utils import retrieve_X_y, bar_format, format_input_datatype
import pandas as pd
from tqdm import tqdm
import numpy as np
import copy
from sklearn.metrics import make_scorer
import pickle
from itertools import groupby
from operator import itemgetter
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import LabelEncoder

import warnings

warnings.filterwarnings("ignore")


class PermutationInterpreter:
    """
    Permuation importance based interpreter.

    This method utilizes sklearn's `permutation_importance
    <https://scikit-learn.org/stable/modules/generated/sklearn.inspection.permutation_importance.html>`_
    function.

    EI : EnsembleIntegration class object
        Fitted EnsembleIntegration model, i.e. with model_building=True.
    metric : function
        sklearn-like metric function.
    n_repeats : int, default=10
        Number of repeats in PermutationImportance.
    meta_predictor_keys: default='all'
        Meta predictor keys used in EnsembleIntegration. If 'all' then all meta predictors
        seen by EI are interpreted. Recommended to pass a subset of meta_predctor keys as
        a list.
    metric_greater_is_better: default=True
        Metric greater is better.

    Attributes
    ----------
    ensemble_feature_ranking : pandas.DataFrame
        Feature rankings for each ensemble method.
    LFR : pandas.DataFrame
        Local feature rankings for each base predictor.
    LMR : pandas.Dataframe
        self.LMR = None

    Returns
    -------
    self
        Feature rankings of final ensemble models trained with EnsembleIntegration.

    """

    def __init__(
        self,
        EI,
        metric,
        n_repeats=10,
        meta_predictor_keys="all",
        metric_greater_is_better=True,  # can be "all" or a list of keys for ensemble methods
    ):
        self.EI = EI
        self.metric = metric
        self.n_repeats = n_repeats
        self.meta_predictor_keys = meta_predictor_keys
        self.metric_greater_is_better = metric_greater_is_better

        self.LFR = None
        self.LMR = None

    def rank_product_score(self, X_dict, y):
        """
        Compute feature ranking of ensemble methods using LFR and LMR.

        Parameters
        ----------
        X_dict : dict
            Dictionary of X modalities. Keys and n_features
            must match those seen by EnsembleIntegration.train_base().
        y : array of shape (n_samples,)
            Target vector relative to X.

        Returns
        -------
        self
            Feature ranking of ensemble methods
        """

        print("Interpreting ensembles...\n")

        if self.meta_predictor_keys == "all":
            meta_predictor_keys = self.EI.meta_predictors.keys()
        else:
            meta_predictor_keys = self.meta_predictor_keys

        if self.LFR is None:
            self.local_feature_rank(X_dict, y)

        if self.LMR is None:
            self.local_model_rank(meta_predictor_keys=meta_predictor_keys)

        print("Calculating combined rank product score...")

        feature_ranking_list = {}
        self.merged_lmr_lfr = {}
        for model_name in meta_predictor_keys:
            lmr_interest = self.LMR[self.LMR["ensemble_method"] == model_name].copy()
            self.merged_lmr_lfr[model_name] = pd.merge(
                lmr_interest,
                self.LFR,
                how="right",
                left_on=["base predictor", "modality"],
                right_on=["base predictor", "modality"],
            )

            self.merged_lmr_lfr[model_name]["LMR_LFR_product"] = (
                self.merged_lmr_lfr[model_name]["LMR"]
                * self.merged_lmr_lfr[model_name]["LFR"]
            )
            # take mean of LMR*LFR for each feature
            RPS_list = {"modality": [], "feature": [], "RPS": []}

            for modal in self.merged_lmr_lfr[model_name]["modality"].unique():
                merged_lmr_lfr_modal = self.merged_lmr_lfr[model_name].loc[
                    self.merged_lmr_lfr[model_name]["modality"] == modal
                ]
                for feat in merged_lmr_lfr_modal["local_feature_id"].unique():
                    RPS_list["modality"].append(modal)
                    RPS_list["feature"].append(feat)
                    RPS_list["RPS"].append(
                        merged_lmr_lfr_modal.loc[
                            merged_lmr_lfr_modal["local_feature_id"] == feat,
                            "LMR_LFR_product",
                        ].mean()
                    )
            RPS_df = pd.DataFrame(RPS_list)
            RPS_df["feature rank"] = RPS_df["RPS"].rank(ascending=True)
            RPS_df["ensemble method"] = model_name
            RPS_df.sort_values(by="feature rank", inplace=True)
            feature_ranking_list[model_name] = RPS_df
        self.ensemble_feature_ranking = feature_ranking_list
        print("... complete!")

        return self

    def local_feature_rank(self, X_dict, y):
        """
        Local Feature Ranks (LFRs) for each base predictor

        Parameters
        ----------
        X_dict : dict
            Dictionary of X modalities. Keys and n_features
            must match those seen by EnsembleIntegration.train_base().
        y : array of shape (n_samples,)
            Target vector relative to X.

        Returns
        -------
        self
            Local feature ranks.
        """

        importance_list = []

        for modality_name in tqdm(
            self.EI.modality_names,
            desc="Calculating local feature ranks",
            bar_format=bar_format,
        ):
            # TODO: handle dataframe here
            X = X_dict[modality_name]
            X_np, feature_names = format_input_datatype(X, modality_name=modality_name)

            base_models = copy.deepcopy(
                self.EI.final_models["base models"][modality_name]
            )

            base_models = sorted(base_models, key=itemgetter("model name"))

            for _key, base_models_per_sample in groupby(
                base_models, key=itemgetter("model name")
            ):
                list_of_base_models = []

                for base_model_dict in base_models_per_sample:
                    base_model = pickle.loads(base_model_dict["pickled model"])
                    list_of_base_models.append(
                        (
                            str(base_model_dict["sample id"]),
                            base_model,
                        )
                    )  # list of tuples for VotingClassifier

                if (
                    len(list_of_base_models) > 1
                ):  # take mean of base predictors with different sample ids
                    ###################################################################
                    #  This code is a work around and may be fragile. We use VotingClassifier
                    # to combine models trained on different samples (taking a mean of model
                    # output). The current sklearn implementation of VotingClassifier does not
                    # accept pretrained models, so we set parameters ourselves to allow it. In
                    # the future it may be possible to use VotingClassifier alone without
                    # additional code. An sklearn-like model is needed to be passed to
                    # permutation_importance.

                    model = VotingClassifier(
                        estimators=list_of_base_models,
                        voting="soft",
                        weights=np.ones(len(list_of_base_models)),
                    )  # average predictions of models built on different data samples

                    model.estimators_ = [j for _, j in list_of_base_models]
                    model.le_ = LabelEncoder().fit(y)
                    model.classes_ = model.le_.classes_

                    ##################################################################

                else:
                    model = list_of_base_models[0][1]

                needs_proba = hasattr(base_model, "predict_proba")
                scorer_ = make_scorer(
                    self.metric,
                    greater_is_better=self.metric_greater_is_better,
                    needs_proba=needs_proba,
                )

                pi = permutation_importance(
                    estimator=model,
                    X=X_np,
                    y=y,
                    n_repeats=self.n_repeats,
                    n_jobs=self.EI.n_jobs,
                    random_state=self.EI.random_state,
                    scoring=scorer_,
                )

                pi_df = pd.DataFrame(
                    {
                        "local_importance_mean": pi.importances_mean,
                        "local_importance_std": pi.importances_std,
                        "local_feature_id": self.EI.feature_names_dict[modality_name],
                    }
                )

                pi_df["base predictor"] = base_model_dict["model name"]
                pi_df["modality"] = modality_name
                pi_df["LFR"] = pi_df["local_importance_mean"].rank(
                    pct=True, ascending=False
                )
                importance_list.append(pi_df)

        self.LFR = pd.concat(importance_list)

        return self

    def local_model_rank(self, meta_predictor_keys):
        """
        Local Model Ranks (LMRs)

        Parameters
        ----------
        meta_predictor_keys : list of str
            List of meta predictor keys that will be used to select
            ensembles classifiers to interpret.

        Returns
        -------
        self
            Local model ranks.
        """
        #  load meta training data from EI training

        meta_X_train, meta_y_train = retrieve_X_y(
            labelled_data=self.EI.meta_training_data_final[0]
        )

        if self.EI.sampling_aggregation == "mean":
            meta_X_train = meta_X_train.groupby(level=[0, 1], axis=1).mean()

        #  calculate importance for ensemble models of interest

        lm_pi_list = []

        ensemble_models = copy.deepcopy(self.EI.final_models["meta models"])

        ensemble_models = [ensemble_models[key] for key in meta_predictor_keys]

        ensemble_models = dict(zip(meta_predictor_keys, ensemble_models))

        for model_name, model in tqdm(
            ensemble_models.items(),
            desc="Calculating local model ranks",
            bar_format=bar_format,
        ):
            meta_predictor = pickle.loads(model)

            if ("Mean" in model_name) or ("Median" in model_name):
                importances_mean = np.ones(len(meta_X_train.columns))
                importances_std = np.zeros(len(meta_X_train.columns))

            elif model_name == "CES":
                model_selected_freq = []
                for bp in meta_X_train.columns:
                    model_selected_freq.append(
                        meta_predictor.selected_ensemble.count(bp)
                    )
                importances_mean = model_selected_freq
                importances_std = np.ones(len(meta_X_train.columns)) * np.nan

            else:
                needs_proba = hasattr(model, "predict_proba")
                scorer_ = make_scorer(
                    self.metric,
                    greater_is_better=self.metric_greater_is_better,
                    needs_proba=needs_proba,
                )
                pi = permutation_importance(
                    estimator=meta_predictor,
                    X=meta_X_train,
                    y=meta_y_train,
                    n_repeats=self.n_repeats,
                    n_jobs=-1,
                    random_state=self.EI.random_state,
                    scoring=scorer_,
                )

                importances_mean = pi.importances_mean
                importances_std = pi.importances_std

            pi_df = pd.DataFrame(
                {
                    "local_importance_mean": importances_mean,
                    "local_importance_std": importances_std,
                    "base predictor": [
                        column_name[1] for column_name in meta_X_train.columns
                    ],
                    "modality": [
                        column_name[0] for column_name in meta_X_train.columns
                    ],
                }
            )

            pi_df["ensemble_method"] = model_name
            pi_df["LMR"] = pi_df["local_importance_mean"].rank(
                pct=True, ascending=False
            )
            lm_pi_list.append(pi_df)
        self.LMR = pd.concat(lm_pi_list)

        return self
