"""
Ensemble Integration

@author: Jamie Bennett, Yan Chak (Richard) Li
"""

import pandas as pd
import numpy as np
from copy import copy
import dill as pickle
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import StratifiedKFold
from joblib import Parallel, delayed
from joblib.externals.loky import set_loky_pickler
import warnings
from utils import scores, set_seed, random_integers, sample, retrieve_X_y, append_modality, metric_threshold_dataframes, create_base_summary, safe_predict_proba
from ens_selection import CES
warnings.filterwarnings("ignore", category=DeprecationWarning)

def remove_correlated_features(df_train, df_test, correlation_removal_threshold=0.95):

    df = pd.concat([df_train, df_test], axis=0)
    df = df.drop("labels", level=0, axis=1)
    # Create correlation matrix
    corr_matrix = df.corr().abs()

    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

    # Find features with correlation greater than 0.95
    to_drop = [column for column in upper.columns if any(upper[column] > correlation_removal_threshold)]

    # Drop features 
    df_train_new = df_train.drop(to_drop, axis=1)
    df_test_new = df_test.drop(to_drop, axis=1)

    return df_train_new, df_test_new


class MeanAggregation:
    def __init__(self):
        pass

    def fit(self, X, y):
        pass

    def predict_proba(self, X):
        predict_positive = X.mean(axis=1)
        return np.transpose(np.array([1 - predict_positive, predict_positive]))


class MedianAggregation:
    def __init__(self):
        pass

    def fit(self, X, y):
        pass

    def predict_proba(self, X):
        predict_positive = X.median(axis=1)
        return np.transpose(np.array([1 - predict_positive, predict_positive]))


class EnsembleIntegration:
    """
    Algorithms to test a variety of ensemble methods.

    Parameters
    ----------
    base_predictors : dictionary
        Base predictors.
    k_outer : int, optional
        Number of outer folds. Default is 5.
    k_inner : int, optional
        Number of inner folds. Default is 5.
    random_state : int, optional
        Random state for cross-validation. The default is 42.

    Returns
    -------
    predictions_df : Pandas dataframe of shape (n_samples, n_base_predictors)
        Matrix of data intended for training of a meta-algorithm.

    To be done:
        - EI.save() does not work with TF models in base predictors. Need to save models separately then set base_predictors=None to save. Load models from separate files
        - create wrapper for TF models. Needs to take TF model + fit parameters. Then create new fit function.
        - CES ensemble
        - interpretation
        - best base predictor
        - model building
        - think about the use of calibrated classifier in base and meta
    """

    def __init__(self,
                 base_predictors=None,  # dictionary of sklearn models
                 meta_models=None,  # dictionary of sklearn models
                 k_outer=5,  # number of outer folds, int
                 k_inner=5,  # number of inner folds, int
                 n_samples=1,  # number of samples of training sets to be taken 
                 sampling_strategy="undersampling",  # sampling method: "undersampling", "oversampling", "hybrid", None
                 sampling_aggregation="mean",
                 n_jobs=-1,
                 random_state=None,
                 parallel_backend="loky",  # change to "threading" if including TensorFlow models in base_predictors
                 project_name="project",
                 additional_ensemble_methods=["Mean", "Median", "CES"],
                 correlation_removal_threshold=1):  # remove features in meta_training_data and meta_test_data with correlation > correlation_removal_threshold
                                                    # default is 1 (no removal). Typical value would be 0.95. NEEDS REMOVING BEFORE PUSH

        set_seed(random_state)

        self.base_predictors = base_predictors
        if meta_models is not None:
            self.meta_models = {"S." + k: v for k, v in meta_models.items()}  # suffix denotes stacking
        self.k_outer = k_outer
        self.k_inner = k_inner
        self.n_samples = n_samples
        self.sampling_strategy = sampling_strategy
        self.sampling_aggregation = sampling_aggregation
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.parallel_backend = parallel_backend
        self.project_name = project_name
        self.additional_ensemble_methods = additional_ensemble_methods
        self.correlation_removal_threshold = correlation_removal_threshold

        self.trained_meta_models = {}
        self.trained_base_predictors = {}

        self.cv_outer = StratifiedKFold(n_splits=self.k_outer, shuffle=True,
                                            random_state=self.random_state)

        self.cv_inner = StratifiedKFold(n_splits=self.k_inner, shuffle=True,
                                            random_state=self.random_state)

        self.random_numbers_for_samples = random_integers(n_integers=n_samples, 
                                                              seed=self.random_state)

        self.meta_training_data = None
        self.meta_test_data = None
        self.base_summary = None

        self.meta_predictions = None
        self.meta_summary = None

    @ignore_warnings(category=ConvergenceWarning)
    def train_meta(self, meta_models=None, display_metrics=True):

        if meta_models is not None:
            self.meta_models = meta_models
            self.meta_models = {"S." + k: v for k, v in meta_models.items()}  # suffix denotes stacking
        
        for k, v in self.meta_models.items():
            if hasattr(v, 'random_state'):
                v.set_params(**{'random_state': self.random_state})

        additional_meta_models = {"Mean": MeanAggregation(),
                                  "Median": MedianAggregation(),
                                  "CES": CES()}

        additional_meta_models = dict((k, additional_meta_models[k]) for k in self.additional_ensemble_methods)

        self.meta_models = {**additional_meta_models, **self.meta_models}

        print("\nWorking on meta models")

        y_test_combined = []

        for fold_id in range(self.k_outer):
            _, y_test = retrieve_X_y(labelled_data=self.meta_test_data[fold_id])
            y_test_combined.extend(y_test)

        meta_predictions = {}
        performance_metrics = []

        for model_name, model in self.meta_models.items():
            print("\n{model_name:}...".format(model_name=model_name))

            y_pred_combined = []

            for fold_id in range(self.k_outer):
                meta_training_data, meta_test_data = remove_correlated_features(self.meta_training_data[fold_id], 
                                                                                self.meta_test_data[fold_id], 
                                                                                self.correlation_removal_threshold)

                X_train, y_train = retrieve_X_y(labelled_data=meta_training_data)
                X_test, _ = retrieve_X_y(labelled_data=meta_test_data)

                if self.sampling_aggregation == "mean":
                    X_train = X_train.groupby(level=[0, 1], axis=1).mean()
                    X_test = X_test.groupby(level=[0, 1], axis=1).mean()

                model.fit(X_train, y_train)
                y_pred = safe_predict_proba(model, X_test)
                y_pred_combined.extend(y_pred)

            meta_predictions[model_name] = y_pred_combined
            performance_metrics.append(scores(y_test_combined, y_pred_combined, display=display_metrics))

        meta_predictions["labels"] = y_test_combined

        self.meta_predictions = pd.DataFrame.from_dict(meta_predictions)
        self.meta_summary = metric_threshold_dataframes(self.meta_predictions)

        return self

    @ignore_warnings(category=ConvergenceWarning)
    def train_base(self, X, y, base_predictors=None, modality=None):

        if base_predictors is not None:
            self.base_predictors = base_predictors  # update base predictors

        for k, v in self.base_predictors.items():
            if hasattr(v, 'random_state') and hasattr(v, 'set_params'):
                v.set_params(**{'random_state': self.random_state})

        self.train_base_inner(X, y, self.base_predictors, modality)
        self.train_base_outer(X, y, self.base_predictors, modality)

        return self

    def train_base_inner(self, X, y, base_predictors=None, modality=None):
        """
        Perform a round of (inner) k-fold cross validation on each outer
        training set for generation of training data for the meta-algorithm

        Parameters
        ----------
        X : array of shape (n_samples, n_features)
            Dataset.
        y : array of shape (n_samples,)
            Labels.

        Returns
        -------
        meta_training_data : List of length k_outer containing Pandas dataframes
        of shape (n_outer_training_samples, n_base_predictors * n_samples)
        """
        if base_predictors is not None:
            self.base_predictors = base_predictors  # update base predictors

        if modality is not None:
            print(f"\n{modality} modality: training base predictors on inner training sets...")
        else:
            print("Training base predictors on inner training sets...")

        # dictionaries for meta train/test data for each outer fold
        meta_training_data = []
        # define joblib Parallel function
        parallel = Parallel(n_jobs=self.n_jobs, verbose=10, backend=self.parallel_backend)
        for outer_fold_id, (train_index_outer, test_index_outer) in enumerate(self.cv_outer.split(X, y)):
            print("\nGenerating meta-training data for outer fold {outer_fold_id:}...".format(
                outer_fold_id=outer_fold_id))

            X_train_outer = X[train_index_outer]
            y_train_outer = y[train_index_outer]

            # spawn n_jobs jobs for each sample, inner_fold and model
            output = parallel(delayed(self.train_model_fold_sample)(X=X_train_outer,
                                                                    y=y_train_outer,
                                                                    model_params=model_params,
                                                                    fold_params=inner_fold_params,
                                                                    sample_state=sample_state)
                              for model_params in self.base_predictors.items()
                              for inner_fold_params in enumerate(self.cv_inner.split(X_train_outer, y_train_outer))
                              for sample_state in enumerate(self.random_numbers_for_samples))
            combined_predictions = self.combine_data_inner(output, modality)
            meta_training_data.append(combined_predictions)

        self.meta_training_data = append_modality(self.meta_training_data, meta_training_data)

        return self

    def train_base_outer(self, X, y, base_predictors=None, modality=None):
        """
        Train each base predictor on each outer training set

        Parameters
        ----------
        X : array of shape (n_samples, n_features)
            Dataset.
        y : array of shape (n_samples,)
            Labels.

        Returns
        -------

        meta_test_data : List of length k_outer containing Pandas dataframes
        of shape (n_outer_test_samples, n_base_predictors * n_samples)
        """
        if base_predictors is not None:
            self.base_predictors = base_predictors  # update base predictors

        if modality is not None:
            print(f"\n{modality} modality: training base predictors on outer training sets...")
        else:
            print("Training base predictors on outer training sets...")

        # define joblib Parallel function
        parallel = Parallel(n_jobs=self.n_jobs, verbose=10, backend=self.parallel_backend)
        # spawn job for each sample, outer_fold and model
        output = parallel(delayed(self.train_model_fold_sample)(X=X,
                                                                y=y,
                                                                model_params=model_params,
                                                                fold_params=outer_fold_params,
                                                                sample_state=sample_state)
                          for model_params in self.base_predictors.items()
                          for outer_fold_params in enumerate(self.cv_outer.split(X, y))
                          for sample_state in enumerate(self.random_numbers_for_samples))
        # meta_test_data = self.combine_data_outer(output, modality)

        self.meta_test_data = append_modality(self.meta_test_data, self.combine_data_outer(output, modality))
        self.base_summary = create_base_summary(self.meta_test_data)

        return self

    @ignore_warnings(category=ConvergenceWarning)
    def train_model_fold_sample(self, X, y, model_params, fold_params, sample_state):
        model_name, model = model_params
        # model.set_params(**{'random_state': self.random_state})

        fold_id, (train_index, test_index) = fold_params
        sample_id, sample_random_state = sample_state

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        X_sample, y_sample = sample(X_train, y_train, strategy=self.sampling_strategy, random_state=sample_random_state)

        model.fit(X_sample, y_sample)

        y_pred = safe_predict_proba(model, X_test)

        # if str(model.__class__).find("sklearn") != -1: # was using for TensorFlow models
        #     y_pred = y_pred[:, 1]  # assumes other models are sklearn

        metrics = scores(y_test, y_pred)

        results_dict = {"model_name": model_name,
                        "sample_id": sample_id,
                        "fold_id": fold_id,
                        "scores": metrics,
                        "model": model,
                        "y_pred": y_pred,
                        "labels": y_test}

        return results_dict

    def combine_data_inner(self, list_of_dicts, modality):  # we don't save the models trained here
        # dictionary to store predictions
        combined_predictions = {}
        # combine fold predictions for each model
        for model_name in self.base_predictors.keys():
            for sample_id in range(self.n_samples):
                model_predictions = np.concatenate(
                    list(d["y_pred"] for d in list_of_dicts if
                         d["model_name"] == model_name and d["sample_id"] == sample_id))
                combined_predictions[modality, model_name, sample_id] = model_predictions
        labels = np.concatenate(list(d["labels"] for d in list_of_dicts if
                                     d["model_name"] == list(self.base_predictors.keys())[0] and d["sample_id"] == 0))
        combined_predictions = pd.DataFrame(combined_predictions).rename_axis(["modality", "base predictor", "sample"],
                                                                              axis=1)
        combined_predictions["labels"] = labels
        return combined_predictions

    def combine_data_outer(self, list_of_dicts, modality):
        combined_predictions = []
        for fold_id in range(self.k_outer):
            predictions = {}
            for model_name in self.base_predictors.keys():
                for sample_id in range(self.n_samples):
                    model_predictions = list([d["y_pred"], d["model"]] for d in list_of_dicts if
                                             d["fold_id"] == fold_id and d["model_name"] == model_name and d[
                                                 "sample_id"] == sample_id)
                    predictions[modality, model_name, sample_id] = model_predictions[0][0]
                    # self.trained_base_predictors[(model_name, sample_id)] = model_predictions[0][1] # need to write to file to avoid memory issues
            labels = [d["labels"] for d in list_of_dicts if
                      d["fold_id"] == fold_id and d["model_name"] == list(self.base_predictors.keys())[0] and d[
                          "sample_id"] == 0]
            predictions = pd.DataFrame(predictions)
            predictions["labels"] = labels[0]
            combined_predictions.append(predictions.rename_axis(["modality", "base predictor", "sample"], axis=1))
        return combined_predictions

    def save(self, path=None):
        if path is None:
            path = f"EI.{self.project_name}"
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        print(f"\nSaved to {path}\n")

    @classmethod
    def load(cls, filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)
