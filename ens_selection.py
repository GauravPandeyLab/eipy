from utils import scores, set_seed, \
    random_integers, sample, \
    retrieve_X_y, update_keys, append_modality

from numpy.random import choice, seed
from numpy import argmax, argmin, argsort, corrcoef, mean, nanmax, sqrt, triu_indices_from, where
import pandas as pd

class CES:
    """
    Caruana et al's Ensemble Selection
    """
    def __init__(self,
                 scoring_func,
                 max_ensemble_size=50,
                 random_state=0,
                 greater_is_better=True
                 ):
        set_seed(random_state)
        self.seed = random_state
        self.scoring_func = scoring_func
        self.max_ensemble_size = max_ensemble_size
        self.selected_ensemble = []
        self.train_performance = []
        self.greater_is_better = greater_is_better
        self.argbest = argmax if greater_is_better else argmin
        self.best = max if greater_is_better else min

    def fit(self, X, y):
        self.selected_ensemble = []
        self.train_performance = []
        best_classifiers = X.apply(lambda x: self.scoring_func(y, x).sort_values(ascending=self.greater_is_better))
        for i in range(min(self.max_ensemble_size, len(best_classifiers))):
            best_candidate = self.select_candidate_enhanced(X, y, best_classifiers, self.selected_ensemble, i)
            self.selected_ensemble.append(best_candidate)
            self.train_performance.append(self.get_performance(X, y))

        train_performance_df = pd.DataFrame.from_records(self.train_performance)
        best_ensemble_size = self.get_best_performer(train_performance_df)['ensemble_size'].values
        self.best_ensemble = train_performance_df['ensemble'][:best_ensemble_size.item(0) + 1]


    def predict_proba(self, X):
        ces_bp_df = X[self.best_ensemble]
        return ces_bp_df.mean(axis=1).values

    def select_candidate_enhanced(self, X, y, best_classifiers, ensemble, i):
        initial_ensemble_size = 2
        max_candidates = 50
        if len(ensemble) >= initial_ensemble_size:
            candidates = choice(best_classifiers.index.values, min(max_candidates, len(best_classifiers)),
                                replace=False)
            candidate_scores = [self.scoring_func(y, X[ensemble + [candidate]].mean(axis=1)) for
                                candidate in
                                candidates]
            best_candidate = candidates[self.argbest(candidate_scores)]
        else:
            best_candidate = best_classifiers.index.values[i]
        return best_candidate

    def get_performance(self, X, y):

        predictions = X[self.selected_ensemble].mean(axis=1)
        score = self.scoring_func(y, predictions)

        return {'seed': self.seed, 'score': score,
                'ensemble': self.selected_ensemble[-1],
                'ensemble_size': len(self.selected_ensemble)}

    def get_best_performer(self, df, one_se=False):
        if not one_se:
            return df[df.score == self.best(df.score)].head(1)
        se = df.score.std() / sqrt(df.shape[0] - 1)
        if self.greater_is_better:
            return df[df.score >= (self.best(df.score) - se)].head(1)
        return df[df.score <= (self.best(df.score) + se)].head(1)

# class BestBasePredictor:
#     """
#     Picking the best base predictor
#     """
#     def __init__(self, scoring_func, greater_is_better=True):
#         self.scoring_func = scoring_func
#         self.greater_is_better = greater_is_better
#         self.argbest = argmax if greater_is_better else argmin
#         self.best = max if greater_is_better else min
#
#     def fit(self, X, y):
#

class MeanAggregation:
    def __init__(self):
        pass

    def fit(self, X, y):
        pass

    def predict_proba(self, X):
        return X.mean(axis=1)

