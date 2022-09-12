from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import RobustScaler
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
import sys

path_to_ei = "/home/jamie/Projects/ei-python"
sys.path.append(path_to_ei)
from ei import EnsembleIntegration

from sklearn.datasets import make_classification

X, y = make_classification(n_samples=1000, n_features=20, n_redundant=0,
n_clusters_per_class=1, weights=[0.7], flip_y=0, random_state=1)

X_view_0 = X[:, :5]
X_view_1 = X[:, 5:10]
X_view_2 = X[:, 10:15]
X_view_3 = X[:, 15:]

base_predictors = {
    'AdaBoost': AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=3)),
    'DT': DecisionTreeClassifier(max_depth=3),
    'GradientBoosting': GradientBoostingClassifier(),
    'KNN': KNeighborsClassifier(n_neighbors=21),
    'LR': LogisticRegression(),
    'NB': GaussianNB(),
    'MLP': MLPClassifier(),
    'RF': RandomForestClassifier(),
    'SVM': LinearSVC(),
    'XGB': XGBClassifier(use_label_encoder=False, eval_metric='error')
}

EI = EnsembleIntegration(base_predictors=base_predictors,
                         k_outer=5,
                         k_inner=5,
                         n_bags=1,
                         bagging_strategy="mean",
                         n_jobs=-1, # set as -1 to use all available CPUs
                         random_state=42,
                         project_name="demo")

modalities = {"view_0": X_view_0,
              "view_1": X_view_1,
              "view_2": X_view_2,
              "view_3": X_view_3,}

for name, modality in modalities.items():
    EI.train_base(modality, y, base_predictors, modality=name)

EI.save() # save EI as EI.demo

meta_models = {
    "AdaBoost": AdaBoostClassifier(),
    "DT": DecisionTreeClassifier(max_depth=5),
    "GradientBoosting": GradientBoostingClassifier(),
    "KNN": KNeighborsClassifier(n_neighbors=21),
    "LR": LogisticRegression(),
    "NB": GaussianNB(),
    "MLP": MLPClassifier(),
    "RF": RandomForestClassifier(),
    "SVM": LinearSVC(tol=1e-2, max_iter=10000),
    "XGB": XGBClassifier(use_label_encoder=False, eval_metric='error')
}

EI = EnsembleIntegration().load("EI.demo")  # load models from disk

EI.train_meta(meta_models=meta_models)  # train meta classifiers
