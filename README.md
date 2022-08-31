# Ensemble Integration (EI): Integrating multimodal data through interpretable heterogeneous ensembles
Ensemble Integration (EI) is a customizable pipeline for generating diverse ensembles of heterogeneous classifiers, as well as the accompanying metadata needed for ensemble learning approaches utilizing ensemble diversity for improved performance. It also fairly evaluates the performance of several ensemble learning methods including ensemble selection [Caruana2004], and stacked generalization (stacking) [Wolpert1992]. Though other tools exist, we are unaware of a similar modular, scalable pipeline designed for large-scale ensemble learning. This fully python version of EI was implemented by Jamie J. R. Bennett and Yan Chak Li, and is based on the original version: https://github.com/GauravPandeyLab/ensemble_integration.git.

EI is designed for generating extremely large ensembles (taking days or weeks to generate) and thus consists of an initial data generation phase tuned for multicore and distributed computing environments. The output is a set of compressed CSV files containing the class distribution produced by each classifier that serves as input to a later ensemble learning phase.

More details of EI can be found in our Biorxiv preprint:

Full citation:

Yan Chak Li, Linhua Wang, Jeffrey Law, T. M. Murali, Gaurav Pandey (2020): Integrating multimodal data through interpretable heterogeneous ensembles, bioRxiv. Preprint. 2020.05.29.123497; doi: https://doi.org/10.1101/2020.05.29.123497

This repository is protected by CC-BY-NC-ND-4.0.

## Requirements ##

The following packages are requirements for EI:

```
python==3.9.12
scikit-learn==1.1.1
pandas==1.4.3
numpy==1.22.3
joblib==1.1.0
```

## Demo ##

Import scikit-learn classifiers and EI.

```
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

path_to_ei = "/home/opc/ei-python/"
sys.path.append(path_to_ei)
from ei import EnsembleIntegration, read_arff_to_pandas_df
```

Generate a toy multimodal dataset.

```
from sklearn.dataset import make_classification

X, y = make_classification(n_samples=1000, n_features=20, n_redundant=0,
n_clusters_per_class=1, weights=[0.7], flip_y=0, random_state=1)

X_view_0 = X[:, :5]
X_view_1 = X[:, 5:10]
X_view_2 = X[:, 10:15]
X_view_3 = X[:, 15:]
  
```

Define base predictors as a dictionary,

```
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
```

Set up a 5-fold outer cross validation, along with a 5-fold inner cross validation for meta-training data generation.

```
EI = EnsembleIntegration(base_predictors=base_predictors,
                         k_outer=5,
                         k_inner=5,
                         n_bags=1,
                         bagging_strategy="mean",
                         n_jobs=-1,
                         random_state=42,
                         project_name="demo")
```

Generate meta-training data and train base classifiers on outer folds.

```
modalities = ["view_0", "view_1", "view_2", "view_3"]

for modality in modalities:
    X, y = read_data(f"data/{modality}/data.arff")
    EI.train_base(X, y, base_predictors, modality=modality)

EI.save() # save EI as EI.demo
```

Define stacking classifiers and test ensembles.

```
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

EI = EnsembleIntegration().load("EI.demo") # load models from disk

EI.train_meta(meta_models=meta_models) # train meta classifiers
```


