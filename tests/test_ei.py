import pytest

@pytest.mark.parametrize(
    "sampling_strategy",
    [   
        (None),
        ("undersampling"),
        ("oversampling"),
        ("hybrid")
    ],
)

def test_ensemble_integration(sampling_strategy):

    from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from xgboost import XGBClassifier
    from sklearn.datasets import make_classification
    from eipy.ei import EnsembleIntegration
    from eipy.additional_ensembles import MeanAggregation, MedianAggregation, CES
    import pandas as pd

    # Generate toy data for testing
    X, y = make_classification(n_samples=200, n_features=10, n_classes=2, weights=[0.7, 0.3], n_redundant=0)
    
    X_1 = X[:, :4]
    X_2 = X[:, 4:]

    modalities = {
                "modality_1": X_1,
                "modality_2": X_2
                }

    modalities_df = {
            "modality_1": pd.DataFrame(X_1, columns=['a', 'b', 'c', 'd']),
            "modality_2": pd.DataFrame(X_1, columns=['e', 'b', 'a', 'd']),
            }
    
    test_data = {'numpy': modalities, 
                 'pandas df': modalities_df}

    for k, multimodal_data in test_data.items():
        # Create base predictor models
        print(f'Testing input as {k}')

        base_predictors = {
            'DT': DecisionTreeClassifier(),
            'LR': LogisticRegression(),
            'NB': GaussianNB(),
            'XGB': XGBClassifier()
        }

        # Initialize EnsembleIntegration
        EI = EnsembleIntegration(base_predictors=base_predictors,
                                k_outer=2,
                                k_inner=2,
                                n_samples=2,
                                sampling_strategy=sampling_strategy,
                                sampling_aggregation="mean",
                                n_jobs=-1,
                                random_state=42,
                                project_name="demo",
                                model_building=True)

        # Train base models
        for name, modality in multimodal_data.items():
            EI.train_base(modality, y, base_predictors, modality=name)

        # Train meta models
        meta_predictors = {
            "Mean": MeanAggregation(),
            "Median": MedianAggregation(),
            "CES": CES(),
            "S.DT": DecisionTreeClassifier(),
            "S.LR": LogisticRegression(),
            "S.XGB": XGBClassifier()
            # "S.NB": GaussianNB(),
        }

        EI.train_meta(meta_predictors=meta_predictors)

        # Assertions

        # Check if the trained base models and meta models are not None
        assert EI.base_summary is not None
        assert EI.meta_summary is not None
        assert EI.final_models is not {"base models": {}, "meta models": {}}

        from eipy.interpretation import PermutationInterpreter
        from eipy.utils import f_minority_score

        interpreter = PermutationInterpreter(EI=EI,
                                        metric=f_minority_score,
                                        meta_predictor_keys=['S.LR', 'Mean', 'CES', 'S.XGB'])
        
        interpreter.rank_product_score(X_dict=multimodal_data, y=y)

        assert interpreter.ensemble_feature_ranking is not None