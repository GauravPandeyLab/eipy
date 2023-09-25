import pytest

@pytest.mark.parametrize(
    "sampling_strategy, dtype",
    [   
        (None, "numpy_array"),
        ("undersampling", "numpy_array"),
        ("oversampling", "numpy_array"),
        ("hybrid", "numpy_array"),
        ("undersampling", "pandas_df")
    ],
)

def test_ensemble_integration(sampling_strategy, dtype):

    from sklearn.linear_model import LogisticRegression
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from xgboost import XGBClassifier
    from sklearn.datasets import make_classification
    from eipy.ei import EnsembleIntegration
    from eipy.additional_ensembles import MeanAggregation, MedianAggregation, CES
    import pandas as pd

    # Generate toy data for testing
    X, y = make_classification(n_samples=200, n_features=10, n_classes=2, weights=[0.7, 0.3], n_redundant=0)
    
    X_1 = X[:, :4]
    X_2 = X[:, 4:]

    if dtype=="numpy_array":
        modalities = {
                    "modality_1": X_1,
                    "modality_2": X_2
                    }
    elif dtype=="pandas_df":
        modalities = {
                    "modality_1": pd.DataFrame(X_1, columns=['a', 'b', 'c', 'd']),
                    "modality_2": pd.DataFrame(X_2, columns=['e', 'f', 'g', 'h', 'i', 'j']),
                    }

    # Create base predictor models
    base_predictors = {
        'LR': Pipeline([('scaler', StandardScaler()), ('lr', LogisticRegression())]),
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
                             calibration_model=CalibratedClassifierCV(cv=2),
                             model_building=True)

        # Train base models
    for name, modality in modalities.items():
        EI.train_base(modality, y, base_predictors, modality=name)

    # Train meta models
    meta_predictors = {
        "Mean": MeanAggregation(),
        "Median": MedianAggregation(),
        "CES": CES(),
        "S.LR": Pipeline([('scaler', StandardScaler()), ('lr', LogisticRegression())]),
    }

    EI.train_meta(meta_predictors=meta_predictors)

    # Predict
    EI.predict(modalities, meta_model_key='S.LR')

    # Assertions

    # Check if the trained base models and meta models are not None
    assert EI.base_summary is not None
    assert EI.meta_summary is not None
    assert EI.final_models is not {"base models": {}, "meta models": {}}

    from eipy.interpretation import PermutationInterpreter
    from eipy.utils import f_minority_score

    interpreter = PermutationInterpreter(
                                        EI=EI,
                                        metric=f_minority_score,
                                        meta_predictor_keys=['S.LR', 'Mean', 'CES'],
                                        n_repeats=5,
                                        n_jobs=-1,
                                        metric_greater_is_better=True
                                        )
    
    interpreter.rank_product_score(X_dict=modalities, y=y)

    assert interpreter.ensemble_feature_ranking is not None

    if dtype=="pandas_df":
        assert EI.feature_names_dict.keys() == ["Modality_1", "Modality_2"]
        assert EI.feature_names_dict["Modality_1"] == ["a", "b", "c", "d"]
        assert EI.feature_names_dict["Modality_2"] == ["e", "f", "g", "h", "i", "j"]