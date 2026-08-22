import numpy as np
import pandas as pd

from src.models.ensemble import FraudEnsemble


def test_ensemble_prediction_shape_and_range():
    model = FraudEnsemble(
        {
            "data": {"random_seed": 42},
            "xgboost": {"eval_metric": "auc", "tree_method": "hist"},
            "lightgbm": {},
        }
    )

    class DummyModel:
        def predict_proba(self, X):
            return np.column_stack([np.full(len(X), 0.8), np.full(len(X), 0.2)])

    model.xgb_model = DummyModel()
    model.lgb_model = DummyModel()
    model.cat_model = DummyModel()

    X = pd.DataFrame({"feature": [1.0, 2.0, 3.0]})
    result = model.predict_individual(X)

    assert set(result) == {"XGBoost", "LightGBM", "CatBoost", "Ensemble"}
    assert all(len(values) == len(X) for values in result.values())
    assert np.all((result["Ensemble"] >= 0) & (result["Ensemble"] <= 1))
    assert np.allclose(result["Ensemble"], 0.2)
