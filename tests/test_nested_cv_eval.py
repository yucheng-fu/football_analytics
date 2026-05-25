import numpy as np
import pandas as pd
import pytest

from training.nested_cv_evaluator import ModelCVEvaluator


class LGBMClassifier:
    def __init__(self, best_iteration_):
        self.best_iteration_ = best_iteration_


class XGBClassifier:
    def __init__(self, best_iteration):
        self.best_iteration = best_iteration


class CatBoostClassifier:
    def __init__(self, best_iteration):
        self._best_iteration = best_iteration

    def get_best_iteration(self):
        return self._best_iteration


class UnknownModel:
    pass


def _build_evaluator(categorical_columns=None):
    evaluator = ModelCVEvaluator.__new__(ModelCVEvaluator)
    evaluator.categorical_columns = categorical_columns or []
    return evaluator


def test_extract_and_apply_category_schema_preserves_categories():
    evaluator = _build_evaluator()
    train_df = pd.DataFrame(
        {
            "team": pd.Categorical(["A", "B", "A"], categories=["A", "B"]),
            "value": [1, 2, 3],
        }
    )
    val_df = pd.DataFrame({"team": ["B", "C"], "value": [5, 8]})

    schema = evaluator._extract_category_schema(train_df)
    transformed = evaluator._apply_category_schema(val_df, schema)

    assert list(schema["team"]) == ["A", "B"]
    assert isinstance(transformed["team"].dtype, pd.CategoricalDtype)
    assert list(transformed["team"].cat.categories) == ["A", "B"]


def test_encode_categories_to_codes_transforms_categorical_columns_only():
    evaluator = _build_evaluator()
    input_df = pd.DataFrame(
        {
            "cat": pd.Categorical(["home", "away", "home"]),
            "num": [10, 20, 30],
        }
    )

    encoded = evaluator._encode_categories_to_codes(input_df)

    assert encoded["cat"].tolist() == [1, 0, 1]
    assert encoded["num"].tolist() == [10, 20, 30]


def test_categorical_feature_names_filters_unknown_and_non_string_columns():
    evaluator = _build_evaluator(categorical_columns=["league", 3, "missing"])
    X_pd = pd.DataFrame({"league": ["EPL", "LaLiga"], "x": [1, 2]})

    feature_names = evaluator._categorical_feature_names(X_pd)

    assert feature_names == ["league"]


def test_apply_categorical_dtypes_casts_object_and_configured_columns():
    evaluator = _build_evaluator(categorical_columns=["encoded_cat"])
    X_pd = pd.DataFrame(
        {
            "text_col": ["a", "b"],
            "encoded_cat": [1, 2],
            "num": [0.1, 0.2],
        }
    )

    transformed = evaluator._apply_categorical_dtypes(X_pd)

    assert isinstance(transformed["text_col"].dtype, pd.CategoricalDtype)
    assert isinstance(transformed["encoded_cat"].dtype, pd.CategoricalDtype)
    assert transformed["num"].dtype == X_pd["num"].dtype


def test_init_oof_predictions_binary_and_multiclass_shapes():
    evaluator = _build_evaluator()

    binary = evaluator._init_oof_predictions(n_samples=5, n_classes=2)
    multiclass = evaluator._init_oof_predictions(n_samples=5, n_classes=3)

    assert binary.shape == (5,)
    assert np.allclose(binary, 0)
    assert multiclass.shape == (5, 3)
    assert np.allclose(multiclass, 0)


def test_get_best_iteration_for_supported_models_and_unknown_model():
    evaluator = _build_evaluator()

    assert evaluator.get_best_iteration(LGBMClassifier(best_iteration_=17)) == 17
    assert evaluator.get_best_iteration(XGBClassifier(best_iteration=23)) == 23
    assert evaluator.get_best_iteration(CatBoostClassifier(best_iteration=11)) == 11
    assert evaluator.get_best_iteration(UnknownModel()) is None


def test_augment_params_with_boosting_rounds_adds_iteration_when_present():
    evaluator = _build_evaluator()
    base_params = {"learning_rate": 0.1}

    augmented = evaluator._augment_params_with_boosting_rounds(
        model=LGBMClassifier(best_iteration_=9), best_params=base_params
    )

    assert augmented["learning_rate"] == 0.1
    assert augmented["n_estimators_used"] == 9
    assert "n_estimators_used" not in base_params


def test_extract_validation_loss_for_framework_specific_structures():
    evaluator = _build_evaluator()

    lgbm = LGBMClassifier(best_iteration_=None)
    lgbm.evals_result_ = {"valid_0": {"binary_logloss": [0.8, 0.5]}}
    xgb = XGBClassifier(best_iteration=None)
    xgb.evals_result_ = {"validation_0": {"logloss": [0.7, 0.4]}}
    cat = CatBoostClassifier(best_iteration=None)
    cat.evals_result_ = {"validation": {"Logloss": [0.9, 0.3]}}

    assert evaluator._extract_validation_loss(lgbm) == pytest.approx(0.5)
    assert evaluator._extract_validation_loss(xgb) == pytest.approx(0.4)
    assert evaluator._extract_validation_loss(cat) == pytest.approx(0.3)


def test_extract_validation_loss_raises_for_unknown_model():
    evaluator = _build_evaluator()
    model = UnknownModel()
    model.evals_result_ = {}

    with pytest.raises(ValueError, match="Unknown framework model"):
        evaluator._extract_validation_loss(model)


def test_get_feature_selector_raises_for_invalid_transform():
    evaluator = _build_evaluator()

    with pytest.raises(ValueError, match="Invalid transform"):
        evaluator._get_feature_selector(
            fit_params={"max_depth": 3},
            n_features_to_select=0.5,
            transform="invalid",
        )
