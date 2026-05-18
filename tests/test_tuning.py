import numpy as np
import pandas as pd
from types import SimpleNamespace
from unittest.mock import MagicMock

from training.tuning import ModelParamTuner


class LGBMClassifier:
    pass


class XGBClassifier:
    pass


class CatBoostClassifier:
    pass


def _build_tuner():
    tuner = ModelParamTuner.__new__(ModelParamTuner)
    tuner.model_type = "lightgbm"
    tuner.logger = MagicMock()
    tuner.mlflow_handler = MagicMock()
    tuner.wrapper = MagicMock()
    tuner.use_feature_engineering = False
    tuner.row_wise_transformations = None
    tuner.column_wise_transformations = None
    tuner.open_fe_transformations = None
    tuner.use_ofe = False
    tuner.log_hyperparameter_trials = False
    tuner.experiment_name = "exp"
    tuner.run_name = "run"
    tuner.log_feature_importance = False
    return tuner


def test_extract_loss_history_for_lgbm_training_and_validation():
    tuner = _build_tuner()
    model = LGBMClassifier()
    model.evals_result_ = {
        "training": {"binary_logloss": [0.8, 0.5]},
        "valid_0": {"binary_logloss": [0.9, 0.6]},
    }

    train_loss, valid_loss = tuner._extract_loss_history(model)

    assert train_loss == [0.8, 0.5]
    assert valid_loss == [0.9, 0.6]


def test_extract_loss_history_for_xgboost_with_train_and_validation_sets():
    tuner = _build_tuner()
    model = XGBClassifier()
    model.evals_result_ = {
        "validation_0": {"logloss": [0.7, 0.4]},
        "validation_1": {"logloss": [0.8, 0.5]},
    }

    train_loss, valid_loss = tuner._extract_loss_history(model)

    assert train_loss == [0.7, 0.4]
    assert valid_loss == [0.8, 0.5]


def test_extract_loss_history_for_catboost():
    tuner = _build_tuner()
    model = CatBoostClassifier()
    model.evals_result_ = {
        "learn": {"Logloss": [0.6, 0.3]},
        "validation": {"Logloss": [0.7, 0.35]},
    }

    train_loss, valid_loss = tuner._extract_loss_history(model)

    assert train_loss == [0.6, 0.3]
    assert valid_loss == [0.7, 0.35]


def test_fit_model_and_select_features_with_loss_curve_logs_figure_when_enabled(monkeypatch):
    tuner = _build_tuner()
    X_train = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    y_train = np.array([0, 1])
    X_val = pd.DataFrame({"a": [5], "b": [6]})
    y_val = np.array([1])

    selected_features = np.array(["a", "b"])
    category_schema = {"a": pd.Index([1, 2])}
    tuner._get_selected_features = MagicMock(
        return_value=(X_train, X_val, selected_features)
    )
    tuner._extract_category_schema = MagicMock(return_value=category_schema)

    fitted_model = LGBMClassifier()
    fitted_model.evals_result_ = {
        "training": {"binary_logloss": [0.9, 0.7]},
        "valid_0": {"binary_logloss": [1.0, 0.8]},
    }
    tuner.wrapper.fit.return_value = fitted_model

    fake_fig = object()
    monkeypatch.setattr("training.tuning.plot_loss_curve", lambda *args: fake_fig)

    model, features, schema = tuner._fit_model_and_select_features_with_loss_curve(
        X_train=X_train,
        y_train=y_train,
        params={"lr": 0.1},
        X_val=X_val,
        y_val=y_val,
        should_plot_loss_curve=True,
    )

    assert model is fitted_model
    assert np.array_equal(features, selected_features)
    assert schema == category_schema
    tuner.wrapper.fit.assert_called_once()
    assert tuner.wrapper.fit.call_args.kwargs["use_early_stopping"] is True
    tuner.mlflow_handler.log_figure.assert_called_once_with(
        fig=fake_fig, name="lightgbm_loss_curve"
    )


def test_fit_model_and_select_features_with_loss_curve_skips_curve_without_validation():
    tuner = _build_tuner()
    X_train = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    y_train = np.array([0, 1])

    selected_features = np.array(["a", "b"])
    tuner._get_selected_features = MagicMock(
        return_value=(X_train, None, selected_features)
    )
    tuner._extract_category_schema = MagicMock(return_value={})

    fitted_model = LGBMClassifier()
    fitted_model.evals_result_ = {
        "training": {"binary_logloss": [0.9, 0.7]},
        "valid_0": {"binary_logloss": [1.0, 0.8]},
    }
    tuner.wrapper.fit.return_value = fitted_model

    tuner._fit_model_and_select_features_with_loss_curve(
        X_train=X_train,
        y_train=y_train,
        params={"lr": 0.1},
        X_val=None,
        y_val=None,
        should_plot_loss_curve=True,
    )

    assert tuner.wrapper.fit.call_args.kwargs["use_early_stopping"] is False
    tuner.mlflow_handler.log_figure.assert_not_called()


def test_tune_and_train_wires_pipeline_and_appends_results(monkeypatch):
    tuner = _build_tuner()

    X = pd.DataFrame({"f1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], "f2": [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]})
    y = pd.DataFrame({"target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]})

    class FakePolarsFrame:
        def __init__(self, df):
            self._df = df

        def to_pandas(self):
            return self._df

        def to_numpy(self):
            return self._df.to_numpy()

    X_pl = FakePolarsFrame(X)
    y_pl = FakePolarsFrame(y)

    tuner._setup_mlflow = MagicMock()
    tuner._hyperparameter_tuning = MagicMock(return_value={"depth": 3})
    tuner._fit_final_model_with_loss_curve = MagicMock(
        return_value=(
            LGBMClassifier(),
            np.array(["f1", "f2"]),
            {},
            None,
        )
    )
    tuner._eval_validation_loss = MagicMock(
        return_value=(0.42, np.array([[0.2, 0.8]]))
    )
    tuner._augment_params_with_boosting_rounds = MagicMock(
        return_value={"depth": 3, "n_estimators_used": 10}
    )
    tuner._append_and_log_metrics_and_params = MagicMock()

    run_ctx = SimpleNamespace(
        info=SimpleNamespace(run_id="parent-run", experiment_id="exp-id")
    )

    class DummyRun:
        def __enter__(self):
            return run_ctx

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr("training.tuning.mlflow.start_run", lambda **kwargs: DummyRun())
    monkeypatch.setattr("training.tuning.mlflow.end_run", lambda: None)

    results = tuner.tune_and_train(X_pl, y_pl)

    tuner._setup_mlflow.assert_called_once()
    tuner._hyperparameter_tuning.assert_called_once()
    tuner._fit_final_model_with_loss_curve.assert_called_once()
    tuner._eval_validation_loss.assert_called_once()
    tuner._append_and_log_metrics_and_params.assert_called_once()
    assert results.parent_run_id == "parent-run"
