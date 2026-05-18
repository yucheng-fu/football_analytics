from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from training.train import ModelTrainer


class _BaseEstimator:
    def get_params(self):
        return {"n_estimators": 100, "learning_rate": 0.1}


class _FittedModel:
    def predict_proba(self, X):
        return np.array([[0.2, 0.8] for _ in range(len(X))])


class _FakePolarsFrame:
    def __init__(self, df):
        self._df = df

    def to_pandas(self):
        return self._df

    def to_numpy(self):
        return self._df.to_numpy()


def _build_trainer():
    trainer = ModelTrainer.__new__(ModelTrainer)
    trainer.model_type = "lightgbm"
    trainer.params = {"learning_rate": 0.05, "n_estimators_used": 15, "unused": 1}
    trainer.features = ["f1", "f2"]
    trainer.row_wise_features = []
    trainer.column_wise_features = []
    trainer.row_wise_transformations = MagicMock()
    trainer.row_wise_transformations.apply_row_wise_transformations.side_effect = lambda df: df
    trainer.categorical_columns = []
    trainer.run_name = "baseline"
    trainer.experiment_name = "final_models"
    trainer.logger = MagicMock()
    trainer.mlflow_handler = MagicMock()
    trainer.wrapper = MagicMock()
    return trainer


def test_effective_params_uses_n_estimators_used_and_filters_unknown():
    trainer = _build_trainer()
    estimator = _BaseEstimator()

    effective = trainer._effective_params(estimator)

    assert effective == {"learning_rate": 0.05, "n_estimators": 15}


def test_apply_openfe_rowwise_nodes_returns_original_when_no_nodes():
    trainer = _build_trainer()
    trainer.row_wise_features = []
    X_pd = pd.DataFrame({"x": [1, 2]})

    result = trainer._apply_openfe_rowwise_nodes(X_pd)

    assert result.equals(X_pd)


def test_apply_openfe_rowwise_nodes_uses_safe_production_transform(monkeypatch):
    trainer = _build_trainer()
    trainer.row_wise_features = [object()]
    X_pd = pd.DataFrame({"x": [1, 2]})

    expected = pd.DataFrame({"x": [3, 4]})

    def _fake_transform(df, nodes):
        assert nodes == trainer.row_wise_features
        return expected

    monkeypatch.setattr("training.train.safe_production_transform", _fake_transform)

    result = trainer._apply_openfe_rowwise_nodes(X_pd)

    assert result.equals(expected)


def test_add_legacy_openfe_aliases_adds_missing_alias_columns(monkeypatch):
    trainer = _build_trainer()
    trainer.row_wise_features = ["node1"]

    monkeypatch.setattr("training.train.tree_to_formula", lambda node: "new_formula")

    X_pd = pd.DataFrame({"new_formula": [10, 20]})
    result = trainer._add_legacy_openfe_aliases(X_pd)

    assert "autoFE_f_0" in result.columns
    assert result["autoFE_f_0"].tolist() == [10, 20]


def test_apply_categorical_dtypes_casts_object_and_configured_columns():
    trainer = _build_trainer()
    trainer.categorical_columns = ["explicit_cat"]

    X_pd = pd.DataFrame(
        {
            "text": ["a", "b"],
            "explicit_cat": [1, 2],
            "num": [0.1, 0.2],
        }
    )

    transformed = trainer._apply_categorical_dtypes(X_pd)

    assert isinstance(transformed["text"].dtype, pd.CategoricalDtype)
    assert isinstance(transformed["explicit_cat"].dtype, pd.CategoricalDtype)
    assert transformed["num"].dtype == X_pd["num"].dtype


def test_build_categorical_mapping_returns_only_categorical_columns():
    trainer = _build_trainer()
    X_pd = pd.DataFrame(
        {
            "cat_col": pd.Categorical(["A", "B"], categories=["A", "B", "C"]),
            "num_col": [1, 2],
        }
    )

    mapping = trainer._build_categorical_mapping(X_pd)

    assert mapping == {"cat_col": ["A", "B", "C"]}


def test_train_raises_when_selected_feature_is_missing():
    trainer = _build_trainer()
    trainer.features = ["f1", "f_missing"]
    trainer.setup_mlflow = MagicMock()

    X_train = _FakePolarsFrame(pd.DataFrame({"f1": [1, 2], "f2": [3, 4]}))
    y_train = _FakePolarsFrame(pd.DataFrame({"target": [0, 1]}))

    with pytest.raises(ValueError, match="Missing selected features"):
        trainer.train(X_train, y_train)


def test_train_runs_fit_and_logs_with_categorical_mapping(monkeypatch):
    trainer = _build_trainer()
    trainer.setup_mlflow = MagicMock()
    trainer.features = ["cat_f", "num_f"]

    X_df = pd.DataFrame(
        {
            "cat_f": ["home", "away", "home", "away"],
            "num_f": [1.0, 2.0, 3.0, 4.0],
        }
    )
    y_df = pd.DataFrame({"target": [0, 1, 0, 1]})

    trainer.wrapper.fetch_base_estimator.return_value = _BaseEstimator()
    fitted_model = _FittedModel()
    trainer.wrapper.fit.return_value = fitted_model
    trainer._log_training_run = MagicMock()

    run_ctx = SimpleNamespace(info=SimpleNamespace(run_id="run-123"))

    class DummyRun:
        def __enter__(self):
            return run_ctx

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr("training.train.mlflow.start_run", lambda **kwargs: DummyRun())
    monkeypatch.setattr("training.train.mlflow.end_run", lambda: None)

    model = trainer.train(_FakePolarsFrame(X_df), _FakePolarsFrame(y_df))

    assert model is fitted_model
    trainer.wrapper.fit.assert_called_once()
    assert trainer.wrapper.fit.call_args.kwargs["use_early_stopping"] is False
    assert hasattr(model, "categorical_mapping_")
    assert model.categorical_mapping_["cat_f"] == ["away", "home"]
    trainer._log_training_run.assert_called_once()
    assert trainer._log_training_run.call_args.kwargs["run_id"] == "run-123"
