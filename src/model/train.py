import logging
from typing import List, Optional

import mlflow
import numpy as np
import pandas as pd
import polars as pl
from lightgbm import LGBMClassifier
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.metrics import log_loss

from src.feature_engineering.ColumnTransformer import ColumnTransformer
from src.feature_engineering.OpenFE.FeatureGenerator import Node
from src.feature_engineering.OpenFE.utils import tree_to_formula
from src.feature_engineering.RowWiseTransformations import RowWiseTransformations
from src.utils.statics import lightgbm_model_name, tracking_uri
from src.utils.utils import safe_production_transform


class ModelTrainer:
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    def __init__(
        self,
        model_type: str,
        params: dict,
        features: List[str],
        row_wise_features: Optional[List[Node]] = None,
        column_wise_features: Optional[List[Node]] = None,
        feature_nodes: Optional[List[Node]] = None,
        row_wise_transformations: Optional[RowWiseTransformations] = None,
        categorical_columns: Optional[List[str]] = None,
        run_name: str = "baseline_training",
        experiment_name: str = "Final models",
    ):
        self.model_type = model_type
        self.params = params
        self.features = features
        self.row_wise_features = row_wise_features or []
        self.column_wise_features = column_wise_features or []
        # Backward compatibility: allow a single feature_nodes list.
        if feature_nodes:
            self.row_wise_features.extend([n for n in feature_nodes if n.is_rowwise])
            self.column_wise_features.extend(
                [n for n in feature_nodes if not n.is_rowwise]
            )
        self.row_wise_transformations = (
            row_wise_transformations
            if row_wise_transformations is not None
            else RowWiseTransformations()
        )
        self.categorical_columns = categorical_columns or []
        self.run_name = run_name
        self.experiment_name = experiment_name

    def setup_mlflow(self) -> None:
        """Set up MLflow tracking and experiment."""
        params_str = ", ".join(f"{k}={v}" for k, v in self.params.items())
        features_str = ", ".join(self.features)
        self.logger.info(
            f"""Starting training with model {self.model_type} with the following configuration:
        - Params: {params_str}
        - Features: {features_str}
        - Row-wise OFE nodes: {len(self.row_wise_features)}
        - Column-wise OFE nodes: {len(self.column_wise_features)}
        - Row-wise transformations: enabled
        """
        )

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name=self.experiment_name)

    def _effective_params(self, model: LGBMClassifier) -> dict:
        """Build model params from logged params and metadata fields."""
        params = self.params.copy()
        if "n_estimators_used" in params and "n_estimators" in model.get_params():
            params["n_estimators"] = int(params["n_estimators_used"])

        accepted = set(model.get_params().keys())
        return {k: v for k, v in params.items() if k in accepted}

    def set_params(self, model: LGBMClassifier) -> LGBMClassifier:
        """Apply effective params to model."""
        model.set_params(**self._effective_params(model))
        return model

    def fetch_model(self) -> LGBMClassifier:
        """Return a fresh estimator instance for the configured model type."""
        if self.model_type == lightgbm_model_name:
            return LGBMClassifier(verbose=-1)

        raise ValueError(f"Unsupported model type: {self.model_type}")

    def log_model(
        self,
        final_model: LGBMClassifier,
        X_data: pd.DataFrame,
        output: np.ndarray,
    ) -> None:
        """Log trained model to MLflow."""
        signature = infer_signature(X_data, output)

        log_fn_mapping = {
            lightgbm_model_name: mlflow.lightgbm.log_model,
        }
        log_fn = log_fn_mapping.get(self.model_type)
        if log_fn is None:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        log_fn(final_model, name=self.model_type, signature=signature)

    def _apply_openfe_rowwise_nodes(self, X_pd: pd.DataFrame) -> pd.DataFrame:
        """Apply OpenFE row-wise nodes to a single dataset."""
        if not self.row_wise_features:
            return X_pd
        return safe_production_transform(X_pd, self.row_wise_features)

    def _apply_openfe_columnwise_nodes(self, X_pd: pd.DataFrame) -> pd.DataFrame:
        """Apply OpenFE column-wise nodes via fitted ColumnTransformer."""
        if not self.column_wise_features:
            return X_pd

        formula_to_safe_name = {
            tree_to_formula(node): f"ofe_col_{idx + 1}"
            for idx, node in enumerate(self.column_wise_features)
        }
        column_transformer = ColumnTransformer(
            feature_name_mapping=formula_to_safe_name
        )
        column_transformer.fit(X_pd, feature_nodes=self.column_wise_features)
        return column_transformer.transform(X_pd)

    def _build_training_frame(self, X_train: pl.DataFrame) -> pd.DataFrame:
        """Build final training dataframe with all feature engineering steps applied."""
        X_pd = X_train.to_pandas().copy()
        X_pd = self.row_wise_transformations.apply_row_wise_transformations(X_pd)
        X_pd = self._apply_openfe_rowwise_nodes(X_pd)
        X_pd = self._apply_openfe_columnwise_nodes(X_pd)
        return self._apply_categorical_dtypes(X_pd)

    def _add_legacy_openfe_aliases(self, X_pd: pd.DataFrame) -> pd.DataFrame:
        """Add compatibility aliases for legacy OpenFE row-wise names (autoFE_f_*)."""
        if not self.row_wise_features:
            return X_pd

        X_out = X_pd.copy()
        for idx, node in enumerate(self.row_wise_features):
            legacy_name = f"autoFE_f_{idx}"
            formula_name = tree_to_formula(node)
            if legacy_name not in X_out.columns and formula_name in X_out.columns:
                X_out[legacy_name] = X_out[formula_name]
        return X_out

    def _categorical_feature_names(self, X_pd: pd.DataFrame) -> list[str]:
        """Return configured categorical column names that exist in ``X_pd``."""
        names: list[str] = []
        for col in self.categorical_columns:
            if isinstance(col, str) and col in X_pd.columns:
                names.append(col)
        return names

    def _apply_categorical_dtypes(self, X_pd: pd.DataFrame) -> pd.DataFrame:
        """Return a copy of ``X_pd`` with categorical dtypes enforced."""
        X_pd_copy = X_pd.copy()

        object_like_cols = X_pd_copy.select_dtypes(include=["object", "string"]).columns
        for name in object_like_cols:
            X_pd_copy[name] = X_pd_copy[name].astype("category")

        for name in self._categorical_feature_names(X_pd_copy):
            X_pd_copy[name] = X_pd_copy[name].astype("category")

        return X_pd_copy

    def _register_and_tag_model(self, run_id: str) -> None:
        """Register model and set model-version metadata in MLflow registry."""
        model_name = f"{self.experiment_name}_{self.model_type}"
        registered_model = mlflow.register_model(
            model_uri=f"runs:/{run_id}/{self.model_type}",
            name=model_name,
        )
        client = MlflowClient()
        client.set_model_version_tag(
            name=model_name,
            version=registered_model.version,
            key="alias",
            value="production",
        )
        client.set_registered_model_alias(
            name=model_name,
            alias="production",
            version=registered_model.version,
        )

    def _log_training_run(
        self,
        model: LGBMClassifier,
        X_train_final: pd.DataFrame,
        y_pred_proba: np.ndarray,
        y_train_np: np.ndarray,
        run_id: str,
    ) -> None:
        """Log artifacts/metrics and register trained model."""
        train_log_loss = log_loss(y_train_np, y_pred_proba)
        self.log_model(final_model=model, X_data=X_train_final, output=y_pred_proba)
        mlflow.log_params(self.params)
        mlflow.set_tag("features", ",".join(map(str, self.features)))
        mlflow.log_metric("train_loss", train_log_loss)
        self._register_and_tag_model(run_id=run_id)

    def train(self, X_train: pl.DataFrame, y_train: pl.DataFrame) -> LGBMClassifier:
        """Train final model using params + row/column-wise OFE features + selected features."""
        self.setup_mlflow()

        X_train_pd = self._build_training_frame(X_train)
        X_train_pd = self._add_legacy_openfe_aliases(X_train_pd)
        y_train_np = y_train.to_numpy().ravel()

        missing_features = [f for f in self.features if f not in X_train_pd.columns]
        if missing_features:
            raise ValueError(
                f"Missing selected features after transformations: {missing_features}"
            )

        X_train_final = X_train_pd[self.features]

        with mlflow.start_run(run_name=f"{self.run_name}_{self.model_type}") as run:
            model = self.fetch_model()
            model = self.set_params(model=model)

            model.fit(
                X_train_final, y_train_np, feature_name=list(X_train_final.columns)
            )

            y_pred_proba = model.predict_proba(X_train_final)
            self._log_training_run(
                model=model,
                X_train_final=X_train_final,
                y_pred_proba=y_pred_proba,
                y_train_np=y_train_np,
                run_id=run.info.run_id,
            )

            return model
