import logging
from typing import List, Optional

import mlflow
import numpy as np
import pandas as pd
import polars as pl
from lightgbm import LGBMClassifier
from mlflow.models import infer_signature
from sklearn.metrics import log_loss

from feature_engineering.ColumnTransformer import ColumnTransformer
from feature_engineering.OpenFE.FeatureGenerator import Node
from feature_engineering.OpenFE.utils import transform, tree_to_formula
from feature_engineering.RowWiseTransformations import RowWiseTransformations
from utils.statics import lightgbm_model_name, tracking_uri


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
        input_example = X_data.head(10)

        log_fn_mapping = {
            lightgbm_model_name: mlflow.lightgbm.log_model,
        }
        log_fn = log_fn_mapping.get(self.model_type)
        if log_fn is None:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        log_fn(
            final_model,
            name=self.model_type,
            signature=signature,
            input_example=input_example,
        )

    def _apply_feature_nodes(self, X_train_pd: pd.DataFrame) -> pd.DataFrame:
        """Apply all OpenFE feature nodes to training data.

        Internally, stateful nodes still use ColumnTransformer logic while row-wise
        nodes are expanded via OpenFE transform. This is a single trainer entrypoint.
        """
        if not self.row_wise_features and not self.column_wise_features:
            return X_train_pd

        X_out = X_train_pd
        if self.row_wise_features:
            # OpenFE row-wise transform expects train/test together to keep naming consistent.
            X_out, _ = transform(
                X_train=X_out,
                X_test=X_out.copy(),
                new_features_list=self.row_wise_features,
                n_jobs=1,
            )

        if self.column_wise_features:
            formula_to_safe_name = {
                tree_to_formula(node): f"ofe_col_{idx + 1}"
                for idx, node in enumerate(self.column_wise_features)
            }
            column_transformer = ColumnTransformer(
                feature_name_mapping=formula_to_safe_name
            )
            column_transformer.fit(X_out, feature_nodes=self.column_wise_features)
            X_out = column_transformer.transform(X_out)

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

    def train(self, X_train: pl.DataFrame, y_train: pl.DataFrame) -> LGBMClassifier:
        """Train final model using params + row/column-wise OFE features + selected features."""
        self.setup_mlflow()

        X_train_pd = X_train.to_pandas().copy()
        y_train_np = y_train.to_numpy().ravel()

        X_train_pd = self.row_wise_transformations.apply_row_wise_transformations(
            X_train_pd
        )
        X_train_pd = self._apply_feature_nodes(X_train_pd)
        X_train_pd = self._apply_categorical_dtypes(X_train_pd)

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
            train_log_loss = log_loss(y_train_np, y_pred_proba)

            self.log_model(final_model=model, X_data=X_train_final, output=y_pred_proba)
            mlflow.log_params(self.params)
            mlflow.set_tag("features", ",".join(map(str, self.features)))
            mlflow.log_metric("train_loss", train_log_loss)
            mlflow.register_model(
                model_uri=f"runs:/{run.info.run_id}/{self.model_type}",
                name=f"{self.experiment_name}_{self.model_type}",
            )
            mlflow.set_tag("alias", "production")

            return model
