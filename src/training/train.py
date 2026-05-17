import logging
import json
import random
from typing import List, Optional, Union

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
import mlflow
import numpy as np
import pandas as pd
import polars as pl
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import log_loss

from feature_engineering.ColumnTransformer import ColumnTransformer
from feature_engineering.OpenFE.FeatureGenerator import Node
from feature_engineering.OpenFE.utils import tree_to_formula
from feature_engineering.RowWiseTransformations import RowWiseTransformations
import utils.statics as statics
from utils.mlflow_handler import MLflowHandler
from utils.utils import safe_production_transform
from model.LGBMWrapper import LGBMWrapper
from model.XGBoostWrapper import XGBoostWrapper
from model.CatBoostWrapper import CatBoostWrapper


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
        self.row_wise_transformations = (
            row_wise_transformations
            if row_wise_transformations is not None
            else RowWiseTransformations()
        )
        self.categorical_columns = categorical_columns or []
        self.run_name = run_name
        self.experiment_name = experiment_name
        self.mlflow_handler = MLflowHandler(
            tracking_uri=statics.tracking_uri,
            experiment_name=experiment_name,
            logger=self.logger,
        )
        self.seed = 165
        self._set_global_seed()
        self.wrapper = self._fetch_model_wrapper(self.model_type)

    def _set_global_seed(self):
        """Set global random seeds used by NumPy and Python's random module."""
        np.random.seed(self.seed)
        random.seed(self.seed)

    def _fetch_model_wrapper(
        self, model_name: str
    ) -> Union[LGBMWrapper, XGBoostWrapper, CatBoostWrapper]:
        """Factory method to return a model wrapper instance based on the model name.

        Args:
            model_name (str): Name of the model type (e.g., "lightgbm", "xgboost", "catboost").

        Returns:
            BaseModelWrapper: An instance of a class that inherits from BaseModelWrapper.

        Raises:
            ValueError: If the provided model_name is not supported.
        """
        match model_name:
            case statics.lightgbm_model_name:
                return LGBMWrapper(seed=self.seed)
            case statics.xgboost_model_name:
                return XGBoostWrapper(seed=self.seed)
            case statics.catboost_model_name:
                return CatBoostWrapper(seed=self.seed)

        raise ValueError(f"Unsupported model type: {model_name}")

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
        self.mlflow_handler.setup()

    def _effective_params(
        self, model: Union[LGBMClassifier, XGBClassifier, CatBoostClassifier]
    ) -> dict:
        """Build model params from logged params and metadata fields."""
        params = self.params.copy()
        if "n_estimators_used" in params and "n_estimators" in model.get_params():
            params["n_estimators"] = int(params["n_estimators_used"])

        accepted = set(model.get_params().keys())
        return {k: v for k, v in params.items() if k in accepted}

    def _apply_openfe_rowwise_nodes(self, X_pd: pd.DataFrame) -> pd.DataFrame:
        """Apply OpenFE row-wise nodes to a single dataset."""
        if not self.row_wise_features:
            return X_pd
        return safe_production_transform(X_pd, self.row_wise_features)

    def _apply_openfe_columnwise_nodes(
        self, X_pd: pd.DataFrame
    ) -> tuple[pd.DataFrame, Optional[ColumnTransformer]]:
        """Apply OpenFE column-wise nodes and return transformed data + fitted transformer."""
        if not self.column_wise_features:
            return X_pd, None

        formula_to_safe_name = {
            tree_to_formula(node): f"ofe_col_{idx + 1}"
            for idx, node in enumerate(self.column_wise_features)
        }
        column_transformer = ColumnTransformer(
            feature_name_mapping=formula_to_safe_name
        )
        column_transformer.fit(X_pd, feature_nodes=self.column_wise_features)
        return column_transformer.transform(X_pd), column_transformer

    def _build_training_frame(
        self, X_train: pl.DataFrame
    ) -> tuple[pd.DataFrame, Optional[ColumnTransformer]]:
        """Build training dataframe and return transformed data + fitted column transformer."""
        X_pd = X_train.to_pandas().copy()
        X_pd = self.row_wise_transformations.apply_row_wise_transformations(X_pd)
        X_pd = self._apply_openfe_rowwise_nodes(X_pd)
        X_pd, column_transformer = self._apply_openfe_columnwise_nodes(X_pd)
        return self._apply_categorical_dtypes(X_pd), column_transformer

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
        """Return categorical column names that are present in a dataframe.

        Args:
            X_pd (pd.DataFrame): Input dataframe.

        Returns:
            list[str]: Valid categorical column names that exist in ``X_pd``.
        """
        names: list[str] = []
        for col in self.categorical_columns:
            # If column in self.categorical_columns is string and in dataframe column, append to names
            if isinstance(col, str):
                if col in X_pd.columns:
                    names.append(col)
        return names

    def _apply_categorical_dtypes(self, X_pd: pd.DataFrame) -> pd.DataFrame:
        """Return a copy of ``X_pd`` with categorical dtypes enforced.

        Args:
            X_pd (pd.DataFrame): Input dataframe.

        Returns:
            pd.DataFrame: Copied dataframe with object/string columns cast to
                ``category`` and configured categorical columns enforced.
        """
        X_pd_copy = X_pd.copy()

        # Ensure all text-like columns are categorical for LightGBM/RFE compatibility.
        object_like_cols = X_pd_copy.select_dtypes(include=["object", "string"]).columns
        for name in object_like_cols:
            X_pd_copy[name] = X_pd_copy[name].astype("category")

        for name in self._categorical_feature_names(X_pd_copy):
            X_pd_copy[name] = X_pd_copy[name].astype("category")
        return X_pd_copy

    def _build_categorical_mapping(self, X_pd: pd.DataFrame) -> dict[str, list]:
        """Build categorical mapping from dataframe categorical columns."""
        mapping: dict[str, list] = {}
        for col in X_pd.columns:
            if pd.api.types.is_categorical_dtype(X_pd[col]):
                mapping[col] = X_pd[col].cat.categories.tolist()
        return mapping

    def _log_training_run(
        self,
        model: Union[LGBMClassifier, XGBClassifier, CatBoostClassifier],
        X_train_final: pd.DataFrame,
        y_pred_proba: np.ndarray,
        y_train_np: np.ndarray,
        run_id: str,
        categorical_mapping: dict[str, list],
        column_transformer: Optional[ColumnTransformer] = None,
    ) -> None:
        """Log artifacts/metrics and register trained model."""
        train_log_loss = log_loss(y_train_np, y_pred_proba)
        model_artifact_path = self.model_type
        self.mlflow_handler.log_model(
            model=model,
            X_data=X_train_final,
            y_pred=y_pred_proba,
            model_type=self.model_type,
            name=model_artifact_path,
        )
        mlflow.log_params(self.params)
        mlflow.set_tag("features", ",".join(map(str, self.features)))
        mlflow.set_tag("categorical_mapping", json.dumps(categorical_mapping))
        mlflow.set_tag("model_type", self.model_type)
        if column_transformer is not None:
            self.mlflow_handler.log_artifact_pickle(
                column_transformer, "fitted_column_transformer"
            )
        mlflow.log_metric("train_loss", train_log_loss)

        # Register and tag model
        model_name = f"{self.experiment_name}_{self.model_type}"
        self.mlflow_handler.register_and_tag_model(
            run_id=run_id,
            model_name=model_name,
            model_type=model_artifact_path,
            alias="production",
        )

    def train(
        self, X_train: pl.DataFrame, y_train: pl.DataFrame
    ) -> Union[LGBMClassifier, XGBClassifier, CatBoostClassifier]:
        """Train final model using params + row/column-wise OFE features + selected features."""
        self.setup_mlflow()

        X_train_pd, fitted_column_transformer = self._build_training_frame(X_train)
        X_train_pd = self._add_legacy_openfe_aliases(X_train_pd)
        y_train_np = y_train.to_numpy().ravel()

        missing_features = [f for f in self.features if f not in X_train_pd.columns]
        if missing_features:
            raise ValueError(
                f"Missing selected features after transformations: {missing_features}"
            )

        X_train_final = X_train_pd[self.features]
        categorical_mapping = self._build_categorical_mapping(X_train_final)

        with mlflow.start_run(run_name=f"{self.run_name}_{self.model_type}") as run:
            base_estimator = self.wrapper.fetch_base_estimator()
            effective_fit_params = self._effective_params(model=base_estimator)

            model = self.wrapper.fit(
                X_train=X_train_final,
                y_train=y_train_np,
                use_early_stopping=False,
                params=effective_fit_params,
                trial=None,  # No Optuna trial for final training
            )
            setattr(model, "categorical_mapping_", categorical_mapping)

            y_pred_proba = model.predict_proba(X_train_final)
            self._log_training_run(
                model=model,
                X_train_final=X_train_final,
                y_pred_proba=y_pred_proba,
                y_train_np=y_train_np,
                run_id=run.info.run_id,
                categorical_mapping=categorical_mapping,
                column_transformer=fitted_column_transformer,
            )

            mlflow.end_run()
            return model
