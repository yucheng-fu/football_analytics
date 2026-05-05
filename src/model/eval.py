import logging
from typing import Optional

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import polars as pl
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

from feature_engineering.ColumnTransformer import ColumnTransformer
from feature_engineering.OpenFE.utils import tree_to_formula
from feature_engineering.RowWiseTransformations import RowWiseTransformations
from utils.statics import tracking_uri
from utils.utils import safe_production_transform


class ModelEval:
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    def __init__(
        self,
        model: LGBMClassifier,
        X_train: pl.DataFrame,
        y_train: pl.DataFrame,
        best_features: np.ndarray,
        experiment_name: str,
        row_wise_features: Optional[list] = None,
        column_wise_features: Optional[list] = None,
        row_wise_transformations: Optional[RowWiseTransformations] = None,
        categorical_columns: Optional[list] = None,
        categorical_mapping: Optional[dict[str, list]] = None,
    ):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.best_features = list(best_features)
        self.experiment_name = experiment_name
        self.row_wise_features = row_wise_features or []
        self.column_wise_features = column_wise_features or []
        self.row_wise_transformations = (
            row_wise_transformations
            if row_wise_transformations is not None
            else RowWiseTransformations()
        )
        self.categorical_columns = categorical_columns or []
        self.categorical_mapping = self._resolve_categorical_mapping(
            categorical_mapping
        )

    @property
    def model_type(self) -> str:
        return self.model.__class__.__name__

    def setup_mlflow(self) -> None:
        """Set up MLflow tracking and experiment."""
        self.logger.info(
            f"Starting evaluation for {self.model_type} with the following configuration:"
        )

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name=self.experiment_name)

    def _apply_openfe_rowwise_nodes(self, X_pd: pd.DataFrame) -> pd.DataFrame:
        if not self.row_wise_features:
            return X_pd
        return safe_production_transform(X_pd, self.row_wise_features)

    def _build_column_transformer(self) -> Optional[ColumnTransformer]:
        if not self.column_wise_features:
            return None
        formula_to_safe_name = {
            tree_to_formula(node): f"ofe_col_{idx + 1}"
            for idx, node in enumerate(self.column_wise_features)
        }
        return ColumnTransformer(feature_name_mapping=formula_to_safe_name)

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

    def _add_legacy_openfe_aliases(self, X_pd: pd.DataFrame) -> pd.DataFrame:
        if not self.row_wise_features:
            return X_pd

        X_out = X_pd.copy()
        for idx, node in enumerate(self.row_wise_features):
            legacy_name = f"autoFE_f_{idx}"
            formula_name = tree_to_formula(node)
            if legacy_name not in X_out.columns and formula_name in X_out.columns:
                X_out[legacy_name] = X_out[formula_name]
        return X_out

    def _build_eval_frame(
        self,
        X_data: pl.DataFrame,
        column_transformer: Optional[ColumnTransformer] = None,
        fit_column_transformer: bool = False,
    ) -> pd.DataFrame:
        X_pd = X_data.to_pandas().copy()
        X_pd = self.row_wise_transformations.apply_row_wise_transformations(X_pd)
        X_pd = self._apply_openfe_rowwise_nodes(X_pd)
        if column_transformer is not None:
            if fit_column_transformer:
                column_transformer.fit(X_pd, feature_nodes=self.column_wise_features)
            X_pd = column_transformer.transform(X_pd)
        X_pd = self._add_legacy_openfe_aliases(X_pd)
        return self._apply_categorical_dtypes(X_pd)

    def _resolve_categorical_mapping(
        self, categorical_mapping: Optional[dict[str, list]]
    ) -> dict[str, list]:
        """Resolve categorical mapping from explicit input, then model attribute."""
        if isinstance(categorical_mapping, dict):
            return categorical_mapping
        mapping = getattr(self.model, "categorical_mapping_", None)
        if isinstance(mapping, dict):
            return mapping
        return {}

    def _apply_saved_categorical_mapping(self, X_pd: pd.DataFrame) -> pd.DataFrame:
        """Apply saved training categorical mapping for production-safe inference."""
        if not self.categorical_mapping:
            return X_pd
        X_out = X_pd.copy()
        for col, categories in self.categorical_mapping.items():
            if col in X_out.columns:
                X_out[col] = pd.Categorical(X_out[col], categories=categories)
        return X_out

    def plot_auc_roc(
        self,
        ax: plt.Axes,
        roc_auc: float,
        fpr: np.ndarray,
        tpr: np.ndarray,
        train_roc_auc: float,
        train_fpr: np.ndarray,
        train_tpr: np.ndarray,
    ) -> None:
        ax.plot(fpr, tpr, label=f"Test ROC AUC = {roc_auc:.4f}", color="C1")
        ax.plot(
            train_fpr,
            train_tpr,
            linestyle="--",
            label=f"Train ROC AUC = {train_roc_auc:.4f}",
            color="C0",
        )
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve")
        ax.legend(loc="lower right")

    def eval(self, X_test: pl.DataFrame, y_test: pl.DataFrame) -> None:
        """Evaluate model on the test set."""
        self.setup_mlflow()

        column_transformer = self._build_column_transformer()
        X_train_pd = self._build_eval_frame(
            self.X_train,
            column_transformer=column_transformer,
            fit_column_transformer=True,
        )
        X_test_pd = self._build_eval_frame(
            X_test,
            column_transformer=column_transformer,
            fit_column_transformer=False,
        )
        y_test_np = y_test.to_numpy().ravel()
        y_train_np = self.y_train.to_numpy().ravel()

        missing_features = [f for f in self.best_features if f not in X_test_pd.columns]
        if missing_features:
            raise ValueError(
                f"Missing selected features after transformations: {missing_features}"
            )

        X_test_final = X_test_pd[self.best_features]
        X_train_final = X_train_pd[self.best_features]
        X_test_final = self._apply_saved_categorical_mapping(X_test_final)
        X_train_final = self._apply_saved_categorical_mapping(X_train_final)

        with mlflow.start_run(run_name=self.model_type):
            y_probs = self.model.predict_proba(X_test_final)[:, 1]
            y_train_probs = self.model.predict_proba(X_train_final)[:, 1]
            y_pred = (y_probs >= 0.5).astype(int)

            roc_auc = roc_auc_score(y_test_np, y_probs)
            fpr, tpr, _ = roc_curve(y_test_np, y_probs)
            train_roc_auc = roc_auc_score(y_train_np, y_train_probs)
            train_fpr, train_tpr, _ = roc_curve(y_train_np, y_train_probs)
            acc = accuracy_score(y_test_np, y_pred)
            precision = precision_score(y_test_np, y_pred)
            recall = recall_score(y_test_np, y_pred)
            f1 = f1_score(y_test_np, y_pred)
            logloss = float(log_loss(y_test_np, y_probs))

            mlflow.log_metrics(
                {
                    "roc_auc": roc_auc,
                    "accuracy": acc,
                    "precision": precision,
                    "recall": recall,
                    "f1_score": f1,
                    "log_loss": logloss,
                }
            )

            fig, ax = plt.subplots(figsize=(6, 6))
            self.plot_auc_roc(
                ax, roc_auc, fpr, tpr, train_roc_auc, train_fpr, train_tpr
            )
            mlflow.log_figure(fig, "roc_curve.png")
            plt.close(fig)

            mlflow.set_tag("alias", "production")

            self.logger.info(
                f"Test ROC AUC={roc_auc:.4f} | "
                f"Train ROC AUC={train_roc_auc:.4f} | "
                f"Acc={acc:.4f} | "
                f"Precision={precision:.4f} | Recall={recall:.4f} | F1={f1:.4f}"
                f"| Log Loss={logloss:.4f}"
            )
