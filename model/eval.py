import polars as pl
import logging
import mlflow
from utils.statics import tracking_uri
import numpy as np
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from typing import Tuple
import matplotlib.pyplot as plt


class ModelEval:
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    def __init__(
        self,
        model: XGBClassifier | LGBMClassifier,
        X_train: pl.DataFrame,
        y_train: pl.DataFrame,
        best_features: np.ndarray,
        experiment_name: str,
    ):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.best_features = best_features
        self.experiment_name = experiment_name

    @property
    def model_type(self) -> str:
        return self.model.__class__.__name__

    def setup_mlflow(self) -> None:
        """
        Sets up tracking uri and experiment for MLFlow
        """
        self.logger.info(
            f"""Starting evaluation for {self.model_type} with the following configuration: """
        )

        self.model.__cla

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name=self.experiment_name)

    def plot_auc_roc(
        self,
        ax: plt.Axes,
        roc_auc: float,
        fpr: np.ndarray,
        tpr: np.ndarray,
    ) -> None:
        ax.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.4f}")
        ax.plot([0, 1], [0, 1], linestyle="--")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve")
        ax.legend(loc="lower right")

    def compute_roc_curve(self, y: pl.DataFrame, y_probs: pl.DataFrame) -> Tuple:
        roc_auc = roc_auc_score(y, y_probs)

        fpr, tpr, threshold = roc_curve(y, y_probs)

        return roc_auc, fpr, tpr, threshold

    def compute_accuracy(self, y: pl.DataFrame, y_pred: pl.DataFrame) -> float:
        return accuracy_score(y, y_pred)

    def compute_precision(self, y: pl.DataFrame, y_pred: pl.DataFrame):
        return precision_score(y, y_pred)

    def compute_recall(self, y: pl.DataFrame, y_pred: pl.DataFrame):
        return recall_score(y, y_pred)

    def compute_f1_score(self, y: pl.DataFrame, y_pred: pl.DataFrame):
        return f1_score(y, y_pred)

    def eval(self, X_test: pl.DataFrame, y_test: pl.DataFrame) -> None:
        """Evaluate model on the test set"""
        self.setup_mlflow()

        X = X_test.select(self.best_features).to_numpy()
        y = y_test.to_numpy().ravel()

        with mlflow.start_run(run_name=self.model_type):
            # Probabilities and predictions
            y_probs = self.model.predict_proba(X)[:, 1]
            y_pred = (y_probs >= 0.5).astype(int)

            # Metrics
            roc_auc, fpr, tpr, _ = self.compute_roc_curve(y, y_probs)
            acc = self.compute_accuracy(y, y_pred)
            precision = self.compute_precision(y, y_pred)
            recall = self.compute_recall(y, y_pred)
            f1 = self.compute_f1_score(y, y_pred)

            # Log metrics
            mlflow.log_metrics(
                {
                    "roc_auc": roc_auc,
                    "accuracy": acc,
                    "precision": precision,
                    "recall": recall,
                    "f1_score": f1,
                }
            )

            # Plot ROC
            fig, ax = plt.subplots(figsize=(6, 6))
            self.plot_auc_roc(ax, roc_auc, fpr, tpr)
            mlflow.log_figure(fig, "roc_curve.png")
            plt.close(fig)

            self.logger.info(
                f"ROC AUC={roc_auc:.4f} | Acc={acc:.4f} | "
                f"Precision={precision:.4f} | Recall={recall:.4f} | F1={f1:.4f}"
            )
