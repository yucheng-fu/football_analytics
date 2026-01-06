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
from sklearn.metrics import log_loss
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

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name=self.experiment_name)

    def plot_auc_roc(
        self,
        ax: plt.Axes,
        roc_auc: float,
        fpr: np.ndarray,
        tpr: np.ndarray,
        train_roc_auc: np.ndarray,
        train_fpr: np.ndarray,
        train_tpr: np.ndarray,
    ) -> None:
        # Plot test ROC
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

    def compute_log_loss(self, y: pl.DataFrame, y_probs: pl.DataFrame) -> float:
        y_arr = np.ravel(y)
        probs = np.ravel(y_probs)
        return float(log_loss(y_arr, probs))

    def eval(self, X_test: pl.DataFrame, y_test: pl.DataFrame) -> None:
        """Evaluate model on the test set"""
        self.setup_mlflow()

        X_test = X_test.select(self.best_features).to_numpy()
        X_train = self.X_train.select(self.best_features).to_numpy()
        y_test = y_test.to_numpy().ravel()
        y_train = self.y_train.to_numpy().ravel()

        with mlflow.start_run(run_name=self.model_type):
            # Probabilities and predictions
            y_probs = self.model.predict_proba(X_test)[:, 1]
            y_train_probs = self.model.predict_proba(X_train)[:, 1]
            y_pred = (y_probs >= 0.5).astype(int)

            # Metrics
            roc_auc, fpr, tpr, _ = self.compute_roc_curve(y_test, y_probs)
            train_roc_auc, train_fpr, train_tpr, _ = self.compute_roc_curve(
                y_train, y_train_probs
            )
            acc = self.compute_accuracy(y_test, y_pred)
            precision = self.compute_precision(y_test, y_pred)
            recall = self.compute_recall(y_test, y_pred)
            f1 = self.compute_f1_score(y_test, y_pred)
            logloss = self.compute_log_loss(y_test, y_probs)

            # Log metrics
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

            # Plot ROC
            fig, ax = plt.subplots(figsize=(6, 6))
            self.plot_auc_roc(
                ax, roc_auc, fpr, tpr, train_roc_auc, train_fpr, train_tpr
            )
            mlflow.log_figure(fig, "roc_curve.png")
            plt.close(fig)

            self.logger.info(
                f"Test ROC AUC={roc_auc:.4f}"
                f"Train ROC AUC={train_roc_auc:.4f} | "
                f"| Acc={acc:.4f} | "
                f"Precision={precision:.4f} | Recall={recall:.4f} | F1={f1:.4f}"
                f"| Log Loss={logloss:.4f}"
            )
