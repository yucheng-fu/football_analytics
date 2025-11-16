import polars as pl
import logging
import mlflow
from utils.statics import tracking_uri
import numpy as np
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, roc_curve
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

    def setup_mlflow(self) -> None:
        """
        Sets up tracking uri and experiment for MLFlow
        """
        self.logger.info(
            f"""Starting evaluation for {self.model_type} with the following configuration: """
        )

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name=self.experiment_name)

    def plot_auc_roc(self, ax: plt.Axes, axroc_auc, fpr, tpr, threshold) -> None:
        pass

    def compute_roc_curve(self, y: pl.DataFrame, y_probs: pl.DataFrame) -> Tuple:
        roc_auc = roc_auc_score(y, y_probs)

        fpr, tpr, threshold = roc_curve(y, y_probs)

        return roc_auc, fpr, tpr, threshold

    def eval(self, X_test: pl.DataFrame, y_test: pl.DataFrame) -> None:
        """Evaluate model on the test set

        Args:
            X_test (pl.DataFrame): _description_
            y_test (pl.DataFrame): _description_
        """
        X = X_test[self.best_features]

        y_probs = self.model.predict_proba(X)
