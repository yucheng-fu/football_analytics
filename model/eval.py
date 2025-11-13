import polars as pl
import logging
import mlflow
from utils.statics import tracking_uri
import numpy as np
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


class ModelEval:
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    def __init__(
        self,
        model: XGBClassifier | LGBMClassifier,
        best_features: np.ndarray,
        experiment_name: str,
    ):
        self.model = model
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

    def eval(self, X_test: pl.DataFrame, y_test: pl.DataFrame) -> None:
        """Evaluate model on the test set

        Args:
            X_test (pl.DataFrame): _description_
            y_test (pl.DataFrame): _description_
        """
        X = X_test[self.best_features]

        pred_proba = self.model.predict_proba(X)
