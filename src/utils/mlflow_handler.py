import os
import pickle
import tempfile
from typing import Any, Union, Optional
import logging

import mlflow
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient

import utils.statics as statics


class MLflowHandler:
    """Encapsulates MLflow operations for model training and evaluation."""

    def __init__(
        self,
        tracking_uri: str = statics.tracking_uri,
        experiment_name: str = "Model selection and hyperparameter tuning",
        logger: Optional[logging.Logger] = None,
    ):
        """Initialize MLflow handler.

        Args:
            tracking_uri (str): MLflow tracking URI.
            experiment_name (str): MLflow experiment name.
            logger (Optional[logging.Logger]): Logger instance for info messages.
        """
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self.logger = logger or logging.getLogger(__name__)

    def setup(self) -> None:
        """Configure MLflow tracking URI, experiment, and logging verbosity."""
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(experiment_name=self.experiment_name)
        logging.getLogger("mlflow").setLevel(logging.ERROR)
        logging.getLogger("kaleido").setLevel(logging.WARNING)
        logging.getLogger("choreographer").setLevel(logging.WARNING)

    def log_artifact_pickle(
        self, obj: Any, name: str, artifact_path: str = "pickles"
    ) -> None:
        """Serialize an object to a pickle file and log it to MLflow.

        Args:
            obj (Any): Python object to serialize.
            name (str): Artifact name stem (without .pkl extension).
            artifact_path (str): Artifact path within MLflow. Defaults to "pickles".
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_file_path = os.path.join(tmp_dir, f"{name}.pkl")
            with open(tmp_file_path, "wb") as f:
                pickle.dump(obj, f)
            mlflow.log_artifact(
                local_path=tmp_file_path,
                artifact_path=artifact_path,
            )

    def log_model(
        self,
        model: Union[LGBMClassifier, XGBClassifier, CatBoostClassifier],
        X_data: pd.DataFrame,
        y_pred: np.ndarray,
        model_type: str,
        name: str = "model",
    ) -> None:
        """Log a trained model to MLflow with signature inference.

        Args:
            model (Union[LGBMClassifier, XGBClassifier, CatBoostClassifier]): Trained model.
            X_data (pd.DataFrame): Sample input data for signature inference.
            y_pred (np.ndarray): Sample predictions for signature inference.
            model_type (str): Model type name (e.g., "lightgbm", "xgboost", "catboost").
            name (str): Model artifact name. Defaults to "model".

        Raises:
            ValueError: If model_type is not supported.
        """
        signature = infer_signature(X_data, y_pred)
        input_example = X_data.head(10)

        log_fn_mapping = {
            statics.lightgbm_model_name: mlflow.lightgbm.log_model,
            statics.xgboost_model_name: mlflow.xgboost.log_model,
            statics.catboost_model_name: mlflow.catboost.log_model,
        }

        log_fn = log_fn_mapping.get(model_type)
        if log_fn is None:
            raise ValueError(f"Unsupported model type: {model_type}")

        log_fn(
            model,
            name=name,
            signature=signature,
            input_example=input_example,
        )

    def log_figure(self, fig, name: str) -> None:
        """Log a Matplotlib figure to MLflow.

        Args:
            fig: Matplotlib figure object.
            name (str): Figure name stem (without file extension).
        """
        mlflow.log_figure(fig, f"plots/{name}.png")

    def register_and_tag_model(
        self,
        run_id: str,
        model_name: str,
        model_type: str,
        alias: str = "production",
    ) -> int:
        """Register a model, set its alias, and return the version number.

        Args:
            run_id (str): MLflow run ID where model was logged.
            model_name (str): Name for the registered model.
            model_type (str): Model type (e.g., "lightgbm") used as artifact name in MLflow.
            alias (str): Alias to assign (e.g., "production"). Defaults to "production".

        Returns:
            int: Version number of the registered model.
        """
        registered_model = mlflow.register_model(
            model_uri=f"runs:/{run_id}/{model_type}",
            name=model_name,
        )
        version = registered_model.version

        client = MlflowClient()
        client.set_registered_model_alias(
            name=model_name,
            alias=alias,
            version=version,
        )
        client.set_model_version_tag(
            name=model_name,
            version=version,
            key="alias",
            value=alias,
        )

        return version
