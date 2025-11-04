import polars as pl
from typing import List
import mlflow
from utils.statics import xgboost_model_name, lightgbm_model_name, tracking_uri
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import log_loss
from mlflow.models import infer_signature
import logging
import numpy as np


class ModelTrainer:

    logger = logging.getLogger(__name__)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    def __init__(
        self,
        model_type: str,
        best_params: dict,
        best_features: List[str],
        run_name: str = "baseline_training",
        experiment_name: str = "Final models",
    ):
        self.model_type = model_type
        self.best_params = best_params
        self.best_features = best_features
        self.run_name = run_name
        self.experiment_name = experiment_name

    def setup_mlflow(self):
        self.logger.info(
            f"""Starting training with model {self.model_type} with the following configuration:
        - {self.n_inner_splits} inner splits
        - {self.n_outer_splits} outer splits
        - {self.n_trials} trials"""
        )

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name=self.experiment_name)

    def set_params(self, model: XGBClassifier | LGBMClassifier):
        accepted_params = set(model.get_params().keys())
        valid_params = {
            k: v for k, v in self.best_params.items() if k in accepted_params
        }
        invalid_params = [
            k for k in self.best_params.keys() if k not in accepted_params
        ]
        if invalid_params:
            raise ValueError(
                f"Invalid parameter(s) for {type(model).__name__}: {', '.join(invalid_params)}"
            )
        else:
            model.set_params(**valid_params)
            return model

    def fetch_model(self) -> XGBClassifier | LGBMClassifier:
        """
        Return a fresh estimator instance for the configured model type.

        Returns:
            XGBClassifier | LGBMClassifier: An uninitialized classifier corresponding to self.model_type.

        Raises:
            ValueError: If self.model_type is not a supported model name.
        """
        if self.model_type == xgboost_model_name:
            return XGBClassifier()
        elif self.model_type == lightgbm_model_name:
            return LGBMClassifier(verbose=-1)

        raise ValueError(f"Unsupported model type: {self.model_type}")

    def log_model(
        self,
        final_model: XGBClassifier | LGBMClassifier,
        X_data: pl.DataFrame,
        output: np.ndarray,
    ) -> None:
        """Log model in outer fold run

        Args:
            final_model (XGBClassifier | LGBMClassifier): Trained model
            X_data (pl.DataFrame): One outer fold of X
            output (np.ndarray): Predictions from final_model

        Raises:
            ValueError: If self.model_type is not a supported model name.
        """
        X_data_pd = X_data.to_pandas()
        signature = infer_signature(X_data_pd, output)
        input_example = X_data_pd.head(10)

        log_fn_mapping = {
            xgboost_model_name: mlflow.xgboost.log_model,
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

    def train(self, X_train: pl.DataFrame, y_train) -> XGBClassifier | LGBMClassifier:
        self.setup_mlflow()
        with mlflow.start_run(run_name=f"{self.run_name}_{self.model_type}") as run:
            model = self.fetch_model()
            model = self.set_params(model=model)

            X_train_final = X_train[self.best_features]
            model.fit(X_train_final, y_train)

            y_pred_proba = model.predict_proba(X_train_final)
            train_log_loss = log_loss(y_train, y_pred_proba)

            self.log_model(model=model, X_data=X_train_final, output=y_pred_proba)
            mlflow.log_params(self.best_params)
            mlflow.set_tag("features", ",".join(map(str, self.best_features)))
            mlflow.log_metric("train_loss", train_log_loss)
            mlflow.register_model(
                model_uri=f"runs:/{run.info.run_id}/model",
                name=f"{self.experiment_name}_{self.model_type}",
            )

            return model
