import polars as pl
from lightgbm import LGBMClassifier
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import optuna
import matplotlib.pyplot as plt
from utils.statics import lightgbm_model_name, tracking_uri
import mlflow
import numpy as np
import logging
from model.data_classes import OuterCVResults
from model.nested_cv_eval import ModelCVEvaluator


class ModelParamTuner(ModelCVEvaluator):

    def _setup_mlflow(self):
        """
        Sets up tracking uri and experiment for MLFlow

        Returns:
            None
        """
        self.logger.info(
            f"""Starting training with model {self.model_type} with the following configuration:
        - {self.n_inner_splits} inner splits
        - {self.n_trials} trials"""
        )
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        logging.getLogger("mlflow").setLevel(logging.ERROR)
        logging.getLogger("optuna").setLevel(logging.WARNING)

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name=self.experiment_name)

    def plot_loss_curve(self, train_loss: list, valid_loss: list) -> None:
        """Plot loss curve for training

        Args:
            train_loss (list): Training loss values
            valid_loss (list): Validation loss values
        """
        fig = plt.figure(figsize=(10, 6))
        epochs = range(1, len(train_loss) + 1)
        plt.plot(epochs, train_loss, "r", label="Training loss")
        plt.plot(epochs, valid_loss, "b", label="Validation loss")
        plt.title(f"Training and Validation Loss - {self.model_type}")
        plt.xlabel("Boosting Rounds")
        plt.ylabel("Log Loss")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.5)

        mlflow.log_figure(fig, artifact_file=f"{self.model_type}_loss_curve.png")

    def fit_and_evaluate_model(
        self,
        X_train: pl.DataFrame,
        y_train: pl.DataFrame,
        X_val: pl.DataFrame,
        y_val: pl.DataFrame,
        best_params: dict,
        selected_features: np.ndarray,
    ) -> LGBMClassifier:
        """Fit final model using best hyerparameters and selected features

        Args:
            X_train (pl.DataFrame): Training set
            y_train (pl.DataFrame): Training labels
            X_val (pl.DataFrame): Validation set
            y_val (pl.DataFrame): Validation labels
            best_params (dict): Best hyperparameters found in the outer fold
            selected_features (np.ndarray): Array of selected features from RFECV
        Returns:
            LGBMClassifier: Trained final model
        """
        self.logger.info("Fitting final model with best hyperparameters...")
        final_model = self._fetch_model()
        final_model.set_params(**best_params)

        results = {}

        if self.model_type == lightgbm_model_name:
            final_model.fit(
                X_train[selected_features],
                y_train,
                eval_set=[
                    (X_train[selected_features], y_train),
                    (X_val[selected_features], y_val),
                ],
                eval_names=["train", "valid"],
                eval_metric="binary_logloss",
                callbacks=[lgb.record_evaluation(results)],
                categorical_feature=(
                    self.categorical_columns if self.categorical_columns else ""
                ),
            )
            train_loss = results["train"]["binary_logloss"]
            valid_loss = results["valid"]["binary_logloss"]

        if train_loss and valid_loss:
            self.plot_loss_curve(train_loss, valid_loss)

        return final_model

    def tune_and_train(self, X: pl.DataFrame, y: pl.DataFrame) -> OuterCVResults:
        """Tune and train on the entire X_train set. Use one-layer CV for hyperparameter optimisation.

        Args:
            X_train (pl.DataFrame): Input features used for training
            y_train (pl.DataFrame): Ground truths used for training

        Returns:
            OuterCVResults: Outer cross-validation results
        """
        outer_cv_results = OuterCVResults()
        self.setup_mlflow()

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

        with mlflow.start_run(run_name=f"{self.run_name}_{self.model_type}") as run:

            parent_id = run.info.run_id
            outer_cv_results.parent_run_id = parent_id
            callback_fn = self._mlflow_callback(
                outer_fold_run_id=parent_id,
                experiment_name=self.experiment_name,
            )

            # 1. Perform full hyperparamter tuning
            study, best_params = self._hyperparameter_tuning(
                X_train_outer=X_train,
                y_train_outer=y_train,
                callback_fn=callback_fn,
            )

            # 2. Fit final model with best hyperparameters on the entire outer training fold
            final_model, selected_features, category_schema = self._fit_model(
                X_train_outer=X_train,
                y_train_outer=y_train,
                best_params=best_params,
            )

            # 3. Evaluate final model on the outer validation fold and log final model to model registry
            outer_fold_log_loss = self._eval_validation_loss(
                X_val_outer=X_val,
                y_val_outer=y_val,
                selected_features=selected_features,
                final_model=final_model,
                category_schema=category_schema,
            )

            self._append_and_log_metrics_and_params(
                outer_cv_results=outer_cv_results,
                selected_features=selected_features,
                outer_fold_log_loss=outer_fold_log_loss,
                best_params=best_params,
                run=run,
                study=study,
            )

        mlflow.end_run()

        return outer_cv_results
