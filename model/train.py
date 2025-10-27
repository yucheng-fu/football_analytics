from typing import Tuple

import polars as pl
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.feature_selection import RFECV
from sklearn.metrics import log_loss
import optuna.trial as trial
import optuna
from utils.statics import (
    xgboost_model_name,
    lightgbm_model_name,
    france_argentina_match_id,
)
import mlflow
import numpy as np


class ModelTrainer:

    def __init__(self, model_type: str):
        self.model_type = model_type

    def set_params(
        self, model: XGBClassifier | LGBMClassifier, params: dict
    ) -> XGBClassifier | LGBMClassifier:
        return model.set_params(**params)

    def fetch_model(self) -> XGBClassifier | LGBMClassifier:
        if self.model_type == xgboost_model_name:
            return XGBClassifier()
        elif self.model_type == lightgbm_model_name:
            return LGBMClassifier()

        raise ValueError(f"Unsupported model type: {self.model_type}")

    def fetch_param_suggestions(self, trial: trial) -> dict:
        if self.model_type == xgboost_model_name:
            return {
                "n_estimators": trial.suggest_int(
                    "n_estimators", 50, 1000
                ),  # number of trees
                "max_depth": trial.suggest_int(
                    "max_depth", 3, 16
                ),  # maximum depth of each tree
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.01, 0.3
                ),  # step size for optimisation
                "subsample": trial.suggest_float(
                    "subsample", 0.5, 1.0
                ),  # fraction of samples to be used for each tree
                "alpha": trial.suggest_float("alpha", 0.0, 1.0),  # L1 regularisation
                "lambda": trial.suggest_float("lambda", 0.0, 1.0),  # L2 regularisation
                "eval_metric": "logloss",  # evaluation metric
                "random_state": 165,  # seed for reproducibility
            }

        elif self.model_type == lightgbm_model_name:
            return {
                "n_estimators": trial.suggest_int(
                    "n_estimators", 50, 1000
                ),  # number of trees
                "max_depth": trial.suggest_int(
                    "max_depth", 3, 16
                ),  # maximum depth of each tree
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.01, 0.3
                ),  # step size for optimisation
                "subsample": trial.suggest_float(
                    "subsample", 0.5, 1.0
                ),  # fraction of samples to be used for each tree
                "reg_alpha": trial.suggest_float(
                    "reg_alpha", 0.0, 1.0
                ),  # L1 regularisation
                "reg_lambda": trial.suggest_float(
                    "reg_lambda", 0.0, 1.0
                ),  # L2 regularisation
                "metric": "binary_logloss",  # evaluation metric
                "random_state": 165,  # seed for reproducibility
                "verbose": -1,  # suppress warnings and info
            }

        raise ValueError(f"Unsupported model type: {self.model_type}")

    def _objective(self, trial: optuna.Trial, X_train_outer, y_train_outer):
        # fetch params for the trial based on the model
        params = self.fetch_param_suggestions(trial)
        base_estimator = self.fetch_model()
        base_estimator.set_params(**params)

        inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        rfecv = RFECV(
            estimator=base_estimator,
            step=1,
            cv=inner_cv,
            scoring="neg_log_loss",
            n_jobs=-1,
        )

        # perform inner CV and get mean_score of best subset of features
        rfecv.fit(X_train_outer, y_train_outer)
        mean_score = np.max(
            rfecv.cv_results_["mean_test_score"]
        )  # max to get the score of the max neg_log_loss

        # log to mlflow
        with mlflow.start_run(nested=True, run_name="Inner_fold"):
            mlflow.log_params(params)
            mlflow.log_metric("neg_log_loss", mean_score)

        return mean_score

    def tune_and_train(
        self, X_train: pl.DataFrame, y_train: pl.DataFrame
    ) -> Tuple[dict, np.ndarray]:

        print("Starting training with model {}")

        mlflow.set_tracking_uri("http://127.0.0.1:8080/")
        mlflow.set_experiment("Model selection and hyperparameter tuning")
        with mlflow.start_run() as run:

            outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            outer_cv_scores = []
            outer_cv_params = []
            outer_cv_features = []
            for i, (train_idx, val_idx) in enumerate(outer_cv.split(X_train, y_train)):
                X_train_outer, X_val_outer = X_train[train_idx], X_train[val_idx]
                y_train_outer, y_val_outer = y_train[train_idx], y_train[val_idx]

                study = optuna.create_study(direction="maximize")

                study.optimize(
                    lambda trial: self._objective(trial, X_train_outer, y_train_outer),
                    n_trials=30,
                    n_jobs=-1,
                    show_progress_bar=True,
                )

                best_params = study.best_params

                # Fit model on outer training fold
                final_estimator = self.fetch_model()
                final_estimator.set_params(**best_params)

                final_rfe_cv = StratifiedKFold(
                    n_splits=5, shuffle=True, random_state=42
                )
                rfecv = RFECV(
                    estimator=final_estimator,
                    step=1,
                    cv=final_rfe_cv,
                    scoring="neg_log_loss",
                    n_jobs=-1,
                )
                pipeline = Pipeline(steps=[("feature_selection", rfecv)])
                pipeline.fit(X_train_outer, y_train_outer)

                selected_feature_mask = rfecv.get_support(indices=True)
                selected_features = X_train.columns[selected_feature_mask]

                # predict
                final_model = rfecv.estimator_
                y_pred_proba = final_model.predict_proba(X_val_outer[selected_features])

                outer_fold_log_loss = log_loss(y_val_outer, y_pred_proba)
                outer_cv_scores.append(outer_fold_log_loss)
                outer_cv_params.append(best_params)
                outer_cv_features.append(selected_feature_mask)

                with mlflow.start_run(nested=True, run_name=f"Outer_fold_{i+1}"):
                    # Log the final model
                    mlflow.log_param(
                        "outer_selected_features", ",".join(map(str, selected_features))
                    )
                    mlflow.log_metric("outer_log_loss", outer_fold_log_loss)
                    mlflow.log_params("params", best_params)
                    mlflow.log_model(final_model, "model")

        #
        best_idx = np.argmax(outer_cv_scores)

        return (outer_cv_params[best_idx], outer_cv_features[best_idx])


if __name__ == "__main__":
    pass
