import polars as pl
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.feature_selection import RFECV
import optuna.trial as trial
import optuna
from utils.statics import xgboost_model_name, lightgbm_model_name
import mlflow


class XGBoostTrainer:

    def __init__(self, X_train: pl.DataFrame, y_train: pl.DataFrame, model_type: str):
        self.X_train = X_train
        self.y_train = y_train
        self.model_type = model_type
        self.best_model_params = None
        self.tuning_params = None

    def set_params(self, model: XGBClassifier):
        return model.set_params(**self.best_model_params)

    def fetch_param_suggestions(self) -> dict:
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

    # def tune(self, )

    def _objective(self, trial: trial.Trial, X_train_outer, y_train_outer):

        params = self.fetch_param_suggestions()

        inner_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

        scores = []

        with mlflow.start_run(nested=True):

            for train_idx, val_idx in inner_cv.split(X_train_outer, y_train_outer):
                X_inner_train, X_inner_val = (
                    X_train_outer[train_idx],
                    X_train_outer[val_idx],
                )
                y_inner_train, y_inner_val = (
                    y_train_outer[train_idx],
                    y_train_outer[val_idx],
                )

                base_estimater = XGBClassifier(**params)

                model = Pipeline(steps=[("xgboost", base_estimater)])
                model.fit(X_inner_train, y_inner_train)
                score = cross_val_score(model, X_inner_val, y_inner_val, cv=inner_cv)
                scores.append(score.mean())

                # log
                mlflow.log_params(params)

            return scores

    def train(self, X_train, y_train):

        with mlflow.start_run() as run:

            outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            for i, (train_idx, val_idx) in enumerate(outer_cv.split(X_train, y_train)):
                X_train_outer, X_val_outer = X_train[train_idx], X_train[val_idx]
                y_train_outer, y_val_outer = y_train[train_idx], y_train[val_idx]

                study = optuna.create_study(direction="minimize")

                study.optimize(
                    lambda trial: self._objective(trial, X_train_outer, y_train_outer),
                    n_trials=30,
                    n_jobs=-1,
                )
