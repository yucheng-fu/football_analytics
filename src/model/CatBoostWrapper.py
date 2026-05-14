import optuna
import pandas as pd
import numpy as np

from catboost import CatBoostClassifier
from lightgbm.callback import early_stopping
from optuna.integration import CatBoostPruningCallback
from typing import Dict, Any
from model.BaseModelWrapper import BaseModelWrapper


class CatBoostWrapper(BaseModelWrapper):
    def get_optuna_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        params = {
            "iterations": trial.suggest_int("iterations", 100, 1000),  # number of trees
            "depth": trial.suggest_int("depth", 4, 8),  # depth of tree
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.01, 0.2
            ),  # step size for optimisation
            "reg_lambda": trial.suggest_float(
                "reg_lambda", 1e-4, 0.3, log=True
            ),  # L2 regularisation
            "eval_metric": "Logloss",  # evaluation metric
            "random_seed": self.seed,  # seed for reproducibility
            "verbose": False,  # suppress warnings and info
        }

        return params

    def fetch_base_estimator(self, params: Dict[str, Any] = None) -> CatBoostClassifier:
        return CatBoostClassifier(verbose=False, random_state=self.seed, **params)

    def fit(
        self,
        X_train,
        y_train,
        X_val=None,
        y_val=None,
        use_early_stopping=False,
        params=None,
        trial=None,
    ):
        self.model = self.fetch_base_estimator(params=params)
        cat_cols = X_train.select_dtypes(include=["category"]).columns.tolist()

        callbacks = [CatBoostPruningCallback(trial, "Logloss")] if trial else None

        self.model.fit(
            X_train,
            y_train,
            eval_set=(X_val, y_val) if X_val is not None else None,
            cat_features=cat_cols,
            early_stopping_rounds=(
                self.early_stopping_rounds if use_early_stopping else None
            ),
            callbacks=callbacks,
        )

    @property
    def best_iteration(self):
        return self.model.get_best_iteration()
