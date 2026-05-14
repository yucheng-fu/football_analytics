import optuna
import pandas as pd
import numpy as np

from xgboost import XGBClassifier
from lightgbm.callback import early_stopping
from optuna.integration import XGBoostPruningCallback
from typing import Dict, Any
from model.BaseModelWrapper import BaseModelWrapper


class XGBoostWrapper(BaseModelWrapper):
    def get_optuna_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        params = {
            "n_estimators": trial.suggest_int(
                "n_estimators", 100, 1000
            ),  # number of trees
            "max_depth": trial.suggest_int(
                "max_depth", 3, 10
            ),  # maximum depth of a tree
            "grow_policy": "depthwise",  # depthwise is default.
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.01, 0.2
            ),  # step size for optimisation
            "subsample": trial.suggest_float(
                "subsample", 0.5, 1.0
            ),  # fraction of samples to be used for each tree
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float(
                "reg_alpha", 1e-4, 0.1, log=True
            ),  # L1 regularisation
            "reg_lambda": trial.suggest_float(
                "reg_lambda", 1e-4, 0.3, log=True
            ),  # L2 regularisation
            "eval_metric": "logloss",  # evaluation metric
            "random_state": self.seed,  # seed for reproducibility
            "verbosity": 0,  # 0 for silent, 1 for warning
            "tree_method": "hist",  # Use histogram-based algorithm for faster training on large datasets (default)
        }
        return params

    def fetch_base_estimator(self, params: Dict[str, Any] = None) -> XGBClassifier:
        return XGBClassifier(verbosity=0, random_state=self.seed, **params)

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
        callbacks = (
            [XGBoostPruningCallback(trial, "validation_0-logloss")] if trial else None
        )

        self.model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)] if X_val is not None else None,
            early_stopping_rounds=(
                self.early_stopping_rounds if use_early_stopping else None
            ),
            callbacks=callbacks,
            verbose=False,
        )

    @property
    def best_iteration(self):
        return getattr(self.model, "best_iteration", None)
