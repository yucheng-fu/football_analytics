import optuna
import pandas as pd
import numpy as np

from lightgbm import LGBMClassifier
from lightgbm.callback import early_stopping
from optuna.integration import LightGBMPruningCallback
from typing import Dict, Any
from model.BaseModelWrapper import BaseModelWrapper


class LGBMWrapper(BaseModelWrapper):
    def get_optuna_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        params = {
            "n_estimators": trial.suggest_int(
                "n_estimators", 100, 1000
            ),  # number of trees
            "num_leaves": trial.suggest_int(
                "num_leaves", 16, 256
            ),  # number of leaves in one tree
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.01, 0.2
            ),  # step size for optimisation
            "subsample": trial.suggest_float(
                "subsample", 0.5, 1.0
            ),  # fraction of samples to be used for each tree
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree", 0.5, 1.0
            ),  # fraction of features used per tree
            "reg_alpha": trial.suggest_float(
                "reg_alpha", 1e-4, 0.1, log=True
            ),  # L1 regularisation
            "reg_lambda": trial.suggest_float(
                "reg_lambda", 1e-4, 0.3, log=True
            ),  # L2 regularisation
        }
        return params

    def fetch_base_estimator(self, params: Dict[str, Any] = None) -> LGBMClassifier:
        return LGBMClassifier(
            metric="binary_logloss",
            verbose=-1,
            importance_type="gain",
            random_state=self.seed,
            **params,
        )

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: pd.DataFrame = None,
        y_val: np.ndarray = None,
        use_early_stopping: bool = False,
        params: Dict[str, Any] = None,
        trial: optuna.Trial = None,
    ):
        model = self.fetch_base_estimator(params=params)
        cat_cols = X_train.select_dtypes(include=["category"]).columns.tolist()

        callbacks = []
        if trial:
            callbacks.append(LightGBMPruningCallback(trial, "binary_logloss"))
        if use_early_stopping and X_val is not None:
            callbacks.append(
                early_stopping(
                    stopping_rounds=self.early_stopping_rounds, verbose=False
                )
            )

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)] if X_val is not None else None,
            eval_metric="binary_logloss",
            callbacks=callbacks or None,
            categorical_feature=cat_cols or "auto",
        )

        return model
