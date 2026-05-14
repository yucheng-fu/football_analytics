from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import optuna
from typing import Tuple, Dict, Any, Optional


class BaseModelWrapper(ABC):
    def __init__(self, seed: int, early_stopping_rounds: int = 20):
        self.seed = seed
        self.early_stopping_rounds = early_stopping_rounds
        self.model = None

    @abstractmethod
    def get_optuna_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Return model specific hyperparameters for Optuna tuning.

        Args:
            trial (Optuna.Trial): Optuna trial object

        Returns:
            Dict[str, Any]: Dictionary of hyperparameters to be used for model training
        """
        pass

    @abstractmethod
    def fetch_base_estimator(self) -> Any:
        """Return a base estimator of the model type"""
        pass

    @abstractmethod
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[np.ndarray] = None,
        use_early_stopping: bool = False,
        params: Optional[Dict[str, Any]] = None,
        trial: Optional[optuna.Trial] = None,
    ) -> None:
        """Fit model and use early stopping if specified.

        Args:
            X_train (pd.DataFrame): Training features
            y_train (np.ndarray): Training labels
            X_val (Optional[pd.DataFrame], optional): Validation features. Defaults to None.
            y_val (Optional[np.ndarray], optional): Validation labels. Defaults to None.
            use_early_stopping (bool, optional): Whether to use early stopping. Defaults to False.
            params (Optional[Dict[str, Any]], optional): Model parameters. Defaults to None.
            trial (Optional[optuna.Trial], optional): Optuna trial object. Defaults to None.
        """
        pass

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Make prediction with probability estimates

        Args:
            X (pd.DataFrame): Input features for prediction

        Returns:
            np.ndarray: Predicted probabilities for each class
        """
        return self.model.predict_proba(X)

    @property
    @abstractmethod
    def best_iteration(self) -> Optional[int]:
        """
        Return the best iteration for early stopping if applicable."""
        pass
