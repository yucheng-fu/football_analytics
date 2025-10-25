import polars as pl
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
import mlflow


class XGBoostTrainer:

    def __init__(
        self, model_params: dict, X_train: pl.DataFrame, y_train: pl.DataFrame
    ):
        self.model_params = model_params
        self.X_train = X_train
        self.y_train = y_train

    def set_params(self):
        self.model.set_params(**self.model_params)

    def train(self, X_train, y_train):
        outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        for i, (train_idx, val_idx) in enumerate(outer_cv.split(X_train, y_train)):
            X_train_outer, X_val_outer = X_train[train_idx], X_train[val_idx]
            y_train_outer, y_val_outer = y_train[train_idx], y_train[val_idx]

            inner_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

            for j, (inner_train_idx, inner_val_idx) in enumerate(
                inner_cv.split(X_train_outer, y_train_outer)
            ):
                X_inner_train, X_inner_val = (
                    X_train_outer[inner_train_idx],
                    X_train_outer[inner_val_idx],
                )
                y_inner_train, y_inner_val = (
                    y_train_outer[inner_train_idx],
                    y_train_outer[inner_val_idx],
                )

                self.model.fit(X_inner_train, y_inner_train)
                score = cross_val_score(
                    self.model, X_inner_val, y_inner_val, cv=inner_cv
                )
                print(f"Outer fold {i}, Inner fold {j}, Score: {score.mean()}")
