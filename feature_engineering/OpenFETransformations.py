from typing import Tuple, List
import pandas as pd
import numpy as np
from feature_engineering.OpenFE.utils import tree_to_formula, transform
from feature_engineering.OpenFE.openfe import OpenFE


class OpenFETransformations:
    def __init__(self, n_features: int):
        self.n_features = n_features

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        task: str,
        categorical_features: List[str],
        feature_boosting: bool,
        n_jobs: int,
    ) -> Tuple[np.ndarray, np.ndarray, OpenFE]:
        ofe = OpenFE()
        features = ofe.fit(
            data=X_train,
            label=y_train,
            categorical_features=categorical_features,
            task=task,
            feature_boosting=feature_boosting,
            n_jobs=n_jobs,
        )

        selected_features = features[: self.n_features]

        row_wise_features = []
        column_wise_features = []

        for feature in selected_features:
            if feature.is_rowwise:
                row_wise_features.append(feature)
            else:
                column_wise_features.append(feature)

        return row_wise_features, column_wise_features, ofe

    def apply_openfe_features(
        self,
        features: np.ndarray,
        X_train: pd.DataFrame,
        X_val: pd.DataFrame,
        n_jobs=-1,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
        """
        Applies OpenFE transformations, keeping generic column names (auto_fe_1, etc.)
        to ensure compatibility with LightGBM, and returns a formula mapping.
        """
        if len(features) == 0:
            return X_train, X_val, {}

        selected_features = features[: self.n_features]
        original_cols = X_train.columns.tolist()

        # 1. Transform
        X_train_transformed, X_val_transformed = transform(
            X_train=X_train,
            X_test=X_val,
            new_features_list=selected_features,
            n_jobs=n_jobs,
        )

        generated_cols = [
            c for c in X_train_transformed.columns if c not in original_cols
        ]

        mapping = {
            gen_col: tree_to_formula(f)
            for gen_col, f in zip(generated_cols, selected_features)
        }

        return X_train_transformed, X_val_transformed, mapping
