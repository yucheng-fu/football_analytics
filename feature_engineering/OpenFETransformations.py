from typing import Tuple, List
import pandas as pd
import numpy as np
from feature_engineering.utils import tree_to_formula, transform
from feature_engineering.openfe import OpenFE


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
    ) -> Tuple[np.ndarray, OpenFE]:
        ofe = OpenFE()
        features = ofe.fit(
            data=X_train,
            label=y_train,
            categorical_features=categorical_features,
            task=task,
            feature_boosting=feature_boosting,
            n_jobs=n_jobs,
        )

        return features, ofe

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
        selected_features = features[: self.n_features]
        original_cols = X_train.columns.tolist()

        # 1. Transform
        X_train_transformed, X_val_transformed = transform(
            X_train=X_train,
            X_test=X_val,
            new_features_list=selected_features,
            n_jobs=n_jobs,
        )

        # 2. Identify the new columns created by OpenFE (e.g., 'auto_fe_1')
        generated_cols = [
            c for c in X_train_transformed.columns if c not in original_cols
        ]

        # 3. Create the map: { 'auto_fe_1': '(Age + Fare)', ... }
        # This allows you to look up what 'auto_fe_1' actually means later.
        mapping = {
            gen_col: tree_to_formula(f)
            for gen_col, f in zip(generated_cols, selected_features)
        }

        # We return the transformed DFs as-is (no renaming) plus the mapping
        return X_train_transformed, X_val_transformed, mapping
