from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer

from api.schemas.request import InferenceRequest
from feature_engineering.OpenFE.openfe import tree_to_formula
from feature_engineering.RowWiseTransformations import RowWiseTransformations
from utils.inference_utils import apply_saved_categorical_mapping, safe_production_transform


class InferenceFrameService:
    """Service for building model-ready inference frames."""

    def __init__(self, bundle: dict[str, Any]) -> None:
        """Initialize the inference frame service.

        Args:
            bundle (dict[str, Any]): Loaded inference bundle.
        """
        self.row_wise_features = bundle.get("row_wise_features")
        self.column_wise_features = bundle.get("column_wise_features")
        self.column_transformer = bundle.get("fitted_column_transformer")
        self.best_features = bundle.get("best_features", bundle.get("selected_features"))
        self.categorical_mapping = bundle.get("categorical_mapping")

    def build_from_payload(self, payload: InferenceRequest) -> pd.DataFrame:
        """Build inference frame from request payload.

        Args:
            payload (InferenceRequest): Inference payload.

        Returns:
            pd.DataFrame: Model-ready inference frame.
        """
        input_df = pd.DataFrame([payload.model_dump(mode="json")])
        return self.build_frame(
            X_pd=input_df,
            row_wise_features=self.row_wise_features,
            column_wise_features=self.column_wise_features,
            column_transformer=self.column_transformer,
            best_features=self.best_features,
            categorical_mapping=self.categorical_mapping,
        )

    @staticmethod
    def build_frame(
        X_pd: pd.DataFrame,
        row_wise_features: list | None = None,
        column_wise_features: list | None = None,
        column_transformer: ColumnTransformer | None = None,
        best_features: np.ndarray | list[str] | None = None,
        categorical_mapping: dict[str, list] | None = None,
        row_wise_transformations: RowWiseTransformations | None = None,
    ) -> pd.DataFrame:
        """Build inference frame using eval-equivalent preprocessing steps.

        Args:
            X_pd (pd.DataFrame): Raw input frame.
            row_wise_features (list | None, optional): OpenFE row-wise nodes.
            column_wise_features (list | None, optional): OpenFE column-wise nodes.
            column_transformer (ColumnTransformer | None, optional): Fitted transformer.
            best_features (np.ndarray | list[str] | None, optional): Selected features.
            categorical_mapping (dict[str, list] | None, optional): Categorical map.
            row_wise_transformations (RowWiseTransformations | None, optional):
                Row-wise transformations instance.

        Returns:
            pd.DataFrame: Model-ready inference frame.
        """
        transformer = row_wise_transformations if row_wise_transformations is not None else RowWiseTransformations()
        X_out = transformer.apply_row_wise_transformations(X_pd.copy())

        if row_wise_features:
            X_out = safe_production_transform(X_out, row_wise_features)

        if column_transformer is not None:
            transformed = column_transformer.transform(X_out, feature_nodes=column_wise_features)
            if isinstance(transformed, pd.DataFrame):
                X_out = transformed
            else:
                X_out = pd.DataFrame(transformed, index=X_out.index)

        if row_wise_features:
            X_alias = X_out.copy()
            for idx, node in enumerate(row_wise_features):
                legacy_name = f"autoFE_f_{idx}"
                formula_name = tree_to_formula(node)
                if legacy_name not in X_alias.columns and formula_name in X_alias.columns:
                    X_alias[legacy_name] = X_alias[formula_name]
            X_out = X_alias

        if best_features is not None:
            X_out = X_out.loc[:, list(best_features)]

        return apply_saved_categorical_mapping(X_out, categorical_mapping)
