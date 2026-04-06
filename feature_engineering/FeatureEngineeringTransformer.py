from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import numpy as np


class ColumnTransformer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        ohe_columns: list[str],
        cat_columns: list[str],
        use_ofe_features: bool = False,
    ):
        self.ohe_columns = ohe_columns
        self.cat_columns = cat_columns
        self.use_ofe_features = use_ofe_features
        # handle_unknown='ignore' is crucial for consistent nested CV
        self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        self.height_velocity_groupby_mean_map_ = None
        self.body_part_velocity_groupby_mean_map_ = None
        self.height_freq_map_ = None
        self.end_x_freq_map_ = None
        self.duration_freq_map_ = None

    def fit(self, X: pd.DataFrame, y=None):
        self.height_velocity_groupby_mean_map_ = X.groupby("height")[
            "log_velocity"
        ].mean()
        self.body_part_velocity_groupby_mean_map_ = X.groupby("body_part")[
            "log_velocity"
        ].mean()
        self.height_freq_map_ = X["height"].value_counts()
        self.end_x_freq_map_ = X["end_x"].value_counts()

        if self.use_ofe_features:
            self.duration_freq_map_ = X["duration"].value_counts()

        if self.ohe_columns:
            # Fit only on the specified OHE columns
            self.encoder.fit(X[self.ohe_columns])
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()

        # Group by height then mean log_velocity
        df["height_mean_log_velocity"] = (
            df["height"].map(self.height_velocity_groupby_mean_map_).astype(float)
        )
        # Group by height then mean log_velocity
        df["height_mean_log_velocity"] = (
            df["body_part"].map(self.body_part_velocity_groupby_mean_map_).astype(float)
        )

        # Value counts for height
        df["freq_height"] = df["height"].map(self.height_freq_map_).astype(float)

        # Value counts for end_x
        df["freq_end_x"] = df["end_x"].map(self.end_x_freq_map_).astype(float)

        if self.use_ofe_features:
            df["freq_duration"] = (
                df["duration"].map(self.duration_freq_map_).astype(float)
            )

        # 1. Categorical handling (Native pandas categories)
        for col in self.cat_columns:
            if col in df.columns:
                df[col] = df[col].astype("category")

        # 2. One-Hot Encoding
        if self.ohe_columns:
            encoded_array = self.encoder.transform(df[self.ohe_columns])
            encoded_cols = self.encoder.get_feature_names_out(self.ohe_columns)

            # Create a temporary DF for encoded features
            encoded_df = pd.DataFrame(
                encoded_array,
                columns=encoded_cols,
                index=df.index,  # Critical to keep indices aligned
            )

            # Combine and drop original OHE columns
            df = pd.concat([df, encoded_df], axis=1).drop(columns=self.ohe_columns)

        return df
