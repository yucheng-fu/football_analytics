from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import numpy as np


class ColumnTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, ohe_columns: list[str], cat_columns: list[str]):
        self.ohe_columns = ohe_columns
        self.cat_columns = cat_columns
        # handle_unknown='ignore' is crucial for consistent nested CV
        self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    def fit(self, X: pd.DataFrame, y=None):
        if self.ohe_columns:
            # Fit only on the specified OHE columns
            self.encoder.fit(X[self.ohe_columns])
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        # 0. Work on a copy to avoid SettingWithCopyWarnings
        df = X.copy()

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
