import polars as pl
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureEngineeringHandler:
    def __init__(self, X: pl.DataFrame):
        self.X = X

    def encode_columns(self, columns: list[str]) -> pl.DataFrame:
        """Encode categorical columns using one-hot encoding

        Returns:
            pl.DataFrame: DataFrame with one-hot encoded categorical columns
        """

        for col in columns:
            dummies = self.X.select(pl.col(col)).to_dummies()
            self.X = self.X.hstack(dummies).drop(pl.col(col))

        return self.X
