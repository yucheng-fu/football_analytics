import polars as pl


class FeatureEngineeringHandler:
    def __init__(self, train_df: pl.DataFrame):
        self.train_df = train_df

    def encode_columns(self, columns: list[str]) -> pl.DataFrame:
        """Encode categorical columns using one-hot encoding

        Returns:
            pl.DataFrame: DataFrame with one-hot encoded categorical columns
        """

        for col in columns:
            dummies = self.train_df.select(pl.col(col)).to_dummies()
            self.train_df = self.train_df.hstack(dummies).drop(pl.col(col))

        return self.train_df
