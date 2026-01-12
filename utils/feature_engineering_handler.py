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
            # replace spaces in category values with underscores before dummification
            self.X = self.X.with_columns(
                pl.col(col).cast(pl.Utf8).str.replace_all(" ", "_").alias(col)
            )
            dummies = self.X.select(pl.col(col)).to_dummies()
            self.X = self.X.hstack(dummies).drop(pl.col(col))

        return self.X

    def preprocess_length_column(self) -> pl.DataFrame:
        """Convert from yard to meter

        Returns:
            pl.DataFrame: DataFrame with length column converted to meters
        """
        self.X = self.X.with_columns((pl.col("length") * 0.9144).alias("length"))
        return self.X

    def preprocess_log_velocity_column(self) -> pl.DataFrame:
        """Calculate log-velocity and add as a new column

        Returns:
            pl.DataFrame: DataFrame with log-velocity column added
        """
        self.X = self.X.with_columns(
            pl.when(pl.col("duration") != 0)
            .then(pl.col("length") / pl.col("duration"))
            .otherwise(0)
            .log1p()
            .alias("log_velocity")
        )
        return self.X

    def preprocess_columns_with_log1p(self) -> pl.DataFrame:
        """Apply log1p transformation to specified columns

        Returns:
            pl.DataFrame: DataFrame with specified columns log1p transformed
        """
        for col in ["length", "duration"]:
            self.X = self.X.with_columns(pl.col(col).log1p().alias(f"log_{col}"))
        return self.X

    def preprocess_angle_column(self, drop_angle_column: bool = False) -> pl.DataFrame:
        """Create sine and cosine features from angle column

        Returns:
            pl.DataFrame: DataFrame with angle_sin and angle_cos columns added
        """
        self.X = self.X.with_columns(
            [
                pl.col("angle").sin().alias("angle_sin"),
                pl.col("angle").cos().alias("angle_cos"),
            ]
        )

        if drop_angle_column:
            self.X = self.X.drop(pl.col("angle"))

        return self.X

    def preprocess_direction_columns(self) -> pl.DataFrame:
        """Create dx and dy features from direction columns

        Returns:
            pl.DataFrame: DataFrame with dx and dy columns added
        """
        self.X = self.X.with_columns((pl.col("end_x") - pl.col("start_x")).alias("dx"))
        self.X = self.X.with_columns((pl.col("end_y") - pl.col("start_y")).alias("dy"))

        self.X = self.X.drop(pl.col("start_x"))
        self.X = self.X.drop(pl.col("end_x"))
        self.X = self.X.drop(pl.col("start_y"))
        self.X = self.X.drop(pl.col("end_y"))

        return self.X
