import polars as pl


class PreprocessingHandler:
    def __init__(self, df, categorical_columns: list[str]):
        self.df = df
        self.categorical_columns = categorical_columns

    def preprocess_categorical_columns(self) -> pl.DataFrame:
        """Preprocess categorical columns in the DataFrame

        Returns:
            pl.DataFrame: DataFrame with preprocessed categorical columns
        """

        self.df = self.df.with_columns(pl.col("body_part").fill_null("Unknown"))

        self.df = self.df.with_columns(pl.col("under_pressure").fill_null(False))

        return self.df

    def preprocess_outcome_column(self) -> pl.DataFrame:
        """Binary encode the outcome column in the DataFrame

        Returns:
            pl.DataFrame: DataFrame with binary encoded outcome column
        """

        self.df = self.df.with_columns(
            pl.when(pl.col("outcome").is_null()).then(1).otherwise(0).alias("outcome")
        )

        return self.df

    def preprocess_length_column(self) -> pl.DataFrame:
        """Convert from yard to meter

        Returns:
            pl.DataFrame: DataFrame with length column converted to meters
        """
        self.df = self.df.with_columns((pl.col("length") * 0.9144).alias("length"))
        return self.df

    def preprocess_log_velocity_column(self) -> pl.DataFrame:
        """Calculate log-velocity and add as a new column

        Returns:
            pl.DataFrame: DataFrame with log-velocity column added
        """
        self.df = self.df.with_columns(
            pl.when(pl.col("duration") != 0)
            .then(pl.col("length") / pl.col("duration"))
            .otherwise(0)
            .log1p()
            .alias("log_velocity")
        )
        return self.df

    def preprocess_columns_with_log1p(self) -> pl.DataFrame:
        """Apply log1p transformation to specified columns

        Returns:
            pl.DataFrame: DataFrame with specified columns log1p transformed
        """
        for col in ["length", "duration"]:
            self.df = self.df.with_columns(pl.col(col).log1p().alias(f"log_{col}"))
        return self.df

    def preprocess_angle_column(self) -> pl.DataFrame:
        """Create sine and cosine features from angle column

        Returns:
            pl.DataFrame: DataFrame with angle_sin and angle_cos columns added
        """
        self.df = self.df.with_columns(
            [
                pl.col("angle").sin().alias("angle_sin"),
                pl.col("angle").cos().alias("angle_cos"),
            ]
        )

        self.df = self.df.drop(pl.col("angle"))

        return self.df
