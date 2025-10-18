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

        self.df = self.df.with_columns(
            pl.when(pl.col("outcome").is_null())
            .then(1)  # Completed
            .otherwise(0)  # Incompleted
            .alias("outcome")
        )

        return self.df
