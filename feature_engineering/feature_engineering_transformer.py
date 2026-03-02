from sklearn.base import BaseEstimator, TransformerMixin
from utils.statics import PITCH_X
import polars as pl
from sklearn.preprocessing import OneHotEncoder


class ManualFeatureEngineeringTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, ohe_columns: list[str], cat_columns: list[str]):
        self.ohe_columns = ohe_columns
        self.cat_columns = cat_columns

        # Initialize the sklearn encoder
        # handle_unknown='ignore' ensures test sets with new categories do not throw an error
        # sparse_output=False makes it easier to hstack back into Polars
        self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    def fit(self, X, y=None):
        if self.ohe_columns:
            subset = X.select(self.ohe_columns).to_pandas()
            self.encoder.fit(subset)
        return self

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        # Perform manual feature engineering here
        df = X.clone()

        # 1. Geometry / distance features
        df = df.with_columns(
            [
                (pl.col("length") * 0.9144).alias("length"),
                ((pl.col("start_x") - PITCH_X) ** 2 + (pl.col("start_y") - 40) ** 2)
                .sqrt()
                .alias(
                    "start_distance_to_goal"
                ),  # 40 is the y-coordinate of the center of the goal
                ((pl.col("end_x") - PITCH_X) ** 2 + (pl.col("end_y") - 40) ** 2)
                .sqrt()
                .alias("end_distance_to_goal"),
                (pl.col("end_x") - pl.col("start_x")).alias("dx"),
                (pl.col("end_y") - pl.col("start_y")).alias("dy"),
            ]
        )

        # 2. Derived features
        df = df.with_columns(
            [
                (
                    pl.col("start_distance_to_goal") - pl.col("end_distance_to_goal")
                ).alias("progressive_distance"),
                pl.when(pl.col("duration") != 0)
                .then(pl.col("length") / pl.col("duration"))
                .otherwise(0)
                .log1p()
                .alias("log_velocity"),
                pl.col("angle").sin().alias("angle_sin"),
                pl.col("angle").cos().alias("angle_cos"),
            ]
        )

        # 3. Log-transforms
        df = df.with_columns(
            [
                pl.col("length").log1p().alias("log_length"),
                pl.col("duration").log1p().alias("log_duration"),
            ]
        )

        # 4. Categorical handling
        for col in self.cat_columns:
            if col in df.columns:
                df = df.with_columns(pl.col(col).cast(pl.Categorical))

        for col in self.ohe_columns:
            subset_pd = df.select(self.ohe_columns).to_pandas()
            encoded_array = self.encoder.transform(subset_pd)

            encoded_cols = self.encoder.get_feature_names_out(self.ohe_columns)

            encoded_df = pl.DataFrame(encoded_array, schema=list(encoded_cols))
            df = df.hstack(encoded_df).drop(self.ohe_columns)

        # 5. Drop columns
        df = df.drop(["start_x", "start_y", "end_x", "end_y", "angle"])

        return df
