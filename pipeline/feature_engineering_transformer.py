import polars as pl
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureEngineeringTransformer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        how_to_handle_cat_columns: str,
        cat_columns: list[str],
        drop_angle_column: bool = False,
    ):
        self.how_to_handle_cat_columns = how_to_handle_cat_columns
        self.cat_columns = cat_columns
        self.drop_angle_column = drop_angle_column
        self.columns_ = None

    def fit(
        self, X: pl.DataFrame, y: pl.DataFrame = None
    ) -> "FeatureEngineeringTransformer":
        self.columns_ = X.columns
        return self

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        df = X.clone()

        # 1. Encode columns
        if self.how_to_handle_cat_columns == "int":
            for col in self.cat_columns:
                df = df.with_columns(
                    pl.col(col).cast(pl.Categorical).cast(pl.Int32).alias(col)
                )
        elif self.how_to_handle_cat_columns == "onehot":
            for col in self.cat_columns:
                # replace spaces in category values with underscores before dummification
                df = df.with_columns(
                    pl.col(col).cast(pl.Utf8).str.replace_all(" ", "_").alias(col)
                )
                dummies = df.select(pl.col(col)).to_dummies()
                df = df.hstack(dummies).drop(pl.col(col))

        # 2. Length (yard to meter)
        if "length" in df.columns:
            df = df.with_columns((pl.col("length") * 0.9144).alias("length"))

        # 3. Log velocity
        if "duration" in df.columns and "length" in df.columns:
            df = df.with_columns(
                pl.when(pl.col("duration") != 0)
                .then(pl.col("length") / pl.col("duration"))
                .otherwise(0)
                .log1p()
                .alias("log_velocity")
            )

        # 4. Log1p transformations
        for col in ["length", "duration"]:
            if col in df.columns:
                df = df.with_columns(pl.col(col).log1p().alias(f"log_{col}"))

        # 5. Angle sin/cos transformations
        if "angle" in df.columns:
            df = df.with_columns(
                [
                    pl.col("angle").sin().alias("angle_sin"),
                    pl.col("angle").cos().alias("angle_cos"),
                ]
            )
            if self.drop_angle_column:
                df = df.drop("angle")

        # 6. Start distance to goal
        df = df.with_columns(
            (
                ((pl.col("start_x") - 120) ** 2 + (pl.col("start_y") - 40) ** 2).sqrt()
            ).alias("start_distance_to_goal")
        )

        # 7. End distance to goal
        df = df.with_columns(
            (((pl.col("end_x") - 120) ** 2 + (pl.col("end_y") - 40) ** 2).sqrt()).alias(
                "end_distance_to_goal"
            )
        )

        # 8. Progressive distance
        df = df.with_columns(
            (pl.col("start_distance_to_goal") - pl.col("end_distance_to_goal")).alias(
                "progressive_distance"
            )
        )

        # 9. Dx dy
        required_dirs = ["end_x", "start_x", "end_y", "start_y"]
        if all(c in df.columns for c in required_dirs):
            df = df.with_columns(
                [
                    (pl.col("end_x") - pl.col("start_x")).alias("dx"),
                    (pl.col("end_y") - pl.col("start_y")).alias("dy"),
                ]
            ).drop(required_dirs)

        return df
