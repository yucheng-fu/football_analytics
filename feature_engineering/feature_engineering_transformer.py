from sklearn.base import BaseEstimator, TransformerMixin
from utils.statics import PITCH_X
import polars as pl
import pandas as pd
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


class ManualFeatureEngineeringTransformerPandasw(BaseEstimator, TransformerMixin):
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

        # 1. Geometry / distance features
        df["length"] = df["length"] * 0.9144

        # Calculate distances using NumPy for speed
        df["start_distance_to_goal"] = np.sqrt(
            (df["start_x"] - PITCH_X) ** 2 + (df["start_y"] - 40) ** 2
        )
        df["end_distance_to_goal"] = np.sqrt(
            (df["end_x"] - PITCH_X) ** 2 + (df["end_y"] - 40) ** 2
        )

        df["dx"] = df["end_x"] - df["start_x"]
        df["dy"] = df["end_y"] - df["start_y"]

        # 2. Derived features
        df["progressive_distance"] = (
            df["start_distance_to_goal"] - df["end_distance_to_goal"]
        )

        # Velocity calculation with 0-division protection
        velocity = df["length"] / df["duration"]
        df["log_velocity"] = np.log1p(velocity.replace([np.inf, -np.inf], 0).fillna(0))

        df["angle_sin"] = np.sin(df["angle"])
        df["angle_cos"] = np.cos(df["angle"])

        # 3. Log-transforms
        df["log_length"] = np.log1p(df["length"])
        df["log_duration"] = np.log1p(df["duration"])

        # 4. Categorical handling (Native pandas categories)
        for col in self.cat_columns:
            if col in df.columns:
                df[col] = df[col].astype("category")

        # 5. One-Hot Encoding
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

        # 6. Drop original raw columns
        drop_cols = ["start_x", "start_y", "end_x", "end_y", "angle"]
        df = df.drop(columns=[c for c in drop_cols if c in df.columns])

        return df
