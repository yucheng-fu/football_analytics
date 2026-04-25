import pandas as pd
import numpy as np
from utils.statics import PITCH_X
from feature_engineering.OpenFE.FeatureGenerator import Node
from feature_engineering.OpenFE.openfe import tree_to_formula
from typing import List


class RowWiseTransformations:
    def __init__(self, use_ofe_features: bool = False):
        self.use_ofe_features = use_ofe_features

    def apply_row_wise_transformations(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()

        # Yards to meters
        df["length"] = df["length"] * 0.9144

        # Compute log-velocity
        velocity = df["length"] / df["duration"]
        df["log_velocity"] = np.log1p(velocity.replace([np.inf, -np.inf], 0).fillna(0))

        # Splitting angle into cos and sin
        df["angle_sin"] = np.sin(df["angle"])
        df["angle_cos"] = np.cos(df["angle"])

        # Progressive distance
        df["start_distance_to_goal"] = np.sqrt(
            (df["start_x"] - PITCH_X) ** 2 + (df["start_y"] - 40) ** 2
        )
        df["end_distance_to_goal"] = np.sqrt(
            (df["end_x"] - PITCH_X) ** 2 + (df["end_y"] - 40) ** 2
        )
        df["progressive_distance"] = (
            df["start_distance_to_goal"] - df["end_distance_to_goal"]
        )

        # Direction to goal
        goal_angle = np.arctan2(40 - df["start_y"], PITCH_X - df["start_x"])
        df["direction_to_goal"] = df["angle"] - goal_angle
        df["direction_to_goal_cos"] = np.cos(df["direction_to_goal"])

        # Interaction features
        df["duration_x_under_pressure"] = (
            df["duration"] * df["under_pressure"]
        ).astype(float)

        df["log_velocity_x_under_pressure"] = (
            df["log_velocity"] * df["under_pressure"]
        ).astype(float)

        df["length_x_under_pressure"] = (df["length"] * df["under_pressure"]).astype(
            float
        )

        if self.use_ofe_features:
            df["duration_div_log_velocity"] = df["duration"] / df[
                "log_velocity"
            ].replace(0, np.nan)
            df["min_end_x_duration"] = np.minimum(df["end_x"], df["duration"])

        return df

    def apply_openfe_transformations(
        self, X: pd.DataFrame, feature_list: List[Node]
    ) -> pd.DataFrame:
        X_new = pd.DataFrame(index=X.index)

        for feature in feature_list:
            feature.calculate(X_new, is_root=True)
            name = tree_to_formula(feature)
            X_new[name] = feature.data
            feature.delete()

        return pd.concat([X, X_new], axis=1)
