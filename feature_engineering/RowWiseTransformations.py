import pandas as pd
import numpy as np
from utils.statics import PITCH_X


class RowWiseTransformations:
    def __init__(self):
        pass

    def apply_row_wise_transformations(self, X: pd.DataFrame) -> pd.DataFrame:
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

        # 6. Drop original raw columns
        drop_cols = [
            "start_x",
            "start_y",
            "end_x",
            "end_y",
            "angle",
            "duration",
            "length",
        ]
        df = df.drop(columns=[c for c in drop_cols if c in df.columns])

        return df
