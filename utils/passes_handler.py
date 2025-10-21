from typing import Tuple

import numpy as np
import polars as pl
from statsbombpy import sb
from tqdm import tqdm


class PassesHandler:
    def __init__(self, match_ids: list[int]):
        self.match_ids = match_ids

    def get_passes_from_match_ids(self) -> pl.DataFrame:
        """Gatcher passes from all matches in the FIFA World Cup 2018

        Returns:
            pl.DataFrame: DataFrame containing all pass events
        """
        all_features = []

        for match_id in tqdm(self.match_ids):
            passes = self.fetch_passes_from_match(match_id=match_id)

            start_x, start_y = self.parse_pass_location(passes, col_name="location")
            end_x, end_y = self.parse_pass_location(
                passes, col_name="pass_end_location"
            )
            length = self.parse_pass_length(passes)
            height = self.parse_pass_height(passes)
            angle = self.parse_pass_angle(passes)
            duration = self.parse_pass_duration(passes)
            body_part = self.parse_pass_body_part(passes)
            under_pressure = self.parse_under_pressure(passes)
            outcome = self.parse_pass_outcome(passes)
            match_ids_array = np.full(len(start_x), match_id)

            features = np.vstack(
                [
                    match_ids_array,
                    start_x,
                    start_y,
                    end_x,
                    end_y,
                    length,
                    height,
                    angle,
                    duration,
                    body_part,
                    under_pressure,
                    outcome,
                ]
            ).T

            all_features.extend(features)

        df = pl.DataFrame(
            all_features,
            schema=[
                "match_id",
                "start_x",
                "start_y",
                "end_x",
                "end_y",
                "length",
                "height",
                "angle",
                "duration",
                "body_part",
                "under_pressure",
                "outcome",
            ],
        )

        # save to csv
        # df.write_csv("../data/02-analysis/passes.csv")
        df.write_parquet("../data/02-analysis/passes.parquet")

        return df

    def fetch_passes_from_match(self, match_id: int) -> pl.DataFrame:
        """Get passes from match

        Args:
            match_id (int): ID of the match

        Returns:
            pl.DataFrame: DataFrame containing pass events
        """
        events_df = sb.events(match_id=match_id)
        events = pl.from_pandas(events_df)
        passes = events.filter(pl.col("type") == "Pass")
        return passes

    def parse_pass_location(
        self, passes: pl.DataFrame, col_name: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Parsing the location of the pass

        Args:
            passes (pl.DataFrame): passes Dataframe
            col_name (str): column name of

        Returns:
            Tuple[np.ndarray, np.ndarray]: x and y coordinates of the pass location
        """
        return np.array(
            passes.select(pl.col(col_name)).to_series().to_list()
        ).transpose()

    def parse_pass_length(self, passes: pl.DataFrame) -> list[float]:
        """Parsing length of the pass

        Args:
            passes (pl.DataFrame): passes DataFrame

        Returns:
            list[float]: List of pass lengths
        """
        return passes.select(pl.col("pass_length")).to_series().to_list()

    def parse_pass_body_part(self, passes: pl.DataFrame) -> list[str]:
        """Parsing the body part of the pass

        Args:
            passes (pl.DataFrame): passes DataFrame

        Returns:
            list[str]: List of pass body parts
        """
        return passes.select(pl.col("pass_body_part")).to_series().to_list()

    def parse_under_pressure(self, passes: pl.DataFrame) -> list[bool]:
        """Parsing whether the pass was made under pressure

        Args:
            passes (pl.DataFrame): passes DataFrame

        Returns:
            list[bool]: List of whether the pass was made under pressure
        """
        return passes.select(pl.col("under_pressure")).to_series().to_list()

    def parse_pass_duration(self, passes: pl.DataFrame) -> list[float]:
        """Parsing duration of the pass

        Args:
            passes (pl.DataFrame): passes DataFrame

        Returns:
            list[float]: List of pass durations
        """
        return passes.select(pl.col("duration")).to_series().to_list()

    def parse_pass_outcome(self, passes: pl.DataFrame) -> list[str]:
        """Parsing the outcome of the pass

        Args:
            passes (pl.DataFrame): passes DataFrame

        Returns:
            list[str]: List of pass outcomes
        """
        return passes.select(pl.col("pass_outcome")).to_series().to_list()

    def parse_pass_height(self, passes: pl.DataFrame) -> list[str]:
        """Parsing the height of the pass

        Args:
            passes (pl.DataFrame): passes DataFrame

        Returns:
            list[str]: List of pass heights
        """
        return passes.select(pl.col("pass_height")).to_series().to_list()

    def parse_pass_angle(self, passes: pl.DataFrame) -> list[float]:
        """Parsing the angle of the pass

        Args:
            passes (pl.DataFrame): passes DataFrame

        Returns:
            list[float]: List of pass angles
        """
        return passes.select(pl.col("pass_angle")).to_series().to_list()


if __name__ == "__main__":
    passes_handler = PassesHandler(match_ids=[8658, 8657, 8656])
    passes_df = passes_handler.get_passes_from_match_ids()
