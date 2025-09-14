import polars as pl 
import numpy as np
from typing import Tuple

class EventsHandler():

    def __init__(self, events: pl.DataFrame, team_name1: str, team_name2: str) -> None:
        self.events = events
        self.team_name1 = team_name1
        self.team_name2 = team_name2


    def compute_team_events(self, team_name: str ) -> pl.DataFrame:
        """Compute the events for a specific team.

        Args:
            team_name (str): Name of the team
        """
        return self.events.filter((pl.col("location").is_not_null()) & (pl.col("team") == team_name))
    
    def extract_x_y_location_from_events(self, team_name: str) -> Tuple[float, float]:
        
        """Extract x and y location from events DataFrame.

        Args:
            team_events (pl.DataFrame): Events DataFrame for a specific
        """
        team_events = self.compute_team_events(team_name)

        x, y = np.array(team_events.select(pl.col("location")).to_series().to_list()).transpose()

        return (x, y)
    
    def compute_avg_pos_for_starting_xi(self, team_name: str) -> pl.DataFrame:
        """Compute the average position for the starting XI of a team.

        Args:
            team_name (str): Name of the team

        Returns:
            pl.DataFrame: DataFrame containing the average positions of the starting XI players
        """
        starting_xi = self.get_starting_xi_from_events_df()
        team_starting_xi = (starting_xi
                            .filter(pl.col("team") == team_name)
                            .select(pl.col("player_id"))).to_series().to_list()

        team_11_events = (self.compute_team_events(team_name)
                          .filter(pl.col("player_id")
                          .is_in(team_starting_xi)))
        
        return (team_11_events
            .select(pl.col("location").list.get(0).alias("x"), pl.col("location").list.get(1).alias("y"), pl.col("player_id").cast(pl.Int64))
            .group_by("player_id")
            .agg(pl.mean("x").alias("avg_x"), pl.mean("y").alias("avg_y")))

    def get_starting_xi_from_events_df(self) -> pl.DataFrame:
        """Get the starting XI players from the events DataFrame.

        Returns:
            pl.DataFrame: DataFrame containing the starting XI players
        """
        tactics = self.events.filter(pl.col("type") == "Starting XI").select(pl.col("tactics"))

        starting_xi = self.extract_team_from_match_tactics(tactics)
        return starting_xi

    def extract_team_from_match_tactics(self, tactics: pl.DataFrame) -> pl.DataFrame:
        """Extract the team information from the tactics DataFrame.

        Args:
            tactics (pl.DataFrame): DataFrame containing tactics information

        Returns:
            pl.DataFrame: DataFrame containing the team player information for both teams
        """
        players_df = (
            tactics
            .with_columns(
                pl.col("tactics").struct.field("lineup").alias("lineup")
            )
            .with_row_count("team_idx")  # 0 for first team, 1 for second team
            .explode("lineup")
            .select([
                pl.col("lineup").struct.field("jersey_number").alias("jersey_number"),
                pl.col("lineup").struct.field("player").struct.field("id").alias("player_id"),
                pl.col("lineup").struct.field("player").struct.field("name").alias("player_name"),
                pl.col("team_idx")
            ])
            .with_columns(
                pl.when(pl.col("team_idx") == 0).then(pl.lit(self.team_name1))
                .otherwise(pl.lit(self.team_name2))
                .alias("team")
            )
            .drop("team_idx")
        )

        return players_df