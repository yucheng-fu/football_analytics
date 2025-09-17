import polars as pl
import numpy as np

class ShotsHandler():
    def __init__(self, team_shots_events: pl.DataFrame) -> None:
        self.team_shots_events = team_shots_events
        self.team_goals = self.get_goals_from_shots()        
        self.team_non_goals = self.get_non_goals_from_shots()

    def get_non_goals_from_shots(self) -> pl.DataFrame:
        return self.team_shots_events.filter(pl.col("shot_outcome") != "Goal")

    def get_goals_from_shots(self) -> pl.DataFrame:
        return self.team_shots_events.filter(pl.col("shot_outcome") == "Goal")

    def compute_goal_x_y_locations(self) -> tuple[float, float]:
        """Compute the x and y locations of goals.

        Returns:
            tuple[float, float]: The x and y locations of goals.
        """
        return self.extract_x_y_location_from_events(self.team_goals)
    
    def compute_shot_x_y_locations(self) -> tuple[float, float]:
        """Compute the x and y locations of shots.

        Returns:
            tuple[float, float]: The x and y locations of shots.
        """
        return self.extract_x_y_location_from_events(self.team_non_goals)

    def extract_x_y_location_from_events(self, team_events: pl.DataFrame) -> tuple[float, float]:

        """Extract x and y location from events DataFrame.

        Args:
            team_events (pl.DataFrame): Events DataFrame for a specific team
        """

        x, y = np.array(team_events.select(pl.col("location")).to_series().to_list()).transpose()

        return (x, y)

    def compute_xg_from_goal_events(self) -> float:

        """Compute xG from goal events DataFrame.

        Args:
            goals (pl.DataFrame): Goals DataFrame for a specific team
        """

        return self.team_goals.select(pl.col("shot_statsbomb_xg")).to_numpy().flatten()