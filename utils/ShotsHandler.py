import polars as pl

class ShotsHandler():
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