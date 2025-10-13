import polars as pl
import statsbombpy as sb

class PassesHandler():

    def __init__(self, match_ids: list[int]):
        self.match_ids = match_ids


    def passes_from_match_ids(self):

        for match_id in self.match_ids:
            passes = self.fetch_passes_from_match(match_id=match_id)


    def fetch_passes_from_match(self, match_id: int) -> pl.DataFrame:
        events_df = sb.events(match_id=match_id)
        events = pl.from_pandas(events_df)
        passes = events.filter(pl.col("type") == "Pass")


    def parse_pass_location(self):
        pass # Haha, because it's the passes handler. Get it?
    
    def parse_pass_duration(self):
        pass
    def parse_pass_outcome(self):
        pass

    def parse_pass_height(self):
        pass

    def parse_pass_angle(self):
        pass