import polars as pl
import numpy as np
from sklearn.mixture import GaussianMixture
from utils.utils import invert_orientation
from utils.statics import (PITCH_X, PITCH_Y)
from typing import Tuple

class GMMHandler():
    def __init__(self, team_events: pl.DataFrame, invert_coords: bool = False):
        self.team_events = team_events  
        self.invert_coords = invert_coords
        self.passes = self.get_pass_events(team_events)
        self.carries = self.get_carry_events(team_events)
        self.dribbles = self.get_dribble_events(team_events)
        self.optimal_n_components = None

    def get_possession_events(self) -> np.ndarray:
        """Extracts the x and y coordinates of possession events.

        Args:
            passes (pl.DataFrame): The DataFrame containing pass events.
            carries (pl.DataFrame): The DataFrame containing carry events.
            dribbles (pl.DataFrame): The DataFrame containing dribble events.

        Returns:
            np.ndarray: A 2D array containing the x and y coordinates of the filtered events.
        """
        pass_x2, pass_y2 = self.get_x_y_event_locations(self.passes, event_type="pass", invert=self.invert_coords)
        carry_x2, carry_y2 = self.get_x_y_event_locations(self.carries, event_type="carry", invert=self.invert_coords)
        dribble_x2, dribble_y2 = self.get_x_y_event_locations(self.dribbles, event_type="dribble", invert=self.invert_coords)

        all_coords = np.vstack([
            np.column_stack([pass_x2, pass_y2]),
            np.column_stack([carry_x2, carry_y2]),
            np.column_stack([dribble_x2, dribble_y2])
        ])

        return pl.DataFrame(all_coords, schema=["x2", "y2"])

    def get_x_y_event_locations(self, team_events: pl.DataFrame, event_type: str = "pass", invert: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """Extracts the x and y coordinates of events of a specific type.

        Args:
            team_events (pl.DataFrame): The DataFrame containing team event data.
            type (str): The type of event to filter by (e.g., "Pass", "Carry", "Dribble").

        Returns:
            np.ndarray: A 2D array containing the x and y coordinates of the filtered events.
        """
        if event_type == "dribble":
            team_events_coords = team_events.filter(
            (pl.col("location").is_not_null())).select(["location"])
            dribble_x, dribble_y = np.array(team_events_coords["location"].to_list()).T

            if invert:
                dribble_x, dribble_y = invert_orientation(dribble_x, dribble_y, PITCH_X, PITCH_Y)
                return (dribble_x, dribble_y)

            return (dribble_x, dribble_y)
        
        elif event_type == "carry":
            carry_coords = team_events.filter(
                (pl.col("location").is_not_null()) & (pl.col("carry_end_location").is_not_null())
            ).select(["location", "carry_end_location"])
            carry_x2, carry_y2 = np.array(carry_coords["carry_end_location"].to_list()).T

            if invert:
                carry_x2, carry_y2 = invert_orientation(carry_x2, carry_y2, PITCH_X, PITCH_Y)
                return (carry_x2, carry_y2)

            return (carry_x2, carry_y2)

        elif event_type == "pass":
            pass_coords = team_events.filter(
                (pl.col("location").is_not_null()) & (pl.col("pass_end_location").is_not_null())
            ).select(["location", "pass_end_location"])
            pass_x2, pass_y2 = np.array(pass_coords["pass_end_location"].to_list()).T

            if invert:
                pass_x2, pass_y2 = invert_orientation(pass_x2, pass_y2, PITCH_X, PITCH_Y)
                return (pass_x2, pass_y2)

            return (pass_x2, pass_y2)
        else:
            raise ValueError("Invalid event type. Please choose from 'dribble', 'carry', or 'pass'.")
    
    def get_pass_events(self, team_events: pl.DataFrame) -> pl.DataFrame:
        """
        Filters the team events DataFrame to include only pass events.
        
        Parameters:
        team_events (pl.DataFrame): The input DataFrame containing team event data.
        
        Returns:
        pl.DataFrame: A DataFrame containing only pass events.
        """
        
        return team_events.filter(pl.col("type") == "Pass")
    
    def get_carry_events(self, team_events: pl.DataFrame) -> pl.DataFrame:
        """
        Filters the team events DataFrame to include only carry events.
        
        Parameters:
        team_events (pl.DataFrame): The input DataFrame containing team event data.
        
        Returns:
        pl.DataFrame: A DataFrame containing only carry events.
        """
        
        return team_events.filter(pl.col("type") == "Carry")
    
    def get_dribble_events(self, team_events: pl.DataFrame) -> pl.DataFrame:
        """
        Filters the team events DataFrame to include only successful dribble events.
        
        Parameters:
        team_events (pl.DataFrame): The input DataFrame containing team event data.
        
        Returns:
        pl.DataFrame: A DataFrame containing only dribble events.
        """
        
        return team_events.filter((pl.col("type") == "Dribble") & (pl.col("dribble_outcome") == "Complete")).select("location", "dribble_outcome")
    
    def get_optimal_number_of_compoents_using_bic(self, features: pl.DataFrame, max_components: int = 50, random_state: int = 42) -> int:

        n_components_range = range(1, max_components)
        bic = []

        for n_components in n_components_range:
            gmm = GaussianMixture(n_components=n_components, random_state=42)
            gmm.fit(features)
            bic.append(gmm.bic(features))

        self.optimal_n_components = n_components_range[bic.index(min(bic))]
        return self.optimal_n_components
    
    def fit_gmm(self, features: pl.DataFrame, random_state: int = 42) -> GaussianMixture:
        n_components = self.get_optimal_number_of_compoents_using_bic(features) if self.optimal_n_components is None else self.optimal_n_components
        
        gmm = GaussianMixture(n_components=n_components, random_state=random_state)
        gmm.fit(features)

        return gmm