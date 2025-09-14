
import numpy as np
import polars as pl
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap
from matplotlib.axes import Axes
from typing import Tuple

def build_cmap(x : Tuple[int, int, int], y: Tuple[int, int, int]) -> ListedColormap:
    """Build cmap for Matplotlib

    Args:
        x (Tuple[int, int, int]): Tuple of RGB values
        y (Tuple[int, int, int]): Tuple of RGB values

    Returns:
        ListedColormap: ListedColorMap object for Matplotlib
    """
    r,g,b = x
    r_, g_, b_ = y
    N = 256
    A = np.ones((N, 4))
    A[:, 0] = np.linspace(r, 1, N)
    A[:, 1] = np.linspace(g, 1, N)
    A[:, 2] = np.linspace(b, 1, N)
    cmp = ListedColormap(A)
    
    B = np.ones((N, 4))
    B[:, 0] = np.linspace(r_, 1, N)
    B[:, 1] = np.linspace(g_, 1, N)
    B[:, 2] = np.linspace(b_, 1, N)
    cmp_ = ListedColormap(B)
    
    newcolors = np.vstack((cmp(np.linspace(0, 1, 128)),
                            cmp_(np.linspace(1, 0, 128))))
    return ListedColormap(newcolors)

def invert_orientation(x: np.array, y: np.array, PITCH_X: int) -> Tuple[np.array, np.array]:
    """Invert the orientation of the pitch coordinates.

    Args:
        x (np.array): x-coordinates of the events
        y (np.array): y-coordinates of the events
        PITCH_X (int): Width of the pitch

    Returns:
        Tuple[np.array, np.array]: Inverted x and y coordinates
    """
    x_flipped_orientation = PITCH_X - x
    y_flipped_orientation = y

    return (x_flipped_orientation, y_flipped_orientation)

def add_legend(ax: Axes, num_elements: int, colors: list[str], labels: list[str]) -> None:
    """Add a legend to the mpl pitch plot

    Args:
        ax (Axes): ax object
        num_elements (int): Number of elements in the legend
        colors (list[str]): List of colors for the legend
        labels (list[str]): List of labels for the legend
    """
    
    legend_elements = [Patch(facecolor=colors[i], edgecolor='black', alpha=0.5, label=labels[i]) for i in range(num_elements)]
    
    ax.legend(handles=legend_elements, loc='upper right')

def extract_team_from_tactics(tactics: pl.DataFrame) -> pl.DataFrame:

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
            pl.when(pl.col("team_idx") == 0).then(pl.lit("France"))
            .otherwise(pl.lit("Argentina"))
            .alias("team")
        )
        .drop("team_idx")
    )

    return players_df

