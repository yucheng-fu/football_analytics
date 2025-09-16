
import numpy as np
import polars as pl
import mplsoccer as mpl 
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap
from matplotlib.axes import Axes
from typing import Tuple
import matplotlib.pyplot as plt

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

def invert_orientation(x: np.array, y: np.array, PITCH_X: int, PITCH_Y: int) -> Tuple[np.array, np.array]:
    """Invert the orientation of the pitch coordinates.

    Args:
        x (np.array): x-coordinates of the events
        y (np.array): y-coordinates of the events
        PITCH_X (int): Width of the pitch

    Returns:
        Tuple[np.array, np.array]: Inverted x and y coordinates
    """
    x_flipped_orientation = PITCH_X - x
    y_flipped_orientation = PITCH_Y - y

    return (x_flipped_orientation, y_flipped_orientation)

def invert_one_orientation(orientation: np.array, PITCH_DIM: int) -> np.array:
    """Invert one orientation of the pitch coordinates.

    Args:
        orientation (np.array): x or y-coordinates of the events
        PITCH_DIM (int): Width or height of the pitch

    Returns:
        np.array: Inverted x or y coordinates
    """
    return PITCH_DIM - orientation

def add_legend(ax: Axes, num_elements: int, colors: list[str], labels: list[str], markers: list[str] = None) -> None:
    """Add a legend to the mpl pitch plot

    Args:
        ax (Axes): ax object
        num_elements (int): Number of elements in the legend
        colors (list[str]): List of colors for the legend
        labels (list[str]): List of labels for the legend
        markers (list[str], optional): List of marker styles for the legend
    """
    if markers is not None:
        legend_elements = [
            Patch(facecolor=colors[i], edgecolor='black', alpha=0.5, label=labels[i], hatch=markers[i])
            for i in range(num_elements)
        ]
    else:
        legend_elements = [
            Patch(facecolor=colors[i], edgecolor='black', alpha=0.5, label=labels[i])
            for i in range(num_elements)
        ]
    ax.legend(handles=legend_elements, loc='upper right')

def plot_player_positions(x: np.array, y: np.array, jerseys: list[str], names: list[str], pitch: mpl.Pitch, ax: Axes, color: str, fontsize: int, alpha: float, offset: float) -> None:
    # Plot player positions
    pitch.scatter(x, y, s=300, c=color, edgecolors='black', linewidth=1.5, ax=ax, zorder=3)

    # Add jersey numbers and player names 
    for xi, yi, num, name in zip(x, y, jerseys, names):
        ax.text(xi, yi, str(num), ha='center', va='center',
                color='white', fontsize=10, fontweight='bold', zorder=4)
        ax.text(xi, yi + 3.5, name, ha='center', va='bottom',
                color='black', fontsize=8, zorder=5)

    plt.title("Average player positions based on events", fontsize=fontsize)
    plt.show()