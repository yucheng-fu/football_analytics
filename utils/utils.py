
import numpy as np
import polars as pl
import mplsoccer as mpl 
from matplotlib.patches import Ellipse, Patch
from matplotlib.colors import ListedColormap
from matplotlib.axes import Axes
from typing import Tuple
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from scipy.stats import multivariate_normal

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

def plot_player_positions(x: np.array, y: np.array, jerseys: list[str], names: list[str], pitch: mpl.Pitch, ax: Axes, color: str, fontsize: int, fig_name: str) -> None:
    # Plot player positions
    pitch.scatter(x, y, s=300, c=color, edgecolors='black', linewidth=1.5, ax=ax, zorder=3)

    # Add jersey numbers and player names 
    for xi, yi, num, name in zip(x, y, jerseys, names):
        ax.text(xi, yi, str(num), ha='center', va='center',
                color='white', fontsize=10, fontweight='bold', zorder=4)
        ax.text(xi, yi + 3.5, name, ha='center', va='bottom',
                color='black', fontsize=8, zorder=5)

    plt.title("Average player positions based on events", fontsize=fontsize)
    plt.savefig(fig_name, dpi=300, bbox_inches='tight')
    plt.show()

def plot_pitch_with_shots(ax: Axes, team_shots_x: np.array, team_shots_y: np.array, team_goal_x: np.array, team_goal_y: np.array, team_shots_xg: np.array, xg_scale_factor: float, color: str, fig_name: str) -> None:
    """Plot the pitch with shots and goals.

    Args:
        ax (Axes): Axes object
        team_shots_x (np.array): x-coordinates of team shots
        team_shots_y (np.array): y-coordinates of team shots 
        team_goal_x (np.array): x-coordinates of team goals
        team_goal_y (np.array): y-coordinates of team goals
        team_shots_xg (np.array): xG values of team shots
        xg_scale_factor (float): Scale factor for xG values
        color (str): Color for the shots and goals
    """
    ax.scatter(team_shots_x, team_shots_y, color=color, alpha=0.5, label='Shots')
    star_sizes = xg_scale_factor * team_shots_xg
    ax.scatter(team_goal_x, team_goal_y, color=color, marker='*', s=star_sizes, edgecolor='black', label='Goals (scaled by xG)')

    for x, y, xg in zip(team_goal_x, team_goal_y, team_shots_xg):
        ax.text(
            x, y - 1.5,                      
            f"{xg:.4f}",                     
            ha='center', va='bottom', fontsize=8,
            color='black',
            zorder=5
        )

    plt.legend(loc='upper right')
    plt.title("Shots and Goals (scaled by xG)")
    plt.savefig(fig_name, dpi=300, bbox_inches='tight')
    plt.show()

def plot_gmm_components(gmm: GaussianMixture, ax: Axes, color: str, fig_name: str) -> None:
    """Plot GMM components as ellipses on the pitch.

    Args:
        gmm (GaussianMixture): Fitted Gaussian Mixture Model
        ax (Axes): Axes object
        pitch (mpl.Pitch): Pitch object
        color (str): Color for the ellipses 
    """

    for i in range(gmm.n_components):
            mean = gmm.means_[i]
            cov = gmm.covariances_[i]
            eig_val, eig_vec = np.linalg.eig(cov)
            angle = np.arctan2(*eig_vec[:,0][::-1])
            e = Ellipse(mean,
                        2*np.sqrt(eig_val[0]), 
                        2*np.sqrt(eig_val[1]), 
                        angle=np.degrees(angle),
                        color=color)
            e.set_alpha(0.5)
            ax.add_artist(e)

    plt.title("GMM Components for possession-related events")
    plt.savefig(fig_name, dpi=300, bbox_inches='tight')
    plt.show()

def evaluate_and_plot_gmm_pdf(ax: Axes, gmm: GaussianMixture, PITCH_X: int, PITCH_Y: int, cmap: str, fig_name: str) -> None:
    """Evaluate and plot the GMM probability density function (PDF) on the given axes.

    Args:
        ax (Axes): The axes to plot on.
        gmm (GaussianMixture): The fitted Gaussian Mixture Model.
        PITCH_X (int): The width of the pitch.
        PITCH_Y (int): The height of the pitch.
    """
    x_vals = np.linspace(0, PITCH_X, PITCH_X)
    y_vals = np.linspace(0, PITCH_Y, PITCH_Y)

    xx, yy = np.meshgrid(x_vals, y_vals)

    num_components = gmm.n_components
    density_components = np.zeros((yy.shape[0], xx.shape[1], num_components))

    grid_points = np.column_stack([xx.ravel(), yy.ravel()])

    for i, (mean, covariance, weight) in enumerate(
        zip(gmm.means_, gmm.covariances_, gmm.weights_)
    ):
        pdf_values = multivariate_normal.pdf(grid_points, mean=mean, cov=covariance)
        density_components[:, :, i] = weight * pdf_values.reshape(xx.shape)

    total_density = density_components.sum(axis=-1)

    ax.contourf(xx, yy, total_density, levels=10, cmap=cmap, alpha=0.8, antialiased=True)
    plt.title("GMM Probability Density Function (PDF) for possession-related events")
    plt.savefig(fig_name, dpi=300, bbox_inches='tight')
    plt.show()