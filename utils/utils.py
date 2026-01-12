from typing import Tuple, List
from xmlrpc import client

import matplotlib.pyplot as plt
import mplsoccer as mpl
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import ListedColormap
from matplotlib.patches import Ellipse, Patch
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture
import polars as pl
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif
import mlflow
from mlflow.tracking import MlflowClient
from model.data_classes import LGBMParams, OuterCVResults
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier


from utils.statics import (
    france_argentina_match_id,
    tracking_uri,
    lightgbm_model_name,
    xgboost_model_name,
)


def build_cmap(x: Tuple[int, int, int], y: Tuple[int, int, int]) -> ListedColormap:
    """Build cmap for Matplotlib

    Args:
        x (Tuple[int, int, int]): Tuple of RGB values
        y (Tuple[int, int, int]): Tuple of RGB values

    Returns:
        ListedColormap: ListedColorMap object for Matplotlib
    """
    r, g, b = x
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

    newcolors = np.vstack((cmp(np.linspace(0, 1, 128)), cmp_(np.linspace(1, 0, 128))))
    return ListedColormap(newcolors)


def invert_orientation(
    x: np.array, y: np.array, PITCH_X: int, PITCH_Y: int
) -> Tuple[np.array, np.array]:
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


def add_legend(
    ax: Axes,
    num_elements: int,
    colors: list[str],
    labels: list[str],
    markers: list[str] = None,
) -> None:
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
            Patch(
                facecolor=colors[i],
                edgecolor="black",
                alpha=0.5,
                label=labels[i],
                hatch=markers[i],
            )
            for i in range(num_elements)
        ]
    else:
        legend_elements = [
            Patch(facecolor=colors[i], edgecolor="black", alpha=0.5, label=labels[i])
            for i in range(num_elements)
        ]
    ax.legend(handles=legend_elements, loc="upper right")


def plot_player_positions(
    x: np.array,
    y: np.array,
    jerseys: list[str],
    names: list[str],
    pitch: mpl.Pitch,
    ax: Axes,
    color: str,
    fontsize: int,
    fig_name: str,
) -> None:
    # Plot player positions
    pitch.scatter(
        x, y, s=300, c=color, edgecolors="black", linewidth=1.5, ax=ax, zorder=3
    )

    # Add jersey numbers and player names
    for xi, yi, num, name in zip(x, y, jerseys, names):
        ax.text(
            xi,
            yi,
            str(num),
            ha="center",
            va="center",
            color="white",
            fontsize=10,
            fontweight="bold",
            zorder=4,
        )
        ax.text(
            xi,
            yi + 3.5,
            name,
            ha="center",
            va="bottom",
            color="black",
            fontsize=8,
            zorder=5,
        )

    plt.title("Average player positions based on events", fontsize=fontsize)
    plt.savefig(fig_name, dpi=300, bbox_inches="tight")
    plt.show()


def plot_pitch_with_shots(
    ax: Axes,
    team_shots_x: np.array,
    team_shots_y: np.array,
    team_goal_x: np.array,
    team_goal_y: np.array,
    team_shots_xg: np.array,
    xg_scale_factor: float,
    color: str,
    fig_name: str,
) -> None:
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
    ax.scatter(team_shots_x, team_shots_y, color=color, alpha=0.5, label="Shots")
    star_sizes = xg_scale_factor * team_shots_xg
    ax.scatter(
        team_goal_x,
        team_goal_y,
        color=color,
        marker="*",
        s=star_sizes,
        edgecolor="black",
        label="Goals (scaled by xG)",
    )

    for x, y, xg in zip(team_goal_x, team_goal_y, team_shots_xg):
        ax.text(
            x,
            y - 1.5,
            f"{xg:.4f}",
            ha="center",
            va="bottom",
            fontsize=8,
            color="black",
            zorder=5,
        )

    plt.legend(loc="upper right")
    plt.title("Shots and Goals (scaled by xG)")
    plt.savefig(fig_name, dpi=300, bbox_inches="tight")
    plt.show()


def plot_gmm_components(
    gmm: GaussianMixture, ax: Axes, color: str, fig_name: str
) -> None:
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
        angle = np.arctan2(*eig_vec[:, 0][::-1])
        e = Ellipse(
            mean,
            2 * np.sqrt(eig_val[0]),
            2 * np.sqrt(eig_val[1]),
            angle=np.degrees(angle),
            color=color,
        )
        e.set_alpha(0.5)
        ax.add_artist(e)

    plt.title("GMM Components for possession-related events")
    plt.savefig(fig_name, dpi=300, bbox_inches="tight")
    plt.show()


def evaluate_and_plot_gmm_pdf(
    ax: Axes, gmm: GaussianMixture, PITCH_X: int, PITCH_Y: int, cmap: str, fig_name: str
) -> None:
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

    ax.contourf(
        xx, yy, total_density, levels=10, cmap=cmap, alpha=0.8, antialiased=True
    )
    plt.title("GMM Probability Density Function (PDF) for possession-related events")
    plt.savefig(fig_name, dpi=300, bbox_inches="tight")
    plt.show()


def split_train_test(passes_df: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Split passes_df into train and test based on the match_id for the France-Argentina match

    Args:
        passes_df (pl.DataFrame): DataFrame containing pass events

    Returns:
        tuple[pl.DataFrame, pl.DataFrame]: Train and test DataFrames
    """
    train_df = passes_df.filter(pl.col("match_id") != france_argentina_match_id).drop(
        pl.col("match_id")
    )

    test_df = passes_df.filter(pl.col("match_id") == france_argentina_match_id).drop(
        pl.col("match_id")
    )

    return train_df, test_df


def plot_correlations(train_df: pl.DataFrame, numerical_cols: List[str]) -> None:
    """Plot correlation plot for continuous features

    Args:
        train_df (pl.DataFrame): Training DataFrame
        continuous_cols (List[str]): List of column names for numerical features
    """
    corr_matrix = train_df.select(numerical_cols).to_pandas().corr()

    plt.figure(figsize=(16, 9))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Matrix of Continuous Features")
    plt.show()


def plot_numerical_feature_distributions(
    train_df: pl.DataFrame, numerical_cols: List[str], rows: int = 3, cols: int = 3
) -> None:
    """Plot numerical feature distributions

    Args:
        train_df (pl.DataFrame): Train dataframe
        numerical_cols (List[str]): List of numerical columns
    """
    fig, ax = plt.subplots(rows, cols, figsize=(16, 9), tight_layout=True)
    ax = ax.ravel()

    for i, column in enumerate(numerical_cols):
        sns.histplot(
            train_df.select(column).to_series(), bins="auto", kde=True, ax=ax[i]
        )
        ax[i].set_title(f"Distribution of {column}")
        ax[i].set_xlabel(column)
        ax[i].set_ylabel("Frequency")
    plt.show()


def plot_categorical_feature_distributions(
    train_df: pl.DataFrame, categorical_cols: List[str]
) -> None:
    """Plot categorical feature distributions

    Args:
        train_df (pl.DataFrame): Train dataframe
        categorical_cols (List[str]): List of categorical columns
    """
    fig, ax = plt.subplots(2, 2, figsize=(10, 6), tight_layout=True)
    ax = ax.ravel()

    for i, column in enumerate(categorical_cols):
        sns.countplot(x=train_df.select(column).to_series(), ax=ax[i])
        ax[i].set_title(f"Count Plot of {column}")
        ax[i].set_xlabel(column)
        ax[i].set_ylabel("Count")
        if column == "body_part":
            ax[i].tick_params(axis="x", rotation=45)

    sns.countplot(x=train_df.select("outcome").to_series(), ax=ax[-1])
    ax[-1].set_xlabel("Outcome")
    ax[-1].set_ylabel("Count")
    ax[-1].set_title("Count Plot of Outcome")
    plt.show()


def plot_single_feature_distribution(
    train_df: pl.DataFrame, col: str, bins: int | str = 30, kde: bool = True
) -> None:
    """Plot single feature distribution plot

    Args:
        train_df (pl.DataFrame): Training DataFrame
        col (str): Column name to plot
        bins (int | str, optional): Number of bins or binning strategy. Defaults to 30.
        kde (bool, optional): Whether to show KDE curve. Defaults to True.
    """
    if type(bins) == str:
        bins = "auto"
        print("Bins is string type: defaulting to 'auto'")

    sns.histplot(train_df.select(pl.col(col)).to_series(), bins=bins, kde=kde)
    plt.show()


def plot_mutual_information(
    X_train: pl.DataFrame,
    y_train: pl.DataFrame,
    discrete_features: List[bool] | str = "auto",
):
    """Plot mutual information

    Args:
        X_train (pl.DataFrame): Training features
        y_train (pl.DataFrame): Training labels
        discrete_features (List[bool] | str, optional): How to handle discrete features. Defaults to "auto".
    """
    mi = mutual_info_classif(
        X_train, y_train, discrete_features=discrete_features, random_state=165
    )

    mi_df = pl.DataFrame({"Feature": X_train.columns, "Mutual Information": mi})
    mi_df = mi_df.sort(by="Mutual Information", descending=True)

    plt.figure(figsize=(8, 5))
    sns.barplot(x="Mutual Information", y="Feature", data=mi_df, palette="Blues_d")
    plt.title("Mutual Information of Features with Target")
    plt.xlabel("Mutual Information")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.show()


def get_parent_run_id_from_experiment(
    result: OuterCVResults | None, experiment_id: str
) -> str:
    """Get parent run ID from experiment

    Args:
        result (OuterCVResults | None): OuterCVResults object or None
        experiment_id (str): Experiment ID

    Returns:
        str: Parent run ID
    """
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    if result is not None:
        parent_run_id = client.get_run(result.run_ids[0])
    else:
        runs = client.search_runs(
            experiment_ids=[experiment_id],
            order_by=["attributes.start_time DESC"],
        )

        parent_run_id = next(
            run.info.run_id for run in runs if "mlflow.parentRunId" not in run.data.tags
        )

    return parent_run_id


def compute_generalisation_error_from_run_id_and_experiment_id(
    parent_run_id: str, experiment_id: str
) -> None:
    """Compute generalisation error from runs

    Args:
        parent_run_id (str): Parent run ID
        experiment_id (str): Experiment ID
    """
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    child_runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string=f"tags.mlflow.parentRunId = '{parent_run_id}'",
    )

    loss = [run.data.metrics["log_loss"] for run in child_runs]
    mean = np.mean(loss)
    std = np.std(loss, ddof=1)

    print(
        f"95% confidence interval for best estimate of generalisation: {mean} ± {1.96*std / np.sqrt(len(loss))}"
    )


def get_best_params_and_features_from_parent_run_id(
    parent_run_id: str,
) -> Tuple[dict, np.ndarray]:
    """Get params and features from parent run

    Args:
        parent_run_id (str): Parent run ID

    Returns:
        Tuple[dict, np.ndarray]: Best parameters and selected features
    """
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    parent_run = client.get_run(parent_run_id)

    lgbm_params = LGBMParams(**parent_run.data.params)
    best_params = lgbm_params.model_dump()
    best_features = np.array(parent_run.data.tags["selected_features"].split(","))

    return (best_params, best_features)


def get_registered_model(
    model_type: str, model_registry_name: str, version: str = "latest"
) -> XGBClassifier | LGBMClassifier:
    """Get registered model from MLFlow

    Args:
        model_type (str): Type of the model
        model_registry_name (str): Name of the model registry
        version (str, optional): Version of the model. Defaults to "latest".

    Raises:
        ValueError: If the model type is unsupported

    Returns:
        XGBClassifier | LGBMClassifier: Trained model
    """
    log_fn_mapping = {
        xgboost_model_name: mlflow.xgboost.load_model,
        lightgbm_model_name: mlflow.lightgbm.load_model,
    }

    log_fn = log_fn_mapping.get(model_type)
    if log_fn is None:
        raise ValueError(f"Unsupported model type: {model_type}")

    trained_model = log_fn(model_uri=f"models:/{model_registry_name}/{version}")
    return trained_model
