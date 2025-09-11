from utils.statics import (GRID_X, GRID_Y, PITCH_X, PITCH_Y)
import numpy as np


def get_grid_cell(x, y, x_bins, y_bins):
    x_idx = np.digitize(x, x_bins) - 1
    y_idx = np.digitize(y, y_bins) - 1
    return x_idx, y_idx