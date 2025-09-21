import numpy as np
from utils.utils import build_cmap  

# Pitch dimensions and grid settings
GRID_X, GRID_Y = 16, 12
PITCH_X, PITCH_Y = 120, 80  # Standard pitch dimensions in meters. https://mplsoccer.readthedocs.io/en/latest/gallery/pitch_setup/plot_pitches.html

# bins
x_bins = np.linspace(0, PITCH_X, GRID_X + 1)
y_bins = np.linspace(0, PITCH_Y, GRID_Y + 1)

# Color statics
blue, red = (44,123,182), (215,25,28)
blue = [x/256 for x in blue]
red = [x/256 for x in red]
diverging = build_cmap(blue, red)
diverging_r = build_cmap(red, blue)

# plotting settings
figsize = (9, 6)

# expected goal scale factor
xg_scale_factor = 400  

# gmm components
gmm_n_components = 8