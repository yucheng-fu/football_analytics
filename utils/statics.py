import numpy as np

# Pitch dimensions and grid settings
GRID_X, GRID_Y = 16, 12
PITCH_X, PITCH_Y = 105, 68  # Standard pitch dimensions in meters. https://mplsoccer.readthedocs.io/en/latest/gallery/pitch_setup/plot_pitches.html

# bins
x_bins = np.linspace(0, PITCH_X, GRID_X + 1)
y_bins = np.linspace(0, PITCH_Y, GRID_Y + 1)