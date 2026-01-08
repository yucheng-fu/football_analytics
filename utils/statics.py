import numpy as np

# Pitch dimensions and grid settings
GRID_X, GRID_Y = 16, 12
PITCH_X, PITCH_Y = (
    120,
    80,
)  # Standard pitch dimensions in meters. https://mplsoccer.readthedocs.io/en/latest/gallery/pitch_setup/plot_pitches.html

# bins
x_bins = np.linspace(0, PITCH_X, GRID_X + 1)
y_bins = np.linspace(0, PITCH_Y, GRID_Y + 1)

# France-Argentina match id
france_argentina_match_id = 7580

# plotting settings
figsize = (9, 6)

# expected goal scale factor
xg_scale_factor = 400

# model names
xgboost_model_name = "xgboost"
lightgbm_model_name = "lightgbm"

# mlflow
tracking_uri = "http://127.0.0.1:8080/"
EVALUTION_EXPERIMENT_ID = "220281539845020993"
FINAL_MODELS_EXPERIMENT_ID = "441823692328196814"
NESTED_CV_EVAL_EXPERIMENT_ID = "416632419703074027"
MODEL_SELECTION_EXPERIMENT_ID = "428168727943543105"
