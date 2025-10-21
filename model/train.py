from xgboost import XGBClassifier
import optuna
from sklearn.pipeline import pipeline

class XGBoostTrainer():

    def __init__(self):
        pass

    def tune_hyperparameters(self, X, y):
        sampler = 