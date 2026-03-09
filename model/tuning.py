from typing import Callable, Tuple

import polars as pl
import pickle
from lightgbm import LGBMClassifier
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.feature_selection import RFE
from sklearn.metrics import log_loss
from sklearn.pipeline import Pipeline
import optuna
import matplotlib.pyplot as plt
from utils.statics import lightgbm_model_name, tracking_uri
import mlflow
from mlflow.entities import Run
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from mlflow.utils.mlflow_tags import MLFLOW_PARENT_RUN_ID
import numpy as np
import logging
from model.data_classes import OuterCVResults
from optuna.integration import LightGBMPruningCallback
from lightgbm.callback import early_stopping
from feature_engineering.openfe import OpenFE
import pandas as pd


class ModelCVEvaluator:
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    def __init__(
        self,
        model_type: str,
        n_inner_splits: int = 5,
        n_outer_splits: int = 10,
        n_trials: int = 20,
        run_name: str = "baseline_hyperparameter_tuning",
        experiment_name: str = "Model selection and hyperparameter tuning",
        use_initial_tuning: bool = True,
        log_hyperparameter_trials: bool = False,
        categorical_columns: list[str] | None = None,
    ):
        self.model_type = model_type
        self.n_inner_splits = n_inner_splits
        self.n_outer_splits = n_outer_splits
        self.n_trials = n_trials
        self.run_name = run_name
        self.experiment_name = experiment_name
        self.use_initial_tuning = use_initial_tuning
        self.log_hyperparameter_trials = log_hyperparameter_trials
        self.categorical_columns = (
            list(categorical_columns) if categorical_columns is not None else []
        )
        self.inner_cv = StratifiedKFold(
            n_splits=self.n_inner_splits, shuffle=True, random_state=165
        )
        self.outer_cv = StratifiedKFold(
            n_splits=self.n_outer_splits, shuffle=True, random_state=165
        )

    def _categorical_feature_names(self, X_pd: pd.DataFrame) -> list[str]:
        names: list[str] = []
        for col in self.categorical_columns:
            if isinstance(col, str):
                if col in X_pd.columns:
                    names.append(col)
                continue
            if isinstance(col, int) and 0 <= col < X_pd.shape[1]:
                names.append(X_pd.columns[col])
        return names

    def _apply_categorical_dtypes(self, X_pd: pd.DataFrame) -> pd.DataFrame:
        # Ensure all text-like columns are categorical for LightGBM/RFE compatibility.
        object_like_cols = X_pd.select_dtypes(include=["object", "string"]).columns
        for name in object_like_cols:
            X_pd[name] = X_pd[name].astype("category")

        for name in self._categorical_feature_names(X_pd):
            X_pd[name] = X_pd[name].astype("category")
        return X_pd

    def setup_mlflow(self) -> None:
        """
        Sets up tracking uri and experiment for MLFlow

        Returns:
            None
        """
        self.logger.info(
            f"""Starting training with model {self.model_type} with the following configuration:
        - {self.n_inner_splits} inner splits
        - {self.n_outer_splits} outer splits
        - {self.n_trials} trials"""
        )
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        logging.getLogger("mlflow").setLevel(logging.ERROR)
        logging.getLogger("optuna").setLevel(logging.WARNING)

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name=self.experiment_name)

    def fetch_model(self) -> LGBMClassifier:
        """
        Return a fresh estimator instance for the configured model type.

        Returns:
            LGBMClassifier: An uninitialized classifier corresponding to self.model_type.

        Raises:
            ValueError: If self.model_type is not a supported model name.
        """
        if self.model_type == lightgbm_model_name:
            return LGBMClassifier(verbose=-1, importance_type="gain", random_state=165)

        raise ValueError(f"Unsupported model type: {self.model_type}")

    def fetch_param_suggestions(
        self, trial: optuna.trial.Trial, n_features: int
    ) -> dict:
        """
        Generate hyperparameter suggestions for the configured model using an Optuna trial.

        Args:
            trial (optuna.trial.Trial): An Optuna Trial object used to sample hyperparameter values.

        Returns:
            dict: A dictionary of hyperparameter names and sampled values suitable for the selected model type

        Raises:
            ValueError: If self.model_type is not one of the supported model names.
        """
        if self.model_type == lightgbm_model_name:
            max_depth = trial.suggest_int("max_depth", 3, 100)
            max_num_leaves = min(
                2**max_depth - 1, 512
            )  # less than 2^(max_depth), cap at 512
            num_leaves = trial.suggest_int("num_leaves", 7, max_num_leaves)

            # Model paramteres
            params = {
                "n_estimators": trial.suggest_int(
                    "n_estimators", 100, 1000
                ),  # number of trees
                "max_depth": max_depth,  # maximum depth of each tree
                "num_leaves": num_leaves,  # number of leaves in one tree
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.01, 0.2
                ),  # step size for optimisation
                "subsample": trial.suggest_float(
                    "subsample", 0.5, 1.0
                ),  # fraction of samples to be used for each tree
                "colsample_bytree": trial.suggest_float(
                    "colsample_bytree", 0.5, 1.0
                ),  # fraction of features used per tree
                "min_child_samples": trial.suggest_int(
                    "min_child_samples", 20, 100
                ),  # smallest number of data points a leaf can have
                "bagging_freq": trial.suggest_int(
                    "bagging_freq", 0, 10
                ),  # bagging frequency
                "reg_alpha": trial.suggest_float(
                    "reg_alpha", 0.0, 1.0
                ),  # L1 regularisation
                "reg_lambda": trial.suggest_float(
                    "reg_lambda", 0.0, 1.0
                ),  # L2 regularisation
                "metric": "binary_logloss",  # evaluation metric
                "random_state": 165,  # seed for reproducibility
                "verbose": -1,  # suppress warnings and info
            }

            # RFE
            params["n_features_to_select"] = trial.suggest_int(
                "n_features_to_select", 5, n_features
            )

            return params

        raise ValueError(f"Unsupported model type: {self.model_type}")

    def log_model(
        self,
        final_model: LGBMClassifier,
        X_data: pd.DataFrame,
        output: np.ndarray,
    ) -> None:
        """Log model in outer fold run

        Args:
            final_model (LGBMClassifier): Trained model
            X_data (pd.DataFrame): One outer fold of X
            output (np.ndarray): Predictions from final_model

        Raises:
            ValueError: If self.model_type is not a supported model name.
        """
        signature = infer_signature(X_data, output)
        input_example = X_data.head(10)

        log_fn_mapping = {
            lightgbm_model_name: mlflow.lightgbm.log_model,
        }

        log_fn = log_fn_mapping.get(self.model_type)
        if log_fn is None:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        log_fn(
            final_model,
            name=self.model_type,
            signature=signature,
            input_example=input_example,
        )

    def append_and_log_metrics_and_params(
        self,
        outer_cv_results: OuterCVResults,
        selected_features: np.ndarray,
        outer_fold_log_loss: float,
        best_params: dict,
        run: Run,
        study: optuna.study.Study | None,
    ) -> None:
        """Append and log metrics and params to the MLFlow run

        Args:
            outer_cv_results (OuterCVResults): OuterCVResults object to append results to
            selected_features (np.ndarray): array of selected features
            outer_fold_log_loss (float): log_loss evaluated on the outer fold validation set
            best_params (dict): best hyperparameters found in the outer fold
            run (Run): MLFlow run object
            study (optuna.study.Study | None): Optuna study object (optional)

        Returns:
            None
        """

        outer_cv_results.scores.append(outer_fold_log_loss)
        outer_cv_results.params.append(best_params)
        outer_cv_results.features.append(selected_features)
        outer_cv_results.run_ids.append(run.info.run_id)
        outer_cv_results.experiment_ids.append(run.info.experiment_id)

        mlflow.set_tag("selected_features", ",".join(map(str, selected_features)))
        mlflow.log_metric("log_loss", outer_fold_log_loss)
        mlflow.log_params(best_params)
        if study is not None:
            with open("optuna_study.pkl", "wb") as f:
                pickle.dump(study, f)
            mlflow.log_artifact("optuna_study.pkl")

    def _mlflow_callback(
        self, outer_fold_run_id: str, experiment_name: str
    ) -> Callable[[optuna.Study, optuna.trial.Trial], None]:
        """MLFlow callback function for logging parallel Optuna runs as nested runs in MLFlow

        Args:
            outer_fold_run_id (str): MLFlow run id of the outer fold
            experiment_name (str): Name of the experiment

        Returns:
            Callable[[optuna.Study, optuna.trial.Trial]]: _callback method
        """

        def _callback(study: optuna.Study, trial: optuna.trial.Trial):
            """Callback function for Optuna

            Args:
                study (optuna.Study): _description_
                trial (optuna.trial.Trial): _description_
            """

            client = MlflowClient()
            experiment = mlflow.get_experiment_by_name(experiment_name)
            experiment_id = experiment.experiment_id

            tags = {
                MLFLOW_PARENT_RUN_ID: outer_fold_run_id,
                "mlflow.runName": f"Trial_{trial.number}",
                "status": str(trial.state),
            }

            # Create a nested child run
            trial_run = client.create_run(
                experiment_id=experiment_id,
                tags=tags,
            )
            trial_run_id = trial_run.info.run_id
            for key, value in trial.params.items():
                client.log_param(
                    trial_run_id, key, value
                )  # Apparently you cannot just log all parameters for the trial
            if trial.value is not None:
                client.log_metric(trial_run_id, "log_loss", trial.value)

            # Kill the run at teh end
            client.set_terminated(trial_run_id, "FINISHED")

        return _callback

    # def get_features(self, X_train_outer: pl.DataFrame, rfecv: RFE) -> np.ndarray:
    #     """Get features from feature selection using RFECV

    #     Args:
    #         X_train_outer (pl.DataFrame): Outer fold of X_train
    #         rfecv (RFECV): RFECV object

    #     Returns:
    #         np.ndarray: array of selected features
    #     """
    #     selected_feature_mask = rfecv.get_support()
    #     selected_features = np.array(X_train_outer.columns)[selected_feature_mask]
    #     return selected_features

    def eval_validation_loss(
        self,
        X_val_outer: pd.DataFrame,
        y_val_outer: np.ndarray,
        selected_features: np.ndarray,
        final_model: LGBMClassifier,
        category_schema: dict[str, pd.Index],
    ) -> float:
        """Evaluate model on validation set

        Args:
            X_val_outer (pd.DataFrame): X_val outer fold
            y_val_outer (np.ndarray): y_val outer fold
            rfecv (RFECV | None): RFECV obejct (optional)
            final_model (LGBMClassifier): trained final model
            category_schema (dict[str, pd.Index]): Training-time categories for categorical features

        Returns:
            float: log_loss evaluted on the outer fold validation set
        """
        X_val_outer_selected = X_val_outer[selected_features].copy()
        for col, cats in category_schema.items():
            if col in X_val_outer_selected.columns:
                X_val_outer_selected[col] = pd.Categorical(
                    X_val_outer_selected[col], categories=cats
                )

        y_pred_proba = final_model.predict_proba(X_val_outer_selected)

        outer_fold_log_loss = log_loss(y_val_outer, y_pred_proba)

        self.log_model(
            final_model=final_model,
            X_data=X_val_outer_selected,
            output=y_pred_proba,
        )

        return outer_fold_log_loss

    def _objective(
        self,
        trial: optuna.trial.Trial,
        X_train_outer: pd.DataFrame,
        y_train_outer: np.ndarray,
    ) -> np.ndarray:
        """Objective function for Optuna hyperparameter tuning

        Args:
            trial (optuna.trial.Trial): Optuna trial
            X_train_outer (pd.DataFrame): X_train outer fold split
            y_train_outer (np.ndarray): y_train outer fold split

        Returns:
            np.ndarray: Maximum mean cross-validation score obtained by the Optuna trial across the inner folds
        """
        n_features = X_train_outer.shape[1]
        all_params = self.fetch_param_suggestions(trial, n_features)

        fit_params = all_params.copy()
        n_features_to_select = fit_params.pop("n_features_to_select")
        inner_fold_scores = []

        X_pd = X_train_outer
        y_np = y_train_outer

        for inner_train_idx, inner_val_idx in self.inner_cv.split(X_pd, y_np):
            X_train_inner = X_pd.iloc[inner_train_idx]
            X_val_inner = X_pd.iloc[inner_val_idx]
            y_train_inner, y_val_inner = y_np[inner_train_idx], y_np[inner_val_idx]

            base_estimator = self.fetch_model()
            base_estimator.set_params(**fit_params)

            # Fit RFE
            rfe = RFE(
                estimator=base_estimator, n_features_to_select=n_features_to_select
            )
            rfe.set_output(transform="pandas")
            # sklearn RFE expects numeric arrays; encode categoricals for feature ranking.
            X_train_inner_rfe = X_train_inner.copy()
            for col in X_train_inner_rfe.select_dtypes(include=["category"]).columns:
                X_train_inner_rfe[col] = X_train_inner_rfe[col].cat.codes

            rfe.fit(X_train_inner_rfe, y_train_inner)
            selected_features = np.array(X_train_inner.columns)[rfe.support_]
            X_train_inner_selected = X_train_inner[selected_features]
            X_val_inner_selected = X_val_inner[selected_features]

            current_categories = X_train_inner_selected.select_dtypes(
                include=["category"]
            ).columns.tolist()

            # Train model with selected features and hyperparameters, and evaluate on the inner validation fold
            model = self.fetch_model()
            model.set_params(**fit_params)

            model.fit(
                X_train_inner_selected,
                y_train_inner,
                eval_set=[(X_val_inner_selected, y_val_inner)],
                eval_metric="binary_logloss",
                callbacks=[
                    (LightGBMPruningCallback(trial, "binary_logloss")),
                    early_stopping(stopping_rounds=50, verbose=False),
                ],
                categorical_feature=(
                    current_categories if current_categories else "auto"
                ),
            )

            inner_fold_scores.append(model.best_score_["valid_0"]["binary_logloss"])

        mean_score = np.mean(inner_fold_scores)

        return mean_score

    def hyperparameter_tuning(
        self,
        X_train_outer: pd.DataFrame,
        y_train_outer: np.ndarray,
        callback_fn: Callable[[optuna.Study, optuna.trial.Trial], None],
    ) -> Tuple[optuna.study.Study, dict]:
        """Function for performing hyperparameter tuning with optuna.

        Args:
            X_train_outer (pd.DataFrame): Outer fold of X_train
            y_train_outer (np.ndarray): Outer fold of y_train
            callback_fn (Callable[[optuna.Study, optuna.trial.Trial], None]): Callback function for optuna study

        Returns:
            Tuple[optuna.study.Study, dict]: Optuna study object and dictionary of best hyperparameters
        """
        self.logger.info("Starting full hyperparameter tuning...")
        study = optuna.create_study(direction="minimize")

        X_train_outer_pd = self._apply_categorical_dtypes(X_train_outer.copy())
        y_train_outer_np = y_train_outer

        study.optimize(
            lambda trial: self._objective(trial, X_train_outer_pd, y_train_outer_np),
            n_trials=self.n_trials,
            n_jobs=-1,
            show_progress_bar=True,
            callbacks=[callback_fn],
        )

        best_trial = study.best_trial
        best_params = best_trial.params.copy()

        return study, best_params

    # def feature_selection(
    #     self,
    #     X_train_outer: pl.DataFrame,
    #     y_train_outer: pl.DataFrame,
    #     init_params: dict,
    # ) -> np.ndarray:
    #     """Feature selection using RFECV

    #     Args:
    #         X_train_outer (pl.DataFrame): Outer fold of X_train
    #         y_train_outer (pl.DataFrame): Outer fold of y_train
    #         init_params (dict): Hyperparameters found in the initial tuning

    #     Returns:
    #         np.ndarray: Array of selected features
    #     """
    #     self.logger.info("Starting feature selection using RFECV...")
    #     rfe_estimator = self.fetch_model()
    #     rfe_estimator.set_params(**init_params)

    #     X_pd = X_train_outer.to_pandas()
    #     for idx in self.categorical_columns:
    #         X_pd.iloc[:, idx] = X_pd.iloc[:, idx].astype("category")

    #     y_np = y_train_outer.to_numpy().ravel()

    #     rfecv = RFECV(
    #         estimator=rfe_estimator,
    #         step=1,
    #         cv=self.inner_cv,
    #         scoring="neg_log_loss",
    #         n_jobs=-1,
    #     )
    #     rfecv.fit(X_pd, y_np)

    #     selected_features = self.get_features(X_train_outer=X_train_outer, rfecv=rfecv)

    #     return selected_features

    def fit_model(
        self,
        X_train_outer: pd.DataFrame,
        y_train_outer: np.ndarray,
        best_params: dict,
    ) -> Tuple[LGBMClassifier, np.ndarray, dict[str, pd.Index]]:
        """Fit model using the best hyperparameters and selected features

        Args:
            X_train_outer (pd.DataFrame): Outer fold of X_train
            y_train_outer (np.ndarray): Outer fold of y_train
            best_params (dict): Best hyperparameters found in the outer fold
            selected_features (np.ndarray): Array of selected features from RFECV

        Returns:
            Tuple[LGBMClassifier, np.ndarray, dict[str, pd.Index]]: Trained final model, selected features, and categorical schema
        """
        self.logger.info("Fitting final model with best hyperparameters...")

        params = best_params.copy()
        n_features_to_select = params.pop("n_features_to_select")

        X_pd = self._apply_categorical_dtypes(X_train_outer.copy())
        y_np = y_train_outer

        # Fit RFE with the best hyperparameters to get the selected features for the outer fold
        base_estimator = self.fetch_model()
        base_estimator.set_params(**params)
        rfe = RFE(base_estimator, n_features_to_select=n_features_to_select)
        rfe.set_output(transform="pandas")
        X_pd_rfe = X_pd.copy()
        for col in X_pd_rfe.select_dtypes(include=["category"]).columns:
            X_pd_rfe[col] = X_pd_rfe[col].cat.codes
        rfe.fit(X_pd_rfe, y_np)

        selected_features = np.array(X_pd.columns)[rfe.support_]
        X_train_outer_selected = X_pd[selected_features]
        current_categories = X_train_outer_selected.select_dtypes(
            include=["category"]
        ).columns.tolist()
        category_schema = {
            col: X_train_outer_selected[col].cat.categories
            for col in current_categories
        }

        final_model = self.fetch_model()
        final_model.set_params(**params)

        final_model.fit(
            X=X_train_outer_selected,
            y=y_np,
            categorical_feature=current_categories if current_categories else "auto",
        )

        return final_model, selected_features, category_schema

    def get_generalisation_error(
        self, X_train: pl.DataFrame, y_train: pl.DataFrame
    ) -> OuterCVResults:
        """Use nested cross-validation to obtain an unbiased estimate of the loss. Hyperparameter tuning is performed in the inner cross-validation loop.

        Args:
            X_train (pl.DataFrame): Input features used for training
            y_train (pl.DataFrame): Ground truths used for training

        Returns:
            OuterCVResults: Outer cross-validation results
        """
        outer_cv_results = OuterCVResults()
        self.setup_mlflow()
        X_train_pd = X_train.to_pandas()
        y_train_np = y_train.to_numpy().ravel()
        with mlflow.start_run(
            run_name=f"{self.run_name}_{self.model_type}"
        ) as parent_run:
            outer_cv_results.parent_run_id = parent_run.info.run_id
            for i, (train_idx, val_idx) in enumerate(
                self.outer_cv.split(X_train_pd, y_train_np)
            ):
                X_train_outer = X_train_pd.iloc[train_idx].copy()
                X_val_outer = X_train_pd.iloc[val_idx].copy()
                y_train_outer = y_train_np[train_idx]
                y_val_outer = y_train_np[val_idx]

                with mlflow.start_run(nested=True, run_name=f"Outer_fold_{i+1}") as run:
                    parent_id = run.info.run_id
                    callback_fn = (
                        self._mlflow_callback(
                            outer_fold_run_id=parent_id,
                            experiment_name=self.experiment_name,
                        )
                        if self.log_hyperparameter_trials
                        else lambda study, trial: None
                    )

                    # # Log the OpenFE object so you know what features were created
                    # with open(f"pickles/ofe_fold_{i}.pkl", "wb") as f:
                    #     pickle.dump(ofe_obj, f)
                    # mlflow.log_artifact(f"pickles/ofe_fold_{i}.pkl")

                    # 3. Perform full hyperparamter tuning
                    _, best_params = self.hyperparameter_tuning(
                        X_train_outer=X_train_outer,
                        y_train_outer=y_train_outer,
                        callback_fn=callback_fn,
                    )

                    # 4. Fit final model with best hyperparameters on the entire outer training fold
                    final_model, selected_features, category_schema = self.fit_model(
                        X_train_outer=X_train_outer,
                        y_train_outer=y_train_outer,
                        best_params=best_params,
                    )

                    # 5. Evaluate final model on the outer validaiton fold and log final model to model registry
                    outer_fold_log_loss = self.eval_validation_loss(
                        X_val_outer=X_val_outer,
                        y_val_outer=y_val_outer,
                        selected_features=selected_features,
                        final_model=final_model,
                        category_schema=category_schema,
                    )
                    # 6. Log metrics and parameters to MLFlow
                    self.append_and_log_metrics_and_params(
                        outer_cv_results=outer_cv_results,
                        selected_features=selected_features,
                        outer_fold_log_loss=outer_fold_log_loss,
                        best_params=best_params,
                        run=run,
                        study=None,
                    )

            mean_score = np.mean(outer_cv_results.scores)
            std_score = np.std(outer_cv_results.scores)
            mlflow.log_metric("mean_log_loss", mean_score)
            mlflow.log_metric("std_log_loss", std_score)

        mlflow.end_run()

        return outer_cv_results


class ModelParamTuner(ModelCVEvaluator):

    def setup_mlflow(self):
        """
        Sets up tracking uri and experiment for MLFlow

        Returns:
            None
        """
        self.logger.info(
            f"""Starting training with model {self.model_type} with the following configuration:
        - {self.n_inner_splits} inner splits
        - {self.n_trials} trials"""
        )
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        logging.getLogger("mlflow").setLevel(logging.ERROR)
        logging.getLogger("optuna").setLevel(logging.WARNING)

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name=self.experiment_name)

    def plot_loss_curve(self, train_loss: list, valid_loss: list) -> None:
        """Plot loss curve for training

        Args:
            train_loss (list): Training loss values
            valid_loss (list): Validation loss values
        """
        fig = plt.figure(figsize=(10, 6))
        epochs = range(1, len(train_loss) + 1)
        plt.plot(epochs, train_loss, "r", label="Training loss")
        plt.plot(epochs, valid_loss, "b", label="Validation loss")
        plt.title(f"Training and Validation Loss - {self.model_type}")
        plt.xlabel("Boosting Rounds")
        plt.ylabel("Log Loss")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.5)

        mlflow.log_figure(fig, artifact_file=f"{self.model_type}_loss_curve.png")

    def fit_and_evaluate_model(
        self,
        X_train: pl.DataFrame,
        y_train: pl.DataFrame,
        X_val: pl.DataFrame,
        y_val: pl.DataFrame,
        best_params: dict,
        selected_features: np.ndarray,
    ) -> LGBMClassifier:
        """Fit final model using best hyerparameters and selected features

        Args:
            X_train (pl.DataFrame): Training set
            y_train (pl.DataFrame): Training labels
            X_val (pl.DataFrame): Validation set
            y_val (pl.DataFrame): Validation labels
            best_params (dict): Best hyperparameters found in the outer fold
            selected_features (np.ndarray): Array of selected features from RFECV
        Returns:
            LGBMClassifier: Trained final model
        """
        self.logger.info("Fitting final model with best hyperparameters...")
        final_model = self.fetch_model()
        final_model.set_params(**best_params)

        results = {}

        if self.model_type == lightgbm_model_name:
            final_model.fit(
                X_train[selected_features],
                y_train,
                eval_set=[
                    (X_train[selected_features], y_train),
                    (X_val[selected_features], y_val),
                ],
                eval_names=["train", "valid"],
                eval_metric="binary_logloss",
                callbacks=[lgb.record_evaluation(results)],
                categorical_feature=(
                    self.categorical_columns if self.categorical_columns else ""
                ),
            )
            train_loss = results["train"]["binary_logloss"]
            valid_loss = results["valid"]["binary_logloss"]

        if train_loss and valid_loss:
            self.plot_loss_curve(train_loss, valid_loss)

        return final_model

    # def tune_and_train(self, X: pl.DataFrame, y: pl.DataFrame) -> OuterCVResults:
    #     """Tune and train on the entire X_train set. Use one-layer CV for hyperparameter optimisation.

    #     Args:
    #         X_train (pl.DataFrame): Input features used for training
    #         y_train (pl.DataFrame): Ground truths used for training

    #     Returns:
    #         OuterCVResults: Outer cross-validation results
    #     """
    #     outer_cv_results = OuterCVResults()
    #     self.setup_mlflow()

    #     X_train, X_val, y_train, y_val = train_test_split(
    #         X, y, test_size=0.2, stratify=y, random_state=42
    #     )

    #     with mlflow.start_run(run_name=f"{self.run_name}_{self.model_type}") as run:

    #         parent_id = run.info.run_id
    #         outer_cv_results.parent_run_id = parent_id
    #         callback_fn = self._mlflow_callback(
    #             outer_fold_run_id=parent_id,
    #             experiment_name=self.experiment_name,
    #         )

    #         # 1. Perform an initial tuning using 20 trials
    #         init_params = self.init_hyperparameter_tuning(
    #             X_train_outer=X_train, y_train_outer=y_train
    #         )

    #         # 2. Perform feature selection using RFECV
    #         selected_features = self.feature_selection(
    #             X_train_outer=X_train,
    #             y_train_outer=y_train,
    #             init_params=init_params,
    #         )

    #         # 3. Perform full hyperparamter tuning
    #         study, best_params = self.hyperparameter_tuning(
    #             X_train_outer=X_train,
    #             y_train_outer=y_train,
    #             selected_features=selected_features,
    #             callback_fn=callback_fn,
    #         )

    #         # 4. Fit final model with best hyperparameters on the train set
    #         final_model = self.fit_and_evaluate_model(
    #             X_train=X_train,
    #             y_train=y_train,
    #             X_val=X_val,
    #             y_val=y_val,
    #             best_params=best_params,
    #             selected_features=selected_features,
    #         )

    #         # 5. Evaluate final model on the outer validation fold and log final model to model registry
    #         outer_fold_log_loss = self.eval_validation_loss(
    #             X_val_outer=X_val,
    #             y_val_outer=y_val,
    #             selected_features=selected_features,
    #             final_model=final_model,
    #         )

    #         self.append_and_log_metrics_and_params(
    #             outer_cv_results=outer_cv_results,
    #             selected_features=selected_features,
    #             outer_fold_log_loss=outer_fold_log_loss,
    #             best_params=best_params,
    #             run=run,
    #             study=study,
    #         )

    #     mlflow.end_run()

    #     return outer_cv_results


if __name__ == "__main__":
    pass
