from typing import Callable, Tuple, Optional, Any, List
import tempfile
import polars as pl
import os
import random
import pickle
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFE
from sklearn.metrics import log_loss
from sklearn import clone
import optuna
import matplotlib.pyplot as plt
from utils.statics import lightgbm_model_name, tracking_uri
from utils.utils import plot_feature_importance, plot_calibration_curve
import mlflow
from mlflow.entities import Run
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from mlflow.utils.mlflow_tags import MLFLOW_PARENT_RUN_ID
import numpy as np
import logging
from model.data_classes import OuterCVResults
from optuna.integration import LightGBMPruningCallback
from optuna.visualization import plot_param_importances
from lightgbm.callback import early_stopping
import pandas as pd
from feature_engineering.ColumnTransformer import ColumnTransformer
from feature_engineering.RowWiseTransformations import RowWiseTransformations
from feature_engineering.OpenFETransformations import OpenFETransformations
from feature_engineering.OpenFE.FeatureGenerator import Node
from feature_engineering.OpenFE.utils import tree_to_formula


class ModelCVEvaluator:
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    def __init__(
        self,
        model_type: str,
        row_wise_transformations: RowWiseTransformations | None,
        column_wise_transformations: ColumnTransformer | None,
        open_fe_transformations: OpenFETransformations | None,
        n_inner_splits: int = 5,
        n_outer_splits: int = 10,
        n_trials: int = 20,
        use_hyperparameter_tuning: bool = False,
        use_feature_selection: bool = False,
        use_feature_engineering: bool = False,
        use_ofe: bool = False,
        run_name: str = "baseline_hyperparameter_tuning",
        experiment_name: str = "Model selection and hyperparameter tuning",
        log_hyperparameter_trials: bool = False,
        log_parameter_importance: bool = False,
        log_feature_importance: bool = False,
        log_calibration_curve: bool = False,
        categorical_columns: list[str] | None = None,
        ohe_columns: list[str] | None = None,
    ):
        self.model_type = model_type
        self.row_wise_transformations = row_wise_transformations
        self.column_wise_transformations = column_wise_transformations
        self.open_fe_transformations = open_fe_transformations
        self.n_inner_splits = n_inner_splits
        self.n_outer_splits = n_outer_splits
        self.n_trials = n_trials
        self.use_hyperparameter_tuning = use_hyperparameter_tuning
        self.use_feature_selection = use_feature_selection
        self.use_feature_engineering = use_feature_engineering
        self.use_ofe = use_ofe
        self.run_name = run_name
        self.experiment_name = experiment_name
        self.log_hyperparameter_trials = log_hyperparameter_trials
        self.log_parameter_importance = log_parameter_importance
        self.log_feature_importance = log_feature_importance
        self.log_calibration_curve = log_calibration_curve
        self.categorical_columns = (
            categorical_columns if categorical_columns is not None else []
        )
        self.ohe_columns = ohe_columns if ohe_columns is not None else []
        self.seed = 165
        self.n_jobs = os.cpu_count() or 1
        self.inner_cv = StratifiedKFold(
            n_splits=self.n_inner_splits, shuffle=True, random_state=self.seed
        )
        self.outer_cv = StratifiedKFold(
            n_splits=self.n_outer_splits, shuffle=True, random_state=self.seed
        )
        self._set_global_seed()

    def _set_global_seed(self):
        """Set global random seeds used by NumPy and Python's random module."""
        np.random.seed(self.seed)
        random.seed(self.seed)

    def _categorical_feature_names(self, X_pd: pd.DataFrame) -> list[str]:
        """Return categorical column names that are present in a dataframe.

        Args:
            X_pd (pd.DataFrame): Input dataframe.

        Returns:
            list[str]: Valid categorical column names that exist in ``X_pd``.
        """
        names: list[str] = []
        for col in self.categorical_columns:
            # If column in self.categorical_columns is string and in dataframe column, append to names
            if isinstance(col, str):
                if col in X_pd.columns:
                    names.append(col)
        return names

    def _apply_categorical_dtypes(self, X_pd: pd.DataFrame) -> pd.DataFrame:
        """Return a copy of ``X_pd`` with categorical dtypes enforced.

        Args:
            X_pd (pd.DataFrame): Input dataframe.

        Returns:
            pd.DataFrame: Copied dataframe with object/string columns cast to
                ``category`` and configured categorical columns enforced.
        """
        X_pd_copy = X_pd.copy()

        # Ensure all text-like columns are categorical for LightGBM/RFE compatibility.
        object_like_cols = X_pd_copy.select_dtypes(include=["object", "string"]).columns
        for name in object_like_cols:
            X_pd_copy[name] = X_pd_copy[name].astype("category")

        for name in self._categorical_feature_names(X_pd_copy):
            X_pd_copy[name] = X_pd_copy[name].astype("category")
        return X_pd_copy

    def _setup_mlflow(self) -> None:
        """Configure MLflow and logging verbosity for a training session."""
        self.logger.info(
            f"""Starting training with model {self.model_type} with the following configuration:
        - {self.n_inner_splits} inner splits
        - {self.n_outer_splits} outer splits
        - {self.n_trials} trials"""
        )
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        logging.getLogger("mlflow").setLevel(logging.ERROR)
        logging.getLogger("optuna").setLevel(logging.WARNING)
        logging.getLogger("kaleido").setLevel(logging.WARNING)
        logging.getLogger("choreographer").setLevel(logging.WARNING)

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name=self.experiment_name)

    def _fetch_model(self) -> LGBMClassifier:
        """
        Return a fresh estimator instance for the configured model type.

        Returns:
            LGBMClassifier: An uninitialized classifier corresponding to self.model_type.

        Raises:
            ValueError: If self.model_type is not a supported model name.
        """
        if self.model_type == lightgbm_model_name:
            return LGBMClassifier(
                verbose=-1, importance_type="gain", random_state=self.seed
            )

        raise ValueError(f"Unsupported model type: {self.model_type}")

    def _fetch_param_suggestions(self, trial: optuna.trial.Trial) -> dict:
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
            params = {
                "n_estimators": trial.suggest_int(
                    "n_estimators", 100, 1000
                ),  # number of trees
                "num_leaves": trial.suggest_int(
                    "num_leaves", 16, 256
                ),  # number of leaves in one tree
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.01, 0.2
                ),  # step size for optimisation
                "subsample": trial.suggest_float(
                    "subsample", 0.5, 1.0
                ),  # fraction of samples to be used for each tree
                "colsample_bytree": trial.suggest_float(
                    "colsample_bytree", 0.5, 1.0
                ),  # fraction of features used per tree
                "reg_alpha": trial.suggest_float(
                    "reg_alpha", 1e-4, 0.1, log=True
                ),  # L1 regularisation
                "reg_lambda": trial.suggest_float(
                    "reg_lambda", 1e-4, 0.3, log=True
                ),  # L2 regularisation
                "metric": "binary_logloss",  # evaluation metric
                "random_state": 165,  # seed for reproducibility
                "verbose": -1,  # suppress warnings and info
            }

            # RFE
            if self.use_feature_engineering:
                params["n_features_to_select"] = trial.suggest_float(
                    "n_features_to_select", 0.5, 1.0
                )

            return params

        raise ValueError(f"Unsupported model type: {self.model_type}")

    def _log_model(
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

    def _log_artifact(self, name: str, object: Any) -> None:
        """Serialize an object to a temporary pickle file and log it to MLflow.

        Args:
            name (str): Artifact name stem.
            object (Any): Python object to serialize.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_file_path = os.path.join(tmp_dir, f"{name}.pkl")

            with open(tmp_file_path, "wb") as f:
                pickle.dump(object, f)

            mlflow.log_artifact(
                local_path=tmp_file_path,
                artifact_path=f"pickles/{name}",
            )

    def _log_figure(self, name: str, fig: plt.figure):
        """Log a Matplotlib figure artifact to MLflow.

        Args:
            name (str): Figure name stem.
            fig (plt.figure): Matplotlib figure object.
        """
        mlflow.log_figure(fig, f"plots/{name}.png")

    def _augment_params_with_boosting_rounds(
        self, best_params: dict, final_model: LGBMClassifier
    ) -> dict:
        """Add boosted-rounds reporting fields for MLflow logging.

        Args:
            best_params (dict): Tuned parameter dictionary.
            final_model (LGBMClassifier): Fitted model instance.

        Returns:
            dict: Copy of params including used-rounds metadata when available.
        """
        logged_params = best_params.copy()
        n_estimators_budget = logged_params.get("n_estimators")
        best_iteration = getattr(final_model, "best_iteration_", None)

        if best_iteration is not None and int(best_iteration) > 0:
            logged_params["n_estimators_used"] = int(best_iteration)
            if n_estimators_budget is not None:
                logged_params["stopped_early"] = int(
                    int(best_iteration) < int(n_estimators_budget)
                )

        return logged_params

    def _append_and_log_metrics_and_params(
        self,
        outer_cv_results: OuterCVResults,
        selected_features: np.ndarray,
        outer_fold_log_loss: float,
        best_params: dict,
        run: Run,
    ) -> None:
        """Append fold outputs to results and log metrics/params to MLflow.

        Args:
            outer_cv_results (OuterCVResults): Aggregated outer-CV results container.
            selected_features (np.ndarray): Selected feature names for the fold.
            outer_fold_log_loss (float): Validation log-loss for the fold.
            best_params (dict): Best hyperparameters for the fold.
            run (Run): Current MLflow run.
        """

        outer_cv_results.scores.append(outer_fold_log_loss)
        outer_cv_results.params.append(best_params)
        outer_cv_results.features.append(selected_features)
        outer_cv_results.run_ids.append(run.info.run_id)
        outer_cv_results.experiment_ids.append(run.info.experiment_id)

        mlflow.set_tag("selected_features", ",".join(map(str, selected_features)))
        mlflow.log_metric("log_loss", outer_fold_log_loss)
        mlflow.log_params(best_params)

    def _mlflow_callback(
        self, outer_fold_run_id: str, experiment_name: str
    ) -> Callable[[optuna.Study, optuna.trial.Trial], None]:
        """Build a callback that logs each Optuna trial as a nested MLflow run.

        Args:
            outer_fold_run_id (str): MLflow run id for the current outer fold.
            experiment_name (str): MLflow experiment name.

        Returns:
            Callable[[optuna.Study, optuna.trial.Trial], None]: Trial logging callback.
        """

        def _callback(study: optuna.Study, trial: optuna.trial.Trial):
            """Log one Optuna trial under the parent outer-fold run.

            Args:
                study (optuna.Study): Current Optuna study (unused directly).
                trial (optuna.trial.Trial): Trial to log.
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

    def _eval_validation_loss(
        self,
        X_val_outer: pd.DataFrame,
        y_val_outer: np.ndarray,
        selected_features: np.ndarray,
        final_model: LGBMClassifier,
        category_schema: dict[str, pd.Index],
        column_transformer: Optional[ColumnTransformer],
    ) -> float:
        """Evaluate the fitted model on an outer validation fold.

        Args:
            X_val_outer (pd.DataFrame): Validation features for the outer fold.
            y_val_outer (np.ndarray): Validation labels for the outer fold.
            selected_features (np.ndarray): Selected model input columns.
            final_model (LGBMClassifier): Trained final model.
            category_schema (dict[str, pd.Index]): Training-time categories per categorical column.
            column_transformer (Optional[ColumnTransformer]): Fitted transformer for column-wise features.

        Returns:
            float: Log-loss on the validation fold.
        """
        X_val_outer_pd = self._apply_categorical_dtypes(X_val_outer)

        # 1. Feature Engineering
        if (
            self.use_feature_engineering
            and self.column_wise_transformations is not None
        ):
            X_val_outer_pd = column_transformer.transform(X_val_outer_pd)

        X_val_outer_selected = X_val_outer_pd[selected_features]
        for col, cats in category_schema.items():
            if col in X_val_outer_selected.columns:
                X_val_outer_selected[col] = pd.Categorical(
                    X_val_outer_selected[col], categories=cats
                )

        y_pred_proba = final_model.predict_proba(X_val_outer_selected)

        outer_fold_log_loss = log_loss(y_val_outer, y_pred_proba)

        self._log_model(
            final_model=final_model,
            X_data=X_val_outer_selected,
            output=y_pred_proba,
        )

        if self.log_calibration_curve:
            fig = plot_calibration_curve(y_val_outer, y_pred_proba)
            self._log_figure(name="calibration_curve", fig=fig)

        return outer_fold_log_loss

    def _get_selected_features(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        params: dict,
        X_val: Optional[pd.DataFrame] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
        """Apply optional RFE-based feature selection and align train/validation columns.

        Args:
            X_train (pd.DataFrame): Training features.
            y_train (np.ndarray): Training labels.
            params (dict): Model/selector parameters (may be mutated).
            X_val (Optional[pd.DataFrame], optional): Optional validation features.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame | None, np.ndarray]: Selected
                training dataframe, selected validation dataframe (if provided),
                and selected feature names.
        """
        if not self.use_feature_selection:
            selected_features = X_train.columns.to_list()
            return (X_train, X_val, selected_features)

        n_features_to_select = params.pop("n_features_to_select")
        rfe = self._get_feature_selector(
            fit_params=params, n_features_to_select=n_features_to_select
        )

        X_train_rfe = X_train.copy()
        for col in X_train_rfe.select_dtypes(include=["category"]).columns:
            X_train_rfe[col] = X_train_rfe[col].cat.codes

        rfe.fit(X_train_rfe, y_train)
        selected_features = np.array(X_train.columns[rfe.support_])
        X_train_selected = X_train[selected_features]
        X_val_selected = X_val[selected_features] if X_val is not None else None

        return (X_train_selected, X_val_selected, selected_features)

    def _get_feature_selector(
        self, fit_params: dict, n_features_to_select: float, transform: str = "pandas"
    ) -> RFE:
        """Create an RFE selector configured with the current base estimator.

        Args:
            fit_params (dict): Parameters applied to the base estimator.
            n_features_to_select (float): Number/fraction of features to select.
            transform (str, optional): Output container for selector transform.
                One of ``default``, ``pandas``, ``polars``. Defaults to ``pandas``.

        Raises:
            ValueError: If ``transform`` is not supported.

        Returns:
            RFE: Configured recursive feature eliminator.
        """
        if transform not in ["default", "pandas", "polars"]:
            raise ValueError(f"Invalid transform for RFE: {transform}")

        base_estimator = self._fetch_model()
        base_estimator.set_params(**fit_params)

        rfe = RFE(estimator=base_estimator, n_features_to_select=n_features_to_select)
        rfe.set_output(transform=transform)

        return rfe

    def _objective(
        self,
        trial: optuna.trial.Trial,
        X_train_outer: pd.DataFrame,
        y_train_outer: np.ndarray,
        open_fe_nodes: Optional[List[Node]] = None,
        open_fe_feature_name_mapping: Optional[dict[str, str]] = None,
    ) -> np.ndarray:
        """Compute mean inner-CV validation loss for one Optuna trial.

        Args:
            trial (optuna.trial.Trial): Current Optuna trial.
            X_train_outer (pd.DataFrame): Outer-fold training features.
            y_train_outer (np.ndarray): Outer-fold training labels.
            open_fe_nodes (Optional[List[Node]], optional): Column-wise OpenFE nodes.
            open_fe_feature_name_mapping (Optional[dict[str, str]], optional):
                Mapping from OpenFE formula names to safe output column names.

        Returns:
            np.ndarray: Mean inner-fold binary log-loss for the trial.
        """
        all_params = self._fetch_param_suggestions(trial)
        inner_fold_scores = []

        for inner_train_idx, inner_val_idx in self.inner_cv.split(
            X_train_outer, y_train_outer
        ):
            X_train_inner = X_train_outer.iloc[inner_train_idx]
            X_val_inner = X_train_outer.iloc[inner_val_idx]
            y_train_inner, y_val_inner = (
                y_train_outer[inner_train_idx],
                y_train_outer[inner_val_idx],
            )

            # 1. Feature engineering
            if (
                self.use_feature_engineering
                and self.column_wise_transformations is not None
            ):
                column_transformer = clone(self.column_wise_transformations)
                column_transformer.feature_name_mapping = (
                    open_fe_feature_name_mapping or {}
                )
                column_transformer.fit(X_train_inner, feature_nodes=open_fe_nodes)
                X_train_inner = column_transformer.transform(X_train_inner)
                X_val_inner = column_transformer.transform(X_val_inner)

            model, _, _ = self._fit_model_and_select_features(
                X_train=X_train_inner,
                y_train=y_train_inner,
                params=all_params,
                X_val=X_val_inner,
                y_val=y_val_inner,
                trial=trial,
                use_early_stopping=False,
            )

            inner_fold_scores.append(
                model.evals_result_["valid_0"]["binary_logloss"][-1]
            )

        mean_score = np.mean(inner_fold_scores)

        return mean_score

    def _hyperparameter_tuning(
        self,
        X_train_outer: pd.DataFrame,
        y_train_outer: np.ndarray,
        callback_fn: Callable[[optuna.Study, optuna.trial.Trial], None],
        open_fe_nodes: Optional[List[Node]] = None,
        open_fe_feature_name_mapping: Optional[dict[str, str]] = None,
    ) -> dict:
        """Run Optuna hyperparameter tuning for one outer fold.

        Args:
            X_train_outer (pd.DataFrame): Outer-fold training features.
            y_train_outer (np.ndarray): Outer-fold training labels.
            callback_fn (Callable[[optuna.Study, optuna.trial.Trial], None]): Trial callback.
            open_fe_nodes (Optional[List[Node]], optional): Column-wise OpenFE nodes.
            open_fe_feature_name_mapping (Optional[dict[str, str]], optional):
                Mapping from OpenFE formula names to safe output column names.

        Returns:
            Tuple[optuna.study.Study | None, dict]: Optuna study and best parameters.
                Returns ``(None, {})`` if tuning is disabled.
        """
        self.logger.info("Starting full hyperparameter tuning...")

        if not self.use_hyperparameter_tuning:
            return (None, {})

        sampler = optuna.samplers.TPESampler(seed=self.seed)
        study = optuna.create_study(direction="minimize", sampler=sampler)

        X_train_outer_pd = self._apply_categorical_dtypes(X_train_outer)

        study.optimize(
            lambda trial: self._objective(
                trial,
                X_train_outer_pd,
                y_train_outer,
                open_fe_nodes=open_fe_nodes,
                open_fe_feature_name_mapping=open_fe_feature_name_mapping,
            ),
            n_trials=self.n_trials,
            n_jobs=self.n_jobs,
            show_progress_bar=True,
            callbacks=[callback_fn],
        )

        best_trial = study.best_trial
        best_params = best_trial.params.copy()

        if self.log_parameter_importance:
            fig = plot_param_importances(study=study)
            self._log_figure(name="parameter_importance", fig=fig)

        return best_params

    def _fit_model_and_select_features(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        params: dict,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[np.ndarray] = None,
        trial: Optional[optuna.trial.Trial] = None,
        use_early_stopping: bool = False,
    ) -> Tuple[LGBMClassifier, np.ndarray, dict[str, pd.Index]]:
        """Fit the model once, with optional RFE and optional validation eval set.

        Args:
            X_train (pd.DataFrame): Training features.
            y_train (np.ndarray): Training labels.
            params (dict): Model and feature-selection parameters.
            X_val (Optional[pd.DataFrame], optional): Validation features.
            y_val (Optional[np.ndarray], optional): Validation labels.
            trial (Optional[optuna.trial.Trial], optional): Trial for pruning callback.
            use_early_stopping (bool, optional): Whether to use early stopping on the validation set.

        Returns:
            Tuple[LGBMClassifier, np.ndarray, dict[str, pd.Index]]: Fitted model,
                selected feature names, and category schema captured from training data.
        """
        fit_params = params.copy()

        # 1. Feature selection
        X_train_selected, X_val_selected, selected_features = (
            self._get_selected_features(
                X_train=X_train,
                X_val=X_val,
                y_train=y_train,
                params=fit_params,
            )
        )

        # 2. Categorical Handling
        current_categories = X_train_selected.select_dtypes(
            include=["category"]
        ).columns.tolist()
        category_schema = {
            col: X_train_selected[col].cat.categories for col in current_categories
        }

        # 3. Model Training
        model = self._fetch_model()
        model.set_params(**fit_params)

        callbacks = []
        if trial is not None:
            callbacks.append(LightGBMPruningCallback(trial, "binary_logloss"))
        has_validation = X_val_selected is not None and y_val is not None
        if has_validation and use_early_stopping:
            callbacks.append(early_stopping(stopping_rounds=50, verbose=False))

        eval_set = [(X_val_selected, y_val)] if has_validation else None
        model.fit(
            X_train_selected,
            y_train,
            eval_set=eval_set,
            eval_metric="binary_logloss",
            callbacks=callbacks if callbacks else None,
            categorical_feature=current_categories if current_categories else "auto",
        )

        return model, selected_features, category_schema

    def _fit_final_model(
        self,
        X_train_outer: pd.DataFrame,
        y_train_outer: np.ndarray,
        best_params: dict,
        open_fe_nodes: Optional[List[Node]] = None,
        open_fe_feature_name_mapping: Optional[dict[str, str]] = None,
    ) -> Tuple[LGBMClassifier, np.ndarray, dict[str, pd.Index]]:
        """Fit the final outer-fold model using best hyperparameters.

        Args:
            X_train_outer (pd.DataFrame): Outer-fold training features.
            y_train_outer (np.ndarray): Outer-fold training labels.
            best_params (dict): Best hyperparameters from tuning.
            open_fe_nodes (Optional[List[Node]], optional): Column-wise OpenFE nodes.
            open_fe_feature_name_mapping (Optional[dict[str, str]], optional):
                Mapping from OpenFE formula names to safe output column names.

        Returns:
            Tuple[LGBMClassifier, np.ndarray, dict[str, pd.Index], ColumnTransformer | None]:
                Final model, selected feature names, category schema, and fitted transformer.
        """
        self.logger.info("Fitting final model with best hyperparameters...")

        fit_params = best_params
        X_train_outer_pd = self._apply_categorical_dtypes(X_train_outer)

        # 1. Feature Engineering
        column_transformer = None
        if (
            self.use_feature_engineering
            and self.column_wise_transformations is not None
        ):
            column_transformer = clone(self.column_wise_transformations)
            column_transformer.feature_name_mapping = open_fe_feature_name_mapping or {}
            column_transformer.fit(X_train_outer_pd, feature_nodes=open_fe_nodes)
            X_train_outer_pd = column_transformer.transform(X_train_outer_pd)

        final_model, selected_features, category_schema = (
            self._fit_model_and_select_features(
                X_train=X_train_outer_pd,
                y_train=y_train_outer,
                params=fit_params,
                use_early_stopping=True,
            )
        )

        if self.log_feature_importance:
            fig = plot_feature_importance(
                X_train=X_train_outer_pd[selected_features], model=final_model
            )
            self._log_figure(name="feature_importance", fig=fig)

        return final_model, selected_features, category_schema, column_transformer

    def get_generalisation_error(
        self, X_train: pl.DataFrame, y_train: pl.DataFrame
    ) -> OuterCVResults:
        """Run nested CV and return aggregated outer-fold evaluation results.

        Args:
            X_train (pl.DataFrame): Training features.
            y_train (pl.DataFrame): Training labels.

        Returns:
            OuterCVResults: Aggregated per-fold metrics, params, selected features,
                run ids, and parent run id.
        """
        outer_cv_results = OuterCVResults()
        self._setup_mlflow()
        X_train_pd = X_train.to_pandas()
        y_train_np = y_train.to_numpy().ravel()

        # Apply rowwise transformations to the entire dataset
        if self.use_feature_engineering and self.row_wise_transformations is not None:
            X_train_pd = self.row_wise_transformations.apply_row_wise_transformations(
                X_train_pd
            )

        with mlflow.start_run(
            run_name=f"{self.run_name}_{self.model_type}"
        ) as parent_run:
            outer_cv_results.parent_run_id = parent_run.info.run_id
            for i, (train_idx, val_idx) in enumerate(
                self.outer_cv.split(X_train_pd, y_train_np)
            ):
                X_train_outer = X_train_pd.iloc[train_idx]
                X_val_outer = X_train_pd.iloc[val_idx]
                y_train_outer = y_train_np[train_idx]
                y_val_outer = y_train_np[val_idx]
                column_wise_features: List[Node] = []
                formula_to_safe_name: dict[str, str] = {}

                with mlflow.start_run(
                    nested=True, run_name=f"Outer_fold_{i + 1}"
                ) as run:
                    parent_id = run.info.run_id
                    callback_fn = (
                        self._mlflow_callback(
                            outer_fold_run_id=parent_id,
                            experiment_name=self.experiment_name,
                        )
                        if self.log_hyperparameter_trials
                        else lambda study, trial: None
                    )

                    if self.use_ofe and self.open_fe_transformations is not None:
                        self.logger.info(f"Fitting OpenFE on fold {i + 1}")
                        row_wise_features, column_wise_features = (
                            self.open_fe_transformations.fit(
                                X_train=X_train_outer,
                                y_train=y_train_outer,
                                task="classification",
                                categorical_features=self.categorical_columns,
                                feature_boosting=True,
                                n_jobs=self.n_jobs,
                            )
                        )

                        X_train_outer, X_val_outer, mapping = (
                            self.open_fe_transformations.apply_openfe_features(
                                X_train=X_train_outer,
                                X_val=X_val_outer,
                                features=row_wise_features,
                                n_jobs=self.n_jobs,
                            )
                        )

                        column_wise_mapping = {
                            f"ofe_col_{idx + 1}": tree_to_formula(feature)
                            for idx, feature in enumerate(column_wise_features)
                        }
                        formula_to_safe_name = {
                            formula: safe_name
                            for safe_name, formula in column_wise_mapping.items()
                        }
                        full_mapping = {**mapping, **column_wise_mapping}

                        self._log_artifact(
                            f"ofe_row_wise_features_fold_{i}", row_wise_features
                        )
                        self._log_artifact(
                            f"ofe_column_wise_features_fold_{i}",
                            column_wise_features,
                        )
                        self._log_artifact(
                            f"ofe_feature_mapping_fold_{i}", full_mapping
                        )

                        for feat_name, formula in full_mapping.items():
                            self.logger.info(
                                f"Feature: {feat_name:20} | Formula: {formula}"
                            )

                    # 1. Perform full hyperparamter tuning
                    best_params = self._hyperparameter_tuning(
                        X_train_outer=X_train_outer,
                        y_train_outer=y_train_outer,
                        callback_fn=callback_fn,
                        open_fe_nodes=column_wise_features,
                        open_fe_feature_name_mapping=formula_to_safe_name,
                    )

                    # 2. Fit final model with best hyperparameters on the entire outer training fold
                    (
                        final_model,
                        selected_features,
                        category_schema,
                        column_transformer,
                    ) = self._fit_final_model(
                        X_train_outer=X_train_outer,
                        y_train_outer=y_train_outer,
                        best_params=best_params,
                        open_fe_nodes=column_wise_features,
                        open_fe_feature_name_mapping=formula_to_safe_name,
                    )

                    # 3. Evaluate final model on the outer validaiton fold and log final model to model registry
                    outer_fold_log_loss = self._eval_validation_loss(
                        X_val_outer=X_val_outer,
                        y_val_outer=y_val_outer,
                        selected_features=selected_features,
                        final_model=final_model,
                        category_schema=category_schema,
                        column_transformer=column_transformer,
                    )

                    logged_params = self._augment_params_with_boosting_rounds(
                        best_params=best_params, final_model=final_model
                    )

                    # 4. Log metrics and parameters to MLFlow
                    self._append_and_log_metrics_and_params(
                        outer_cv_results=outer_cv_results,
                        selected_features=selected_features,
                        outer_fold_log_loss=outer_fold_log_loss,
                        best_params=logged_params,
                        run=run,
                    )

            mean_score = np.mean(outer_cv_results.scores)
            std_score = np.std(outer_cv_results.scores)
            mlflow.log_metric("mean_log_loss", mean_score)
            mlflow.log_metric("std_log_loss", std_score)

        mlflow.end_run()

        return outer_cv_results


if __name__ == "__main__":
    pass
