import logging
import os
import random
from typing import Callable, List, Optional, Tuple, Union

import mlflow
import numpy as np
import optuna
import pandas as pd
import polars as pl
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from mlflow.entities import Run
from mlflow.tracking import MlflowClient
from mlflow.utils.mlflow_tags import MLFLOW_PARENT_RUN_ID
from optuna.visualization import plot_param_importances
from sklearn import clone
from sklearn.feature_selection import RFE
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

import utils.statics as statics
from feature_engineering.ColumnTransformer import ColumnTransformer
from feature_engineering.OpenFE.FeatureGenerator import Node
from feature_engineering.OpenFE.utils import tree_to_formula
from feature_engineering.OpenFETransformations import OpenFETransformations
from feature_engineering.RowWiseTransformations import RowWiseTransformations
from model.CatBoostWrapper import CatBoostWrapper
from model.data_classes import OuterCVResults
from model.LGBMWrapper import LGBMWrapper
from model.XGBoostWrapper import XGBoostWrapper
from utils.mlflow_handler import MLflowHandler
from utils.utils import plot_calibration_curve, plot_feature_importance


class ModelCVEvaluator:
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

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
        self.categorical_columns = categorical_columns if categorical_columns is not None else []
        self.ohe_columns = ohe_columns if ohe_columns is not None else []
        self.seed = 165
        self.n_jobs = max(1, os.cpu_count() - 4) or 1
        self.inner_cv = StratifiedKFold(n_splits=self.n_inner_splits, shuffle=True, random_state=self.seed)
        self.outer_cv = StratifiedKFold(n_splits=self.n_outer_splits, shuffle=True, random_state=self.seed)
        self._set_global_seed()
        self.wrapper = self._fetch_model_wrapper(model_name=self.model_type)
        self.mlflow_handler = MLflowHandler(
            tracking_uri=statics.tracking_uri,
            experiment_name=experiment_name,
            logger=self.logger,
        )

    def _set_global_seed(self):
        """Set global random seeds used by NumPy and Python's random module."""
        np.random.seed(self.seed)
        random.seed(self.seed)

    def _fetch_model_wrapper(self, model_name: str) -> Union[LGBMWrapper, XGBoostWrapper, CatBoostWrapper]:
        """Factory method to return a model wrapper instance based on the model name.

        Args:
            model_name (str): Name of the model type (e.g., "lightgbm", "xgboost", "catboost").

        Returns:
            BaseModelWrapper: An instance of a class that inherits from BaseModelWrapper.

        Raises:
            ValueError: If the provided model_name is not supported.
        """
        match model_name:
            case statics.lightgbm_model_name:
                return LGBMWrapper(seed=self.seed)
            case statics.xgboost_model_name:
                return XGBoostWrapper(seed=self.seed)
            case statics.catboost_model_name:
                return CatBoostWrapper(seed=self.seed)

        raise ValueError(f"Unsupported model type: {model_name}")

    def _extract_category_schema(self, df: pd.DataFrame) -> dict[str, pd.Index]:
        """Extract and record category levels from categorical columns.

        Args:
            df (pd.DataFrame): Input dataframe to extract categories from.

        Returns:
            dict[str, pd.Index]: Mapping of categorical column names to their category levels.
        """
        cat_cols = df.select_dtypes(include=["category"]).columns
        return {col: df[col].cat.categories for col in cat_cols}

    def _apply_category_schema(self, df: pd.DataFrame, schema: dict[str, pd.Index]) -> pd.DataFrame:
        """Apply a predefined categorical schema to a dataframe.

        Args:
            df (pd.DataFrame): Input dataframe to apply schema to.
            schema (dict[str, pd.Index]): Mapping of column names to their category levels.

        Returns:
            pd.DataFrame: Copy of dataframe with categorical schema applied.
        """
        df_copy = df.copy()
        for col, categories in schema.items():
            if col in df_copy.columns:
                df_copy[col] = pd.Categorical(df_copy[col], categories=categories)
        return df_copy

    def _encode_categories_to_codes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert categorical columns to their underlying integer codes.

        Args:
            df (pd.DataFrame): Input dataframe with categorical columns.

        Returns:
            pd.DataFrame: Copy of dataframe with categorical columns encoded as integers.
        """
        df_copy = df.copy()
        cat_cols = df_copy.select_dtypes(include=["category"]).columns
        for col in cat_cols:
            df_copy[col] = df_copy[col].cat.codes
        return df_copy

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
        - {self.n_trials} trials
        - max {self.n_jobs} concurrent jobs"""
        )
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        logging.getLogger("optuna").setLevel(logging.WARNING)
        self.mlflow_handler.setup()

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
        params = self.wrapper.get_optuna_params(trial)

        # RFE
        if self.use_feature_selection:
            params["n_features_to_select"] = trial.suggest_float("n_features_to_select", 0.5, 1.0)

        return params

    def _build_subrun_tags(self, selected_features: Optional[np.ndarray] = None) -> dict[str, str | bool]:
        """Build standard tags for MLflow sub-runs."""
        categorical_handling = "OHE" if len(self.ohe_columns) > 0 else "native"
        tags: dict[str, str | bool] = {
            "model_type": self.model_type,
            "feature_engineering": bool(self.use_feature_engineering),
            "categorical_handling": categorical_handling,
            "openfe": bool(self.use_ofe),
            "hyperparameter_tuning": bool(self.use_hyperparameter_tuning),
        }
        if selected_features is not None:
            tags["selected_features"] = ",".join(map(str, selected_features))
        return tags

    def get_best_iteration(
        self,
        model: Union[LGBMClassifier, XGBClassifier, CatBoostClassifier],
    ) -> Optional[int]:
        """
        Safely extracts the best early-stopping iteration from LightGBM,
        XGBoost, or CatBoost Scikit-Learn API classifier instances.

        Args:
            model (Union[LGBMClassifier, XGBClassifier, CatBoostClassifier]): The fitted model instance.

        Returns:
            Optional[int]: The best iteration index (0-indexed),
                        or None if early stopping wasn't triggered/supported.
        """
        model_classname = type(model).__name__

        if model_classname == "LGBMClassifier":
            return getattr(model, "best_iteration_", None)

        elif model_classname == "XGBClassifier":
            return getattr(model, "best_iteration", None)

        elif model_classname == "CatBoostClassifier":
            if hasattr(model, "get_best_iteration"):
                return model.get_best_iteration()

        return None

    def _augment_params_with_boosting_rounds(
        self,
        model: Union[LGBMClassifier, XGBClassifier, CatBoostClassifier],
        best_params: dict,
    ) -> dict:
        """Add boosted-rounds reporting fields for MLflow logging.

        Args:
            model (Union[LGBMClassifier, XGBClassifier, CatBoostClassifier]): The fitted model.
            best_params (dict): Tuned parameter dictionary.

        Returns:
            dict: Copy of params including used-rounds metadata when available.
        """
        logged_params = best_params.copy()
        best_iteration = self.get_best_iteration(model)

        if best_iteration:
            logged_params["n_estimators_used"] = int(best_iteration)
        return logged_params

    def _init_oof_predictions(self, n_samples: int, n_classes: int = 2) -> np.ndarray:
        """Initialize an array to store out-of-fold predictions for all samples.

        Args:
            n_samples (int): Total number of samples in the dataset.
            n_classes (int, optional): Number of classes for classification. Defaults to 2.

        Returns:
            np.ndarray: An array initialized to store out-of-fold predictions.
        """
        if n_classes == 2:
            return np.zeros(n_samples)  # Store only positive class probability for binary classification
        else:
            return np.zeros((n_samples, n_classes))  # Store probabilities for all classes for multiclass classification

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

        mlflow.set_tags(self._build_subrun_tags(selected_features=selected_features))
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
            tags.update(self._build_subrun_tags())

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
        model: Union[LGBMClassifier, XGBClassifier, CatBoostClassifier],
        X_val_outer: pd.DataFrame,
        y_val_outer: np.ndarray,
        selected_features: np.ndarray,
        category_schema: dict[str, pd.Index],
        column_transformer: Optional[ColumnTransformer],
    ) -> Tuple[float, np.ndarray]:
        """Evaluate the fitted model on an outer validation fold.

        Args:
            X_val_outer (pd.DataFrame): Validation features for the outer fold.
            y_val_outer (np.ndarray): Validation labels for the outer fold.
            selected_features (np.ndarray): Selected model input columns.
            category_schema (dict[str, pd.Index]): Training-time categories per categorical column.
            column_transformer (Optional[ColumnTransformer]): Fitted transformer for column-wise features.

        Returns:
            float: Log-loss on the validation fold.
        """
        # 1. Feature Engineering
        if self.use_feature_engineering and self.column_wise_transformations is not None:
            X_val_outer = column_transformer.transform(X_val_outer)

        # 2. Choose selected features
        X_val_outer_selected = X_val_outer[selected_features]

        # 3. Enforce training-time category schema to ensure correct integer encoding
        X_val_outer_selected = self._apply_category_schema(X_val_outer_selected, category_schema)

        # 4. Predict and evaluate
        y_pred_proba = model.predict_proba(X_val_outer_selected)

        outer_fold_log_loss = log_loss(y_val_outer, y_pred_proba)

        self.mlflow_handler.log_model(
            model=model,
            X_data=X_val_outer_selected,
            y_pred=y_pred_proba,
            model_type=self.model_type,
            name=self.model_type,
        )

        if self.log_calibration_curve:
            fig = plot_calibration_curve(y_val_outer, y_pred_proba)
            self.mlflow_handler.log_figure(fig=fig, name="calibration_curve")

        return outer_fold_log_loss, y_pred_proba

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
        rfe = self._get_feature_selector(fit_params=params, n_features_to_select=n_features_to_select)

        # Build category schema from training data
        category_schema_rfe = self._extract_category_schema(X_train)

        # Enforce training-time categories before RFE to ensure consistency
        X_train_rfe = self._apply_category_schema(X_train, category_schema_rfe)

        # Convert to codes for RFE
        X_train_rfe = self._encode_categories_to_codes(X_train_rfe)

        rfe.fit(X_train_rfe, y_train)
        selected_features = np.array(X_train.columns[rfe.support_])
        X_train_selected = X_train[selected_features]
        X_val_selected = X_val[selected_features] if X_val is not None else None

        return (X_train_selected, X_val_selected, selected_features)

    def _get_feature_selector(self, fit_params: dict, n_features_to_select: float, transform: str = "pandas") -> RFE:
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

        base_estimator = self.wrapper.fetch_base_estimator(params=fit_params)

        rfe = RFE(estimator=base_estimator, n_features_to_select=n_features_to_select)
        rfe.set_output(transform=transform)

        return rfe

    def _extract_validation_loss(self, model: Union[LGBMClassifier, XGBClassifier, CatBoostClassifier]) -> float:
        """
        Extracts the final inner validation fold loss value from
        the model's evals_result_ dictionary, tailored to each framework.
        """
        model_classname = type(model).__name__
        res = getattr(model, "evals_result_", {})

        # 1. LightGBM
        if model_classname == "LGBMClassifier":
            return float(res["valid_0"]["binary_logloss"][-1])

        # 2. XGBoost
        elif model_classname == "XGBClassifier":
            # XGBoost uses 'logloss' for binary classification
            return float(res["validation_0"]["logloss"][-1])

        # 3. CatBoost
        elif model_classname == "CatBoostClassifier":
            # CatBoost uses 'validation' and capitalizes 'Logloss'
            return float(res["validation"]["Logloss"][-1])

        raise ValueError(f"Unknown framework model: {model_classname}")

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

        for inner_train_idx, inner_val_idx in self.inner_cv.split(X_train_outer, y_train_outer):
            X_train_inner = X_train_outer.iloc[inner_train_idx]
            X_val_inner = X_train_outer.iloc[inner_val_idx]
            y_train_inner, y_val_inner = (
                y_train_outer[inner_train_idx],
                y_train_outer[inner_val_idx],
            )

            # 1. Feature engineering
            if self.use_feature_engineering and self.column_wise_transformations is not None:
                column_transformer = clone(self.column_wise_transformations)
                column_transformer.feature_name_mapping = open_fe_feature_name_mapping or {}
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

            inner_fold_scores.append(self._extract_validation_loss(model=model))

        mean_score = np.mean(inner_fold_scores)

        return mean_score

    def _ofe_transform(
        self,
        X_train_outer: pd.DataFrame,
        y_train_outer: np.ndarray,
        X_val_outer: pd.DataFrame,
        i: int,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, List[Node], dict[str, str]]:
        """Apply OpenFE transformations to the outer fold data and log the generated features and mappings.

        Args:
            X_train_outer (pd.DataFrame): Outer-fold training features.
            y_train_outer (np.ndarray): Outer-fold training labels.
            X_val_outer (pd.DataFrame): Outer-fold validation features.
            i (int): Current outer fold index for logging purposes.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, List[Node], dict[str, str]]: _description_
        """
        self.logger.info(f"Fitting OpenFE on fold {i + 1}")
        row_wise_features, column_wise_features = self.open_fe_transformations.fit(
            X_train=X_train_outer,
            y_train=y_train_outer,
            task="classification",
            categorical_features=self.categorical_columns,
            feature_boosting=True,
            n_jobs=self.n_jobs,
        )

        X_train_outer, X_val_outer, mapping = self.open_fe_transformations.apply_openfe_features(
            X_train=X_train_outer,
            X_val=X_val_outer,
            features=row_wise_features,
            n_jobs=self.n_jobs,
        )

        column_wise_mapping = {
            f"ofe_col_{idx + 1}": tree_to_formula(feature) for idx, feature in enumerate(column_wise_features)
        }
        formula_to_safe_name = {formula: safe_name for safe_name, formula in column_wise_mapping.items()}
        full_mapping = {**mapping, **column_wise_mapping}

        self.mlflow_handler.log_artifact_pickle(row_wise_features, f"ofe_row_wise_features_fold_{i}")
        self.mlflow_handler.log_artifact_pickle(
            column_wise_features,
            f"ofe_column_wise_features_fold_{i}",
        )
        self.mlflow_handler.log_artifact_pickle(full_mapping, f"ofe_feature_mapping_fold_{i}")

        for feat_name, formula in full_mapping.items():
            self.logger.info(f"Feature: {feat_name:20} | Formula: {formula}")

        return X_train_outer, X_val_outer, column_wise_features, formula_to_safe_name

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
            return {}

        sampler = optuna.samplers.TPESampler(seed=self.seed)
        study = optuna.create_study(direction="minimize", sampler=sampler)

        study.optimize(
            lambda trial: self._objective(
                trial,
                X_train_outer,
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
            self.mlflow_handler.log_figure(fig=fig, name="parameter_importance")

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
    ) -> Tuple[
        Union[LGBMClassifier, XGBClassifier, CatBoostClassifier],
        np.ndarray,
        dict[str, pd.Index],
    ]:
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
            Tuple[Union[LGBMClassifier, XGBClassifier, CatBoostClassifier], np.ndarray, dict[str, pd.Index]]: Fitted model,
                selected feature names, and category schema captured from training data.
        """
        local_wrapper = self._fetch_model_wrapper(model_name=self.model_type)
        fit_params = params

        # 1. Feature selection
        X_train_selected, X_val_selected, selected_features = self._get_selected_features(
            X_train=X_train,
            X_val=X_val,
            y_train=y_train,
            params=fit_params,
        )

        # 2. Categorical Handling
        category_schema = self._extract_category_schema(X_train_selected)

        # 3. Model Training
        model = local_wrapper.fit(
            X_train=X_train_selected,
            y_train=y_train,
            X_val=X_val_selected,
            y_val=y_val,
            use_early_stopping=use_early_stopping,
            params=fit_params,
            trial=trial,
        )

        return model, selected_features, category_schema

    def _fit_final_model(
        self,
        X_train_outer: pd.DataFrame,
        y_train_outer: np.ndarray,
        best_params: dict,
        open_fe_nodes: Optional[List[Node]] = None,
        open_fe_feature_name_mapping: Optional[dict[str, str]] = None,
    ) -> Tuple[
        Union[LGBMClassifier, XGBClassifier, CatBoostClassifier],
        np.ndarray,
        dict[str, pd.Index],
    ]:
        """Fit the final outer-fold model using best hyperparameters.

        Args:
            X_train_outer (pd.DataFrame): Outer-fold training features.
            y_train_outer (np.ndarray): Outer-fold training labels.
            best_params (dict): Best hyperparameters from tuning.
            open_fe_nodes (Optional[List[Node]], optional): Column-wise OpenFE nodes.
            open_fe_feature_name_mapping (Optional[dict[str, str]], optional):
                Mapping from OpenFE formula names to safe output column names.

        Returns:
            Tuple[Union[LGBMClassifier, XGBClassifier, CatBoostClassifier], np.ndarray, dict[str, pd.Index], ColumnTransformer | None]:
                Final model classifier, selected feature names, category schema, and fitted transformer.
        """
        self.logger.info("Fitting final model with best hyperparameters...")

        fit_params = best_params

        # 1. Feature Engineering
        column_transformer = None
        if self.use_feature_engineering and self.column_wise_transformations is not None:
            column_transformer = clone(self.column_wise_transformations)
            column_transformer.feature_name_mapping = open_fe_feature_name_mapping or {}
            column_transformer.fit(X_train_outer, feature_nodes=open_fe_nodes)
            X_train_outer = column_transformer.transform(X_train_outer)

        final_model, selected_features, category_schema = self._fit_model_and_select_features(
            X_train=X_train_outer,
            y_train=y_train_outer,
            params=fit_params,
            use_early_stopping=True,
        )
        if self.log_feature_importance:
            fig = plot_feature_importance(X_train=X_train_outer[selected_features], model=final_model)
            self.mlflow_handler.log_figure(fig=fig, name="feature_importance")

        return final_model, selected_features, category_schema, column_transformer

    def get_generalisation_error(self, X_train: pl.DataFrame, y_train: pl.DataFrame) -> OuterCVResults:
        """Run nested CV and return aggregated outer-fold evaluation results.

        Args:
            X_train (pl.DataFrame): Training features.
            y_train (pl.DataFrame): Training labels.

        Returns:
            OuterCVResults: Aggregated per-fold metrics, params, selected features,
                run ids, and parent run id.
        """
        self._setup_mlflow()
        X_train_pd = X_train.to_pandas()
        y_train_np = y_train.to_numpy().ravel()
        n_classes = len(np.unique(y_train_np))
        oof_predictions = self._init_oof_predictions(n_samples=len(X_train_pd), n_classes=n_classes)

        outer_cv_results = OuterCVResults()

        # Cast to categorical dtype
        X_train_pd = self._apply_categorical_dtypes(X_train_pd)

        # Apply rowwise transformations to the entire dataset
        if self.use_feature_engineering and self.row_wise_transformations is not None:
            X_train_pd = self.row_wise_transformations.apply_row_wise_transformations(X_train_pd)

        with mlflow.start_run(run_name=f"{self.run_name}_{self.model_type}") as parent_run:
            outer_cv_results.parent_run_id = parent_run.info.run_id
            for i, (train_idx, val_idx) in enumerate(self.outer_cv.split(X_train_pd, y_train_np)):
                X_train_outer = X_train_pd.iloc[train_idx]
                X_val_outer = X_train_pd.iloc[val_idx]
                y_train_outer = y_train_np[train_idx]
                y_val_outer = y_train_np[val_idx]
                column_wise_features: List[Node] = []
                formula_to_safe_name: dict[str, str] = {}

                with mlflow.start_run(nested=True, run_name=f"Outer_fold_{i + 1}") as run:
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
                        (
                            X_train_outer,
                            X_val_outer,
                            column_wise_features,
                            formula_to_safe_name,
                        ) = self._ofe_transform(
                            X_train_outer=X_train_outer,
                            y_train_outer=y_train_outer,
                            X_val_outer=X_val_outer,
                            i=i,
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

                    # 3. Evaluate final model on the outer validation fold and log final model to model registry
                    outer_fold_log_loss, y_pred_proba = self._eval_validation_loss(
                        model=final_model,
                        X_val_outer=X_val_outer,
                        y_val_outer=y_val_outer,
                        selected_features=selected_features,
                        category_schema=category_schema,
                        column_transformer=column_transformer,
                    )

                    logged_params = self._augment_params_with_boosting_rounds(
                        best_params=best_params, model=final_model
                    )

                    # 4. Log metrics and parameters to MLFlow
                    self._append_and_log_metrics_and_params(
                        outer_cv_results=outer_cv_results,
                        selected_features=selected_features,
                        outer_fold_log_loss=outer_fold_log_loss,
                        best_params=logged_params,
                        run=run,
                    )

                    # 5. Store OOF predictions for the fold
                    if n_classes == 2:
                        oof_predictions[val_idx] = y_pred_proba[:, 1]
                    else:
                        oof_predictions[val_idx, :] = y_pred_proba

            mean_score = np.mean(outer_cv_results.scores)
            std_score = np.std(outer_cv_results.scores)
            mlflow.log_metric("mean_log_loss", mean_score)
            mlflow.log_metric("std_log_loss", std_score)
            self.mlflow_handler.log_oof_predictions(oof_preds=oof_predictions)

        mlflow.end_run()

        return outer_cv_results


if __name__ == "__main__":
    pass
