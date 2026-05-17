import polars as pl
from sklearn.model_selection import train_test_split
import optuna
from utils.statics import tracking_uri
import mlflow
import numpy as np
import logging
from model.data_classes import OuterCVResults
from training.nested_cv_eval import ModelCVEvaluator
from feature_engineering.OpenFE.FeatureGenerator import Node
from utils.utils import plot_feature_importance, plot_loss_curve
from sklearn import clone
from typing import List, Optional, Tuple, Union
import pandas as pd
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier


class ModelParamTuner(ModelCVEvaluator):
    def _setup_mlflow(self):
        """
        Sets up tracking uri and experiment for MLFlow

        Returns:
            None
        """
        self.logger.info(
            f"""Starting training with model {self.model_type} with the following configuration:
        - {self.n_inner_splits} inner splits
        - {self.n_trials} trials
        - max {self.n_jobs} concurrent jobs"""
        )
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        logging.getLogger("mlflow").setLevel(logging.ERROR)
        logging.getLogger("optuna").setLevel(logging.WARNING)

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name=self.experiment_name)

    def _extract_loss_history(
        self, model: Union[LGBMClassifier, XGBClassifier, CatBoostClassifier]
    ) -> Tuple[list[float], list[float]]:
        """
        Extracts training and validation loss histories from evals_result_.
        Returns (train_loss_list, valid_loss_list).
        """
        model_classname = type(model).__name__
        res = getattr(model, "evals_result_", {})

        train_loss = []
        valid_loss = []

        # 1. LightGBM Scikit-Learn API
        if model_classname == "LGBMClassifier":
            # Note: LightGBM only tracks training loss if passed into eval_set
            if "training" in res:
                train_loss = res["training"].get("binary_logloss", [])
            elif "train" in res:
                train_loss = res["train"].get("binary_logloss", [])

            # Validation set defaults to 'valid_0'
            if "valid_0" in res:
                valid_loss = res["valid_0"].get("binary_logloss", [])

        # 2. XGBoost Scikit-Learn API
        elif model_classname == "XGBClassifier":
            # If both train and val sets were passed, validation_0 is usually train, validation_1 is val.
            # If only val was passed, validation_0 is val.
            if "validation_1" in res:
                train_loss = res["validation_0"].get("logloss", [])
                valid_loss = res["validation_1"].get("logloss", [])
            elif "validation_0" in res:
                valid_loss = res["validation_0"].get("logloss", [])

        # 3. CatBoost Scikit-Learn API
        elif model_classname == "CatBoostClassifier":
            # CatBoost natively separates training ('learn') and evaluation ('validation')
            if "learn" in res:
                train_loss = res["learn"].get("Logloss", [])
            if "validation" in res:
                valid_loss = res["validation"].get("Logloss", [])

        return train_loss, valid_loss

    def _fit_model_and_select_features_with_loss_curve(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        params: dict,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[np.ndarray] = None,
        trial: Optional[optuna.trial.Trial] = None,
        should_plot_loss_curve: bool = False,
    ) -> Tuple:
        """Fit model with optional feature selection, pruning, and loss-curve logging.

        Args:
            X_train (pd.DataFrame): Training features.
            y_train (np.ndarray): 1D training labels.
            params (dict): Model and optional selector parameters.
            X_val (Optional[pd.DataFrame], optional): Validation features.
            y_val (Optional[np.ndarray], optional): 1D validation labels.
            trial (Optional[optuna.trial.Trial], optional): Optuna trial for pruning.
            should_plot_loss_curve (bool, optional): If True and validation data is
                provided, logs train/valid binary logloss curves to MLflow.

        Returns:
            Tuple: Fitted model, selected feature names, and categorical schema from
                selected training columns.
        """
        fit_params = params.copy()

        X_train_selected, X_val_selected, selected_features = (
            self._get_selected_features(
                X_train=X_train,
                X_val=X_val,
                y_train=y_train,
                params=fit_params,
            )
        )

        category_schema = self._extract_category_schema(X_train_selected)

        has_validation = X_val_selected is not None and y_val is not None
        record_curve = should_plot_loss_curve and has_validation

        model = self.wrapper.fit(
            X_train=X_train_selected,
            y_train=y_train,
            X_val=X_val_selected,
            y_val=y_val,
            use_early_stopping=has_validation,
            params=fit_params,
            trial=trial,
        )

        # Extract loss curves if available and log to MLflow
        if record_curve and hasattr(model, "evals_result_"):
            train_loss, valid_loss = self._extract_loss_history(model)
            if train_loss and valid_loss:
                fig = plot_loss_curve(train_loss, valid_loss, self.model_type)
                self.mlflow_handler.log_figure(
                    fig=fig, name=f"{self.model_type}_loss_curve"
                )

        return model, selected_features, category_schema

    def _fit_final_model_with_loss_curve(
        self,
        X_train_outer: pd.DataFrame,
        y_train_outer: np.ndarray,
        best_params: dict,
        open_fe_nodes: Optional[List[Node]] = None,
        open_fe_feature_name_mapping: Optional[dict[str, str]] = None,
        X_val_outer: Optional[pd.DataFrame] = None,
        y_val_outer: Optional[np.ndarray] = None,
        plot_loss_curve: bool = False,
    ) -> Tuple:
        """Fit final model, optionally plotting train/valid loss curve when validation data is provided."""
        self.logger.info("Fitting final model with best hyperparameters...")

        X_train_outer_pd = self._apply_categorical_dtypes(X_train_outer)
        X_val_outer_pd = (
            self._apply_categorical_dtypes(X_val_outer)
            if X_val_outer is not None
            else None
        )

        column_transformer = None
        if (
            self.use_feature_engineering
            and self.column_wise_transformations is not None
        ):
            column_transformer = clone(self.column_wise_transformations)
            column_transformer.feature_name_mapping = open_fe_feature_name_mapping or {}
            column_transformer.fit(X_train_outer_pd, feature_nodes=open_fe_nodes)
            X_train_outer_pd = column_transformer.transform(X_train_outer_pd)
            if X_val_outer_pd is not None:
                X_val_outer_pd = column_transformer.transform(X_val_outer_pd)

        final_model, selected_features, category_schema = (
            self._fit_model_and_select_features_with_loss_curve(
                X_train=X_train_outer_pd,
                y_train=y_train_outer,
                params=best_params,
                X_val=X_val_outer_pd,
                y_val=y_val_outer,
                should_plot_loss_curve=plot_loss_curve,
            )
        )

        if self.log_feature_importance:
            fig = plot_feature_importance(
                X_train=X_train_outer_pd[selected_features], model=final_model
            )
            self.mlflow_handler.log_figure(fig=fig, name="feature_importance")

        return final_model, selected_features, category_schema, column_transformer

    def tune_and_train(self, X: pl.DataFrame, y: pl.DataFrame) -> OuterCVResults:
        """Tune and train on the entire X_train set. Use one-layer CV for hyperparameter optimisation.

        Args:
            X_train (pl.DataFrame): Input features used for training
            y_train (pl.DataFrame): Ground truths used for training

        Returns:
            OuterCVResults: Outer cross-validation results
        """
        outer_cv_results = OuterCVResults()
        self._setup_mlflow()

        X_pd = X.to_pandas().copy()
        y_np = y.to_numpy().ravel()

        # Apply rowwise transformations to the entire dataset
        if self.use_feature_engineering and self.row_wise_transformations is not None:
            X_pd = self.row_wise_transformations.apply_row_wise_transformations(X_pd)

        X_train, X_val, y_train, y_val = train_test_split(
            X_pd, y_np, test_size=0.2, stratify=y_np, random_state=42
        )
        column_wise_features: List[Node] = []
        formula_to_safe_name: dict[str, str] = {}

        with mlflow.start_run(run_name=f"{self.run_name}_{self.model_type}") as run:
            parent_id = run.info.run_id
            outer_cv_results.parent_run_id = parent_id
            callback_fn = (
                self._mlflow_callback(
                    outer_fold_run_id=parent_id,
                    experiment_name=self.experiment_name,
                )
                if self.log_hyperparameter_trials
                else lambda study, trial: None
            )
            if self.use_ofe and self.open_fe_transformations is not None:
                X_train, X_val, column_wise_features, formula_to_safe_name = (
                    self._ofe_transform(
                        X_train_outer=X_train,
                        y_train_outer=y_train,
                        X_val_outer=X_val,
                        i=0,
                    )
                )

            # 1. Perform full hyperparamter tuning
            best_params = self._hyperparameter_tuning(
                X_train_outer=X_train,
                y_train_outer=y_train,
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
            ) = self._fit_final_model_with_loss_curve(
                X_train_outer=X_train,
                y_train_outer=y_train,
                best_params=best_params,
                open_fe_nodes=column_wise_features,
                open_fe_feature_name_mapping=formula_to_safe_name,
                X_val_outer=X_val,
                y_val_outer=y_val,
                plot_loss_curve=True,
            )

            # 3. Evaluate final model on the outer validation fold and log final model to model registry
            outer_fold_log_loss, y_pred_proba = self._eval_validation_loss(
                model=final_model,
                X_val_outer=X_val,
                y_val_outer=y_val,
                selected_features=selected_features,
                category_schema=category_schema,
                column_transformer=column_transformer,
            )

            logged_params = self._augment_params_with_boosting_rounds(
                best_params=best_params, model=final_model
            )

            self._append_and_log_metrics_and_params(
                outer_cv_results=outer_cv_results,
                selected_features=selected_features,
                outer_fold_log_loss=outer_fold_log_loss,
                best_params=logged_params,
                run=run,
            )

        mlflow.end_run()

        return outer_cv_results
