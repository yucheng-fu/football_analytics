import polars as pl
from lightgbm import LGBMClassifier
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import optuna
from utils.statics import tracking_uri
import mlflow
import numpy as np
import logging
from model.data_classes import OuterCVResults
from model.nested_cv_eval import ModelCVEvaluator
from feature_engineering.OpenFE.utils import tree_to_formula
from feature_engineering.OpenFE.FeatureGenerator import Node
from utils.utils import plot_feature_importance, plot_loss_curve
from sklearn import clone
from typing import List, Optional, Tuple
import pandas as pd
from optuna.integration import LightGBMPruningCallback
from lightgbm.callback import early_stopping


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

    def _fit_model_and_select_features_with_loss_curve(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        params: dict,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[np.ndarray] = None,
        trial: Optional[optuna.trial.Trial] = None,
        should_plot_loss_curve: bool = False,
    ) -> Tuple[LGBMClassifier, np.ndarray, dict[str, pd.Index]]:
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
            Tuple[LGBMClassifier, np.ndarray, dict[str, pd.Index]]:
                Fitted model, selected feature names, and categorical schema from
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

        current_categories = X_train_selected.select_dtypes(
            include=["category"]
        ).columns.tolist()
        category_schema = {
            col: X_train_selected[col].cat.categories for col in current_categories
        }

        model = self._fetch_model()
        model.set_params(**fit_params)

        has_validation = X_val_selected is not None and y_val is not None
        record_curve = should_plot_loss_curve and has_validation

        callbacks = []
        if trial is not None:
            valid_name = "valid_1" if record_curve else "valid_0"
            callbacks.append(
                LightGBMPruningCallback(trial, "binary_logloss", valid_name=valid_name)
            )

        results = {}
        eval_set = None
        eval_names = None
        if has_validation:
            if record_curve:
                eval_set = [(X_train_selected, y_train), (X_val_selected, y_val)]
                eval_names = ["train", "valid"]
                callbacks.append(lgb.record_evaluation(results))
            else:
                eval_set = [(X_val_selected, y_val)]
            callbacks.append(early_stopping(stopping_rounds=50, verbose=False))

        model.fit(
            X_train_selected,
            y_train,
            eval_set=eval_set,
            eval_names=eval_names,
            eval_metric="binary_logloss",
            callbacks=callbacks if callbacks else None,
            categorical_feature=current_categories if current_categories else "auto",
        )

        if record_curve and results:
            train_loss = results.get("train", {}).get("binary_logloss")
            valid_loss = results.get("valid", {}).get("binary_logloss")
            if train_loss and valid_loss:
                fig = plot_loss_curve(train_loss, valid_loss, self.model_type)
                self._log_figure(name=f"{self.model_type}_loss_curve", fig=fig)

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
    ) -> Tuple[LGBMClassifier, np.ndarray, dict[str, pd.Index], Optional[object]]:
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
            self._log_figure(name="feature_importance", fig=fig)

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
                self.logger.info(f"Fitting OpenFE")
                row_wise_features, column_wise_features = (
                    self.open_fe_transformations.fit(
                        X_train=X_train,
                        y_train=y_train,
                        task="classification",
                        categorical_features=self._categorical_feature_names(X_train),
                        feature_boosting=True,
                        n_jobs=self.n_jobs,
                    )
                )

                X_train, X_val, mapping = (
                    self.open_fe_transformations.apply_openfe_features(
                        X_train=X_train,
                        X_val=X_val,
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

                self._log_artifact(f"ofe_row_wise_features", row_wise_features)
                self._log_artifact(
                    f"ofe_column_wise_features_fold",
                    column_wise_features,
                )
                self._log_artifact(f"ofe_feature_mapping_fold", full_mapping)

                for feat_name, formula in full_mapping.items():
                    self.logger.info(f"Feature: {feat_name:20} | Formula: {formula}")

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
            outer_fold_log_loss = self._eval_validation_loss(
                X_val_outer=X_val,
                y_val_outer=y_val,
                selected_features=selected_features,
                final_model=final_model,
                category_schema=category_schema,
                column_transformer=column_transformer,
            )

            logged_params = self._augment_params_with_boosting_rounds(
                best_params=best_params, final_model=final_model
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
