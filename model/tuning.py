    from typing import Callable

    import polars as pl
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    from sklearn.model_selection import StratifiedKFold
    from sklearn.feature_selection import RFECV
    from sklearn.metrics import log_loss
    import optuna
    from utils.statics import xgboost_model_name, lightgbm_model_name, tracking_uri
    import mlflow
    from mlflow.entities import Run
    from mlflow.models import infer_signature
    from mlflow.tracking import MlflowClient
    from mlflow.utils.mlflow_tags import MLFLOW_PARENT_RUN_ID
    import numpy as np
    import logging
    from model.dataclasses import OuterCVResults


    class ModelParamTuner:
        logger = logging.getLogger(__name__)
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
        )

        def __init__(
            self,
            model_type: str,
            n_inner_splits: int = 10,
            n_outer_splits: int = 5,
            n_trials: int = 20,
            run_name: str = "baseline_hyperparameter_tuning",
            experiment_name: str = "Model selection and hyperparameter tuning",
        ):
            self.model_type = model_type
            self.n_inner_splits = n_inner_splits
            self.n_outer_splits = n_outer_splits
            self.n_trials = n_trials
            self.run_name = run_name
            self.experiment_name = experiment_name
            self.inner_cv = StratifiedKFold(
                n_splits=self.n_inner_splits, shuffle=True, random_state=165
            )
            self.outer_cv = StratifiedKFold(
                n_splits=self.n_outer_splits, shuffle=True, random_state=165
            )

        def setup_mlflow(self):
            self.logger.info(
                f"""Starting training with model {self.model_type} with the following configuration:
            - {self.n_inner_splits} inner splits
            - {self.n_outer_splits} outer splits
            - {self.n_trials} trials"""
            )

            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(experiment_name=self.experiment_name)

        def fetch_model(self) -> XGBClassifier | LGBMClassifier:
            """
            Return a fresh estimator instance for the configured model type.

            Returns:
                XGBClassifier | LGBMClassifier: An uninitialized classifier corresponding to self.model_type.

            Raises:
                ValueError: If self.model_type is not a supported model name.
            """
            if self.model_type == xgboost_model_name:
                return XGBClassifier()
            elif self.model_type == lightgbm_model_name:
                return LGBMClassifier(verbose=-1)

            raise ValueError(f"Unsupported model type: {self.model_type}")

        def fetch_param_suggestions(self, trial: optuna.trial.Trial) -> dict:
            """
            Generate hyperparameter suggestions for the configured model using an Optuna trial.

            Args:
                trial (optuna.trial.Trial): An Optuna Trial object used to sample hyperparameter values.

            Returns:
                dict: A dictionary of hyperparameter names and sampled values suitable for the selected model type

            Raises:
                ValueError: If self.model_type is not one of the supported model names.
            """
            if self.model_type == xgboost_model_name:
                return {
                    "n_estimators": trial.suggest_int(
                        "n_estimators", 100, 1000
                    ),  # number of trees
                    "max_depth": trial.suggest_int(
                        "max_depth", 3, 16
                    ),  # maximum depth of each tree
                    "learning_rate": trial.suggest_float(
                        "learning_rate", 0.01, 0.3
                    ),  # step size for optimisation
                    "subsample": trial.suggest_float(
                        "subsample", 0.5, 1.0
                    ),  # fraction of samples to be used for each tree
                    "colsample_bytree": trial.suggest_float(
                        "colsample_bytree", 0.5, 1.0
                    ),  # fraction of features used per tree
                    "min_child_weight": trial.suggest_int(
                        "min_child_weight", 1, 100
                    ),  # minimum sum of instance weights needed in a leaf to allow a split
                    "alpha": trial.suggest_float("alpha", 0.0, 1.0),  # L1 regularisation
                    "lambda": trial.suggest_float("lambda", 0.0, 1.0),  # L2 regularisation
                    "eval_metric": "logloss",  # evaluation metric
                    "random_state": 165,  # seed for reproducibility
                }

            elif self.model_type == lightgbm_model_name:
                max_depth = trial.suggest_int("max_depth", 3, 16)
                max_num_leaves = min(
                    2**max_depth - 1, 512
                )  # less than 2^(max_depth), cap at 512
                num_leaves = trial.suggest_int("num_leaves", 7, max_num_leaves)

                return {
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

            raise ValueError(f"Unsupported model type: {self.model_type}")

        def log_model(
            self,
            final_model: XGBClassifier | LGBMClassifier,
            X_data: pl.DataFrame,
            output: np.ndarray,
        ) -> None:
            """Log model in outer fold run

            Args:
                final_model (XGBClassifier | LGBMClassifier): Trained model
                X_data (pl.DataFrame): One outer fold of X
                output (np.ndarray): Predictions from final_model

            Raises:
                ValueError: If self.model_type is not a supported model name.
            """
            X_data_pd = X_data.to_pandas()
            signature = infer_signature(X_data_pd, output)
            input_example = X_data_pd.head(10)

            log_fn_mapping = {
                xgboost_model_name: mlflow.xgboost.log_model,
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
        ):
            outer_cv_results.scores.append(outer_fold_log_loss)
            outer_cv_results.params.append(best_params)
            outer_cv_results.features.append(selected_features)
            outer_cv_results.run_ids.append(run.info.run_id)

            mlflow.set_tag("selected_features", ",".join(map(str, selected_features)))
            mlflow.log_metric("log_loss", outer_fold_log_loss)
            mlflow.log_params(best_params)

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
                    client.log_metric(trial_run_id, "neg_log_loss", trial.value)

                # Kill the run at teh end
                client.set_terminated(trial_run_id, "FINISHED")

            return _callback

        def get_features(self, X_train_outer: pl.DataFrame, rfecv: RFECV) -> np.ndarray:
            """Get features from feature selection using RFECV

            Args:
                X_train_outer (pl.DataFrame): Outer fold of X_train
                rfecv (RFECV): RFECV object

            Returns:
                np.ndarray: array of selected features
            """
            selected_feature_mask = rfecv.get_support()
            selected_features = np.array(X_train_outer.columns)[selected_feature_mask]
            return selected_features

        def eval_validation_loss(
            self,
            X_val_outer: pl.DataFrame,
            y_val_outer: pl.DataFrame,
            rfecv: RFECV,
            selected_features: np.ndarray,
        ) -> float:
            """Evaluate model on validation set

            Args:
                X_val_outer (pl.DataFrame): X_val outer fold
                y_val_outer (pl.DataFrame): y_val outer fold
                rfecv (RFECV): RFECV obejct
                selected_features (np.ndarray): Selected features from feature selection step

            Returns:
                float: log_loss evaluted on the outer fold validation set
            """
            final_model = rfecv.estimator_
            y_pred_proba = final_model.predict_proba(
                X_val_outer.select(selected_features).to_numpy()
            )

            outer_fold_log_loss = log_loss(y_val_outer, y_pred_proba)

            self.log_model(
                final_model=rfecv.estimator_,
                X_data=X_val_outer.select(selected_features),
                output=y_pred_proba,
            )

            return outer_fold_log_loss

        def _objective(
            self,
            trial: optuna.trial.Trial,
            X_train_outer: pl.DataFrame,
            y_train_outer: pl.DataFrame,
        ) -> np.ndarray:
            """Objective function for Optuna hyperparameter tuning

            Args:
                trial (optuna.trial.Trial): Optuna trial
                X_train_outer (pl.DataFrame): X_train outer fold split
                y_train_outer (pl.DataFrame): y_train outer fold split

            Returns:
                np.ndarray: Maximum mean cross-validation score obtained by the Optuna trial across the inner folds
            """
            params = self.fetch_param_suggestions(trial)
            base_estimator = self.fetch_model()
            base_estimator.set_params(**params)

            rfecv = RFECV(
                estimator=base_estimator,
                step=1,
                cv=self.inner_cv,
                scoring="neg_log_loss",
                n_jobs=-1,
            )

            # perform CV on inner folds and get maximum mean_score evaluated across the inner folds
            rfecv.fit(X_train_outer, y_train_outer)
            mean_score = np.max(rfecv.cv_results_["mean_test_score"])

            return -mean_score

        def tune_and_train(
            self, X_train: pl.DataFrame, y_train: pl.DataFrame
        ) -> OuterCVResults:
            """Perform hyperparameter optimisation and select best performing model using two-layer nested cross-validation

            Args:
                X_train (pl.DataFrame): Input features used for training
                y_train (pl.DataFrame): Ground truths used for training

            Returns:
                OuterCVResults: Outer cross-validation results
            """
            outer_cv_results = OuterCVResults()
            self.setup_mlflow()
            with mlflow.start_run(run_name=f"{self.run_name}_{self.model_type}"):
                for i, (train_idx, val_idx) in enumerate(
                    self.outer_cv.split(X_train, y_train)
                ):

                    X_train_outer, X_val_outer = X_train[train_idx], X_train[val_idx]
                    y_train_outer, y_val_outer = y_train[train_idx], y_train[val_idx]

                    with mlflow.start_run(nested=True, run_name=f"Outer_fold_{i+1}") as run:
                        parent_id = run.info.run_id
                        callback_fn = self._mlflow_callback(
                            outer_fold_run_id=parent_id,
                            experiment_name=self.experiment_name,
                        )

                        study = optuna.create_study(direction="minimize")

                        study.optimize(
                            lambda trial: self._objective(
                                trial, X_train_outer, y_train_outer
                            ),
                            n_trials=self.n_trials,
                            n_jobs=-1,
                            show_progress_bar=True,
                            callbacks=[callback_fn],
                        )

                        best_trial = study.best_trial
                        best_params = best_trial.params.copy()

                        # Fit model on outer training fold
                        final_estimator = self.fetch_model()
                        final_estimator.set_params(**best_params)

                        rfecv = RFECV(
                            estimator=final_estimator,
                            step=1,
                            cv=self.inner_cv,
                            scoring="neg_log_loss",
                            n_jobs=-1,
                        )
                        rfecv.fit(X_train_outer, y_train_outer)

                        selected_features = self.get_features(
                            X_train_outer=X_train_outer, rfecv=rfecv
                        )

                        outer_fold_log_loss = self.eval_validation_loss(
                            X_val_outer=X_val_outer,
                            y_val_outer=y_val_outer,
                            rfecv=rfecv,
                            selected_features=selected_features,
                        )

                        self.append_and_log_metrics_and_params(
                            outer_cv_results=outer_cv_results,
                            selected_features=selected_features,
                            outer_fold_log_loss=outer_fold_log_loss,
                            best_params=best_params,
                            run=run,
                        )

            best_idx = np.argmax(outer_cv_results.scores)

            result = OuterCVResults(
                scores=[outer_cv_results.scores[best_idx]],
                params=[outer_cv_results.params[best_idx]],
                features=[outer_cv_results.features[best_idx]],
                run_ids=[outer_cv_results.run_ids[best_idx]],
            )

            return result


    if __name__ == "__main__":
        pass
