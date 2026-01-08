run_name = f"baseline_tuning_and_feature_selection_final"
experiment_name = "Model selection and hyperparameter tuning"
model_param_tuner = ModelParamTuner(
    model_type=model_type,
    n_inner_splits=5,
    n_trials=500,
    run_name=run_name,
    experiment_name=experiment_name,
)

result_tuning = model_param_tuner.tune_and_train(X=X_train, y=y_train)
result_tuning = None

parent_run_id = get_parent_run_id_from_experiment(
    result=result_cv, experiment_id=MODEL_SELECTION_EXPERIMENT_ID
)
best_params, best_features = get_best_params_and_features_from_parent_run_id(
    parent_run_id=parent_run_id
)

model_trainer = ModelTrainer(
    model_type=model_type,
    best_params=best_params,
    best_features=best_features,
    run_name="baseline_training",
    experiment_name="Final models",
)

trained_model = model_trainer.train(X_train=X_train, y_train=y_train)
# trained_model = get_registered_model(model_type=lightgbm_model_name, version="latest", model_registry_name="Final models_lightgbm")

test_processing_handler = PreprocessingHandler(
    df=test_df, categorical_columns=categorical_columns
)
test_df_temp = test_processing_handler.preprocess_categorical_columns()
test_df_temp = test_processing_handler.preprocess_outcome_column()
y_test = test_df_temp.select("outcome").to_series()
X_test_temp = test_df_temp.drop(pl.col("outcome"))

test_feature_engineering_handler = FeatureEngineeringHandler(X=X_test_temp)

X_test = test_feature_engineering_handler.encode_columns(
    columns=["height", "body_part"]
)

explainer = shap.TreeExplainer(trained_model)
shap_values = explainer.shap_values(X_test.select(best_features).to_pandas())

# This calculates the mean absolute SHAP value for each feature
shap.summary_plot(
    shap_values,
    X_test.select(best_features).to_pandas(),
    plot_type="bar",
    max_display=20,
)

import numpy as np

shap_importance = np.abs(shap_values).sum(axis=0) / shap_values.shape[0]

# 25th percentile for SHAP importances
percentile_25_shap = np.percentile(shap_importance, 25)
feature_names = X_test.select(best_features).columns

# create polars dataframe for SHAP importances
shap_importance_df = pl.DataFrame(
    {"Feature": feature_names, "ShapImportance": shap_importance}
)
rename_map = {
    "height_Low pass": "height_Low Pass",
    "height_Ground_Pass": "height_Ground Pass",
    "height_High_Pass": "height_High Pass",
}

shap_importance_df = shap_importance_df.with_columns(
    pl.col("Feature").replace(rename_map)
)
shap_importance_df = shap_importance_df.sort("ShapImportance", descending=True)

# filter by 25th percentile
filtered_shap_importance_df = shap_importance_df.filter(
    pl.col("ShapImportance") >= float(percentile_25_shap)
)

print(f"25th percentile (shap): {percentile_25_shap}")
print(filtered_shap_importance_df)

model_evaluator = ModelEval(
    X_train=X_train,
    y_train=y_train,
    model=trained_model,
    best_features=best_features,
    experiment_name="Evaluation",
)

model_evaluator.eval(X_test=X_test, y_test=y_test)
