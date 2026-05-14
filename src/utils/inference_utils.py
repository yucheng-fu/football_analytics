import json
import os
import pickle
import tempfile
from typing import Any, Optional

import mlflow
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient

from feature_engineering.OpenFE.openfe import tree_to_formula
from feature_engineering.RowWiseTransformations import RowWiseTransformations
from model.data_classes import LGBMParams, OuterCVResults
from utils.statics import lightgbm_model_name, tracking_uri


def get_parent_run_id_from_experiment(
    result: OuterCVResults | None, experiment_id: str
) -> str:
    """Get parent run ID from experiment."""
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    if result is not None:
        parent_run_id = client.get_run(result.parent_run_id).info.run_id
    else:
        runs = client.search_runs(
            experiment_ids=[experiment_id],
            order_by=["attributes.start_time DESC"],
        )
        parent_run_id = next(
            run.info.run_id for run in runs if "mlflow.parentRunId" not in run.data.tags
        )

    return parent_run_id


def get_best_params_and_features_from_parent_run_id(
    parent_run_id: str,
) -> tuple[dict, np.ndarray]:
    """Get params and features from parent run."""
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    parent_run = client.get_run(parent_run_id)

    raw_params = parent_run.data.params
    best_params = dict(raw_params)
    lgbm_params = LGBMParams(
        **{k: raw_params[k] for k in LGBMParams.model_fields if k in raw_params}
    )
    best_params.update(lgbm_params.model_dump())

    best_features = np.array(parent_run.data.tags["selected_features"].split(","))

    return (best_params, best_features)


def resolve_downloaded_pickle_path(local_path: str) -> str:
    """Resolve a pickle file path returned by MLflow artifact download."""
    if os.path.isfile(local_path):
        return local_path

    if os.path.isdir(local_path):
        pkl_files = sorted(
            os.path.join(root, filename)
            for root, _, files in os.walk(local_path)
            for filename in files
            if filename.endswith(".pkl")
        )
        if pkl_files:
            return pkl_files[0]

    raise ValueError(
        f"Could not resolve pickle file from downloaded path: {local_path}"
    )


def get_ofe_feature_nodes_from_run_id(
    run_id: str,
    row_artifact_name: Optional[str] = None,
    column_artifact_name: Optional[str] = None,
) -> tuple[list, list]:
    """Load row-wise and column-wise OpenFE feature-node pickles from a run."""
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    artifacts = client.list_artifacts(run_id, path="pickles")
    artifact_paths = [item.path for item in artifacts]

    if row_artifact_name is None:
        row_candidates = sorted(
            [p for p in artifact_paths if "row_wise_features" in os.path.basename(p)]
        )
        if not row_candidates:
            raise ValueError(f"No row-wise feature pickle found under run_id={run_id}")
        row_artifact_path = row_candidates[-1]
    else:
        row_artifact_path = f"pickles/{row_artifact_name}.pkl"

    if column_artifact_name is None:
        column_candidates = sorted(
            [p for p in artifact_paths if "column_wise_features" in os.path.basename(p)]
        )
        if not column_candidates:
            raise ValueError(
                f"No column-wise feature pickle found under run_id={run_id}"
            )
        column_artifact_path = column_candidates[-1]
    else:
        column_artifact_path = f"pickles/{column_artifact_name}.pkl"

    with tempfile.TemporaryDirectory() as tmp_dir:
        row_local = client.download_artifacts(run_id, row_artifact_path, tmp_dir)
        col_local = client.download_artifacts(run_id, column_artifact_path, tmp_dir)
        row_file = resolve_downloaded_pickle_path(row_local)
        col_file = resolve_downloaded_pickle_path(col_local)

        with open(row_file, "rb") as f:
            row_wise_features = pickle.load(f)
        with open(col_file, "rb") as f:
            column_wise_features = pickle.load(f)

    return row_wise_features, column_wise_features


def fetch_model(model_name: str, alias: str = "production"):
    """Fetch a trained model from MLflow Model Registry via alias."""
    mlflow.set_tracking_uri(tracking_uri)
    mlflow_client = MlflowClient()
    model_version_info = mlflow_client.get_model_version_by_alias(model_name, alias)
    model_version = model_version_info.version
    model_uri = f"models:/{model_name}@{alias}"
    print(f"Loading model '{model_name}' alias '{alias}' (version {model_version})...")
    model = mlflow.lightgbm.load_model(model_uri)
    print(f"Loaded model from '{model_uri}'.")
    return model


def fetch_categorical_mapping_by_run_id(run_id: str) -> dict[str, list]:
    """Fetch saved categorical mapping from MLflow run tags."""
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    run = client.get_run(run_id)
    mapping_raw = run.data.tags.get("categorical_mapping")
    if not mapping_raw:
        raise ValueError(f"No categorical_mapping tag found for run_id={run_id}.")
    try:
        mapping = json.loads(mapping_raw)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid categorical_mapping JSON for run_id={run_id}."
        ) from exc
    if not isinstance(mapping, dict):
        raise ValueError(f"categorical_mapping tag is not a dict for run_id={run_id}.")
    return mapping


def fetch_fitted_column_transformer_by_run_id(
    run_id: str, artifact_name: str = "fitted_column_transformer"
):
    """Fetch fitted ColumnTransformer pickle artifact from an MLflow run."""
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    artifact_path = f"pickles/{artifact_name}"

    with tempfile.TemporaryDirectory() as tmp_dir:
        local_path = client.download_artifacts(run_id, artifact_path, tmp_dir)
        pickle_file = resolve_downloaded_pickle_path(local_path)
        with open(pickle_file, "rb") as f:
            return pickle.load(f)


def load_inference_bundle_from_local_artifacts(
    artifact_dir: str = "src/api/artifacts",
) -> dict[str, Any]:
    """Load inference artifacts from local files bundled with the API."""
    model_dir = os.path.join(artifact_dir, "model")
    transformer_path = os.path.join(artifact_dir, "fitted_column_transformer.pkl")
    row_wise_features_path = os.path.join(artifact_dir, "row_wise_features.pkl")
    column_wise_features_path = os.path.join(artifact_dir, "column_wise_features.pkl")
    params_path = os.path.join(artifact_dir, "params.json")
    best_params_path = os.path.join(artifact_dir, "best_params.json")
    best_features_path = os.path.join(artifact_dir, "best_features.json")
    selected_features_path = os.path.join(artifact_dir, "selected_features.json")
    categorical_mapping_path = os.path.join(artifact_dir, "categorical_mapping.json")

    resolved_params_path = params_path if os.path.exists(params_path) else best_params_path
    resolved_best_features_path = (
        best_features_path
        if os.path.exists(best_features_path)
        else selected_features_path
    )

    required_paths = [
        model_dir,
        transformer_path,
        row_wise_features_path,
        column_wise_features_path,
        resolved_params_path,
        resolved_best_features_path,
        categorical_mapping_path,
    ]
    missing = [path for path in required_paths if not os.path.exists(path)]
    if missing:
        raise ValueError(
            "Missing local inference artifacts under "
            f"'{artifact_dir}': {missing}. "
            "Generate them with: "
            "`python -m api.scripts.download_inference_artifacts --tracking-uri <MLFLOW_TRACKING_URI>`"
        )

    model = mlflow.lightgbm.load_model(model_dir)
    with open(transformer_path, "rb") as transformer_file:
        fitted_column_transformer = pickle.load(transformer_file)
    with open(row_wise_features_path, "rb") as row_wise_features_file:
        row_wise_features = pickle.load(row_wise_features_file)
    with open(column_wise_features_path, "rb") as column_wise_features_file:
        column_wise_features = pickle.load(column_wise_features_file)
    with open(resolved_params_path, "r", encoding="utf-8") as params_file:
        params = json.load(params_file)
    with open(
        resolved_best_features_path, "r", encoding="utf-8"
    ) as best_features_file:
        best_features = np.array(json.load(best_features_file))
    with open(
        categorical_mapping_path, "r", encoding="utf-8"
    ) as categorical_mapping_file:
        categorical_mapping = json.load(categorical_mapping_file)

    return {
        "model": model,
        "fitted_column_transformer": fitted_column_transformer,
        "row_wise_features": row_wise_features,
        "column_wise_features": column_wise_features,
        "params": params,
        "best_features": best_features,
        "best_params": params,
        "selected_features": best_features,
        "categorical_mapping": categorical_mapping,
        "artifact_dir": artifact_dir,
    }


def safe_production_transform(X_new, fitted_features_list):
    """Apply fitted OpenFE row-wise features to new data."""
    X_out = X_new.copy()

    for feature in fitted_features_list:
        feature.calculate(X_out, is_root=True)
        name = tree_to_formula(feature)
        X_out[name] = feature.data.values.ravel()
        feature.data = None

    return X_out


def apply_saved_categorical_mapping(
    X_pd: pd.DataFrame, categorical_mapping: dict[str, list] | None
) -> pd.DataFrame:
    """Apply saved training categorical mapping for production-safe inference."""
    if not categorical_mapping:
        return X_pd

    X_out = X_pd.copy()
    for col, categories in categorical_mapping.items():
        if col in X_out.columns:
            X_out[col] = pd.Categorical(X_out[col], categories=categories)
    return X_out


def fetch_inference_bundle(
    model_experiment_id: str,
    model_run_id: str | None = None,
    metadata_run_id: str | None = None,
    column_transformer_artifact_name: str = "fitted_column_transformer",
) -> dict[str, Any]:
    """Fetch a complete inference bundle from MLflow."""
    resolved_metadata_run_id = metadata_run_id or model_run_id
    if resolved_metadata_run_id is None:
        raise ValueError(
            "metadata_run_id is required when model_run_id is not provided."
        )

    model = fetch_model(lightgbm_model_name, alias="production")
    fitted_column_transformer = fetch_fitted_column_transformer_by_run_id(
        run_id=resolved_metadata_run_id,
        artifact_name=column_transformer_artifact_name,
    )
    best_params, selected_features = get_best_params_and_features_from_parent_run_id(
        parent_run_id=resolved_metadata_run_id
    )
    categorical_mapping = fetch_categorical_mapping_by_run_id(
        run_id=resolved_metadata_run_id
    )

    return {
        "model": model,
        "fitted_column_transformer": fitted_column_transformer,
        "best_params": best_params,
        "selected_features": selected_features,
        "categorical_mapping": categorical_mapping,
        "model_run_id": model_run_id,
        "metadata_run_id": resolved_metadata_run_id,
        "model_experiment_id": model_experiment_id,
    }


def prepare_inference_frame(
    X_pd: pd.DataFrame,
    row_wise_features: list | None = None,
    column_wise_features: list | None = None,
    column_transformer: Any | None = None,
    best_features: np.ndarray | list[str] | None = None,
    categorical_mapping: dict[str, list] | None = None,
    row_wise_transformations: RowWiseTransformations | None = None,
) -> pd.DataFrame:
    """Build inference frame using eval-equivalent preprocessing steps."""
    transformer = (
        row_wise_transformations
        if row_wise_transformations is not None
        else RowWiseTransformations()
    )
    X_out = transformer.apply_row_wise_transformations(X_pd.copy())

    if row_wise_features:
        X_out = safe_production_transform(X_out, row_wise_features)

    if column_transformer is not None:
        transformed = column_transformer.transform(
            X_out, feature_nodes=column_wise_features
        )
        if isinstance(transformed, pd.DataFrame):
            X_out = transformed
        else:
            X_out = pd.DataFrame(transformed, index=X_out.index)

    if row_wise_features:
        X_alias = X_out.copy()
        for idx, node in enumerate(row_wise_features):
            legacy_name = f"autoFE_f_{idx}"
            formula_name = tree_to_formula(node)
            if legacy_name not in X_alias.columns and formula_name in X_alias.columns:
                X_alias[legacy_name] = X_alias[formula_name]
        X_out = X_alias

    if best_features is not None:
        X_out = X_out.loc[:, list(best_features)]

    return apply_saved_categorical_mapping(X_out, categorical_mapping)
