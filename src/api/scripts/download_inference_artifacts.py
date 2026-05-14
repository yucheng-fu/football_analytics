import argparse
import json
import os
import pickle
from datetime import datetime, timezone

import mlflow
import numpy as np

from utils.statics import (
    FINAL_MODELS_EXPERIMENT_ID,
    MODEL_SELECTION_EXPERIMENT_ID,
    lightgbm_model_name,
)
from utils.inference_utils import (
    fetch_categorical_mapping_by_run_id,
    fetch_fitted_column_transformer_by_run_id,
    fetch_model,
    get_ofe_feature_nodes_from_run_id,
    get_parent_run_id_from_experiment,
    get_best_params_and_features_from_parent_run_id,
)


def download_inference_artifacts(
    output_dir: str,
    model_name: str,
    model_alias: str,
    final_models_experiment_id: str,
    model_selection_experiment_id: str,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    result = None
    final_model_run_id = get_parent_run_id_from_experiment(
        result=result,
        experiment_id=final_models_experiment_id,
    )
    tuning_run_id = get_parent_run_id_from_experiment(
        result=result,
        experiment_id=model_selection_experiment_id,
    )

    model_uri = f"models:/{model_name}@{model_alias}"
    model = fetch_model(model_name=model_name, alias=model_alias)
    mlflow.lightgbm.save_model(model, os.path.join(output_dir, "model"))

    row_wise_features, column_wise_features = get_ofe_feature_nodes_from_run_id(
        run_id=tuning_run_id
    )
    with open(os.path.join(output_dir, "row_wise_features.pkl"), "wb") as f:
        pickle.dump(row_wise_features, f)
    with open(os.path.join(output_dir, "column_wise_features.pkl"), "wb") as f:
        pickle.dump(column_wise_features, f)

    fitted_column_transformer = fetch_fitted_column_transformer_by_run_id(
        run_id=final_model_run_id
    )
    with open(
        os.path.join(output_dir, "fitted_column_transformer.pkl"), "wb"
    ) as transformer_file:
        pickle.dump(fitted_column_transformer, transformer_file)

    best_params, selected_features = get_best_params_and_features_from_parent_run_id(
        parent_run_id=tuning_run_id
    )
    with open(os.path.join(output_dir, "params.json"), "w", encoding="utf-8") as f:
        json.dump(best_params, f, ensure_ascii=True, indent=2)
    with open(os.path.join(output_dir, "best_params.json"), "w", encoding="utf-8") as f:
        json.dump(best_params, f, ensure_ascii=True, indent=2)
    with open(
        os.path.join(output_dir, "best_features.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(np.asarray(selected_features).tolist(), f, ensure_ascii=True, indent=2)
    with open(
        os.path.join(output_dir, "selected_features.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(np.asarray(selected_features).tolist(), f, ensure_ascii=True, indent=2)

    categorical_mapping = fetch_categorical_mapping_by_run_id(run_id=final_model_run_id)
    with open(
        os.path.join(output_dir, "categorical_mapping.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(categorical_mapping, f, ensure_ascii=True, indent=2)

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_uri": model_uri,
        "final_model_run_id": final_model_run_id,
        "tuning_run_id": tuning_run_id,
        "final_models_experiment_id": final_models_experiment_id,
        "model_selection_experiment_id": model_selection_experiment_id,
    }
    with open(os.path.join(output_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=True, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and freeze inference artifacts for API deployment."
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join("src", "api", "artifacts"),
        help="Directory where artifacts are written.",
    )
    parser.add_argument(
        "--model-name",
        default="Final models_lightgbm",
        help="MLflow registered model name.",
    )
    parser.add_argument(
        "--model-alias",
        default="production",
        help="MLflow registered model alias.",
    )
    parser.add_argument(
        "--tracking-uri",
        default=None,
        help="Optional MLflow tracking URI override.",
    )
    parser.add_argument(
        "--final-models-experiment-id",
        default=FINAL_MODELS_EXPERIMENT_ID,
        help="Experiment ID for FINAL_MODELS artifacts.",
    )
    parser.add_argument(
        "--model-selection-experiment-id",
        default=MODEL_SELECTION_EXPERIMENT_ID,
        help="Experiment ID for tuning/feature artifacts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.tracking_uri:
        mlflow.set_tracking_uri(args.tracking_uri)

    download_inference_artifacts(
        output_dir=args.output_dir,
        model_name=args.model_name,
        model_alias=args.model_alias,
        final_models_experiment_id=args.final_models_experiment_id,
        model_selection_experiment_id=args.model_selection_experiment_id,
    )


if __name__ == "__main__":
    main()
