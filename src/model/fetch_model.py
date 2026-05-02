import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import mlflow
import mlflow.lightgbm
from mlflow.tracking import MlflowClient
import argparse
from src.utils.statics import tracking_uri


def fetch_model(model_name: str, alias: str = "production"):
    """
    Fetch a trained model from MLflow Model Registry via alias.

    Args:
        model_name (str): The name of the model to fetch.
        alias (str): Registered model alias to resolve (default: "production").
    """
    mlflow.set_tracking_uri(tracking_uri)
    mlflow_client = MlflowClient()
    model_version_info = mlflow_client.get_model_version_by_alias(model_name, alias)
    model_version = model_version_info.version
    model_uri = f"models:/{model_name}@{alias}"
    print(f"Loading model '{model_name}' alias '{alias}' (version {model_version})...")
    model = mlflow.lightgbm.load_model(model_uri)
    print(f"Loaded model from '{model_uri}'.")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch a trained model from MLflow.")
    parser.add_argument(
        "--model_name", type=str, help="The name of the model to fetch."
    )
    parser.add_argument(
        "--alias",
        type=str,
        default="production",
        help="Registered model alias to resolve (default: 'production').",
    )
    args = parser.parse_args()

    fetch_model(args.model_name, args.alias)
