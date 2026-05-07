from contextlib import asynccontextmanager
import os

from fastapi import FastAPI

from api.v1.router import api_router
from utils.statics import FINAL_MODELS_EXPERIMENT_ID
from utils.utils import fetch_inference_bundle


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_experiment_id = os.getenv(
        "APP_MODEL_EXPERIMENT_ID", FINAL_MODELS_EXPERIMENT_ID
    )
    model_run_id = os.getenv("APP_MODEL_RUN_ID")
    metadata_run_id = os.getenv("APP_METADATA_RUN_ID", model_run_id)

    app.state.inference_bundle = fetch_inference_bundle(
        model_experiment_id=model_experiment_id,
        model_run_id=model_run_id,
        metadata_run_id=metadata_run_id,
    )
    app.state.model = app.state.inference_bundle["model"]
    yield


app = FastAPI(
    title="Football Analytics Inference API",
    description="Production-ready inference API for local MLflow model artifacts.",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(api_router, prefix="/api/v1")
