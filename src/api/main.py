from contextlib import asynccontextmanager
import os
from pathlib import Path

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI

from api.services.inference_frame_service import InferenceFrameService
from api.services.model_service import ModelService
from api.v1.router import api_router
from utils.inference_utils import load_inference_bundle_from_local_artifacts


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_dotenv()
    default_artifact_dir = str(Path(__file__).resolve().parent / "artifacts")
    artifact_dir = os.getenv("APP_ARTIFACT_DIR", default_artifact_dir)
    app.state.inference_bundle = load_inference_bundle_from_local_artifacts(artifact_dir)
    app.state.inference_frame_service = InferenceFrameService(app.state.inference_bundle)
    app.state.model_service = ModelService(app.state.inference_bundle)
    yield


app = FastAPI(
    title="Football Analytics Inference API",
    description="Production-ready inference API for local MLflow model artifacts.",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(api_router, prefix="/api/v1")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port="8000")
