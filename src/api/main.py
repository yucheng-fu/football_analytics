from contextlib import asynccontextmanager
import os
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from dotenv import load_dotenv

from api.v1.router import api_router
from utils.utils import load_inference_bundle_from_local_artifacts


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_dotenv()
    default_artifact_dir = str(Path(__file__).resolve().parent / "artifacts")
    artifact_dir = os.getenv("APP_ARTIFACT_DIR", default_artifact_dir)
    app.state.inference_bundle = load_inference_bundle_from_local_artifacts(
        artifact_dir
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


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port="8000")
