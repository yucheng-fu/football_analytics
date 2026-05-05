import os
from datetime import datetime, timezone
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

router = APIRouter(prefix="/inference")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def verify_api_key(api_key: str | None = Depends(api_key_header)) -> str:
    expected_key = os.getenv("APP_API_KEY")
    if not expected_key or api_key != expected_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized: invalid API key",
        )
    return api_key


class InferenceRequest(BaseModel):
    start_x: float = Field(..., example=60.0)
    start_y: float = Field(..., example=40.0)
    end_x: float = Field(..., example=78.0)
    end_y: float = Field(..., example=35.0)
    length: float = Field(..., example=22.5)
    height: str = Field(..., example="Ground Pass")
    angle: float = Field(..., example=0.25)
    duration: float = Field(..., example=1.35)
    body_part: str = Field(..., example="Right Foot")
    under_pressure: int = Field(..., example=0)
    log_velocity: float = Field(..., example=2.6)
    angle_sin: float = Field(..., example=0.2474)
    angle_cos: float = Field(..., example=0.9689)
    start_distance_to_goal: float = Field(..., example=61.2)
    end_distance_to_goal: float = Field(..., example=43.6)
    progressive_distance: float = Field(..., example=17.6)
    direction_to_goal: float = Field(..., example=-0.12)
    direction_to_goal_cos: float = Field(..., example=0.9928)
    duration_x_under_pressure: float = Field(..., example=0.0)
    log_velocity_x_under_pressure: float = Field(..., example=0.0)
    length_x_under_pressure: float = Field(..., example=0.0)
    autoFE_f_0: float = Field(..., example=3.95)
    autoFE_f_1: float = Field(..., example=1.35)
    autoFE_f_2: float = Field(..., example=0.66)
    autoFE_f_3: float = Field(..., example=1.35)
    ofe_col_1: float = Field(..., example=0.14)


class InferenceResponse(BaseModel):
    prediction: Any
    probability: float | None = None
    timestamp: str


@router.get("/health", tags=["Inference"])
def health(request: Request) -> dict[str, Any]:
    model_loaded = getattr(request.app.state, "model", None) is not None
    return {"status": "ok", "model_loaded": model_loaded}


@router.post(
    "/predict",
    tags=["Inference"],
    response_model=InferenceResponse,
    responses={
        401: {"description": "Unauthorized: invalid API key"},
        422: {"description": "Validation Error"},
        500: {"description": "Inference failed"},
    },
)
def predict(
    payload: InferenceRequest,
    request: Request,
    _: str = Depends(verify_api_key),
) -> InferenceResponse:
    model = getattr(request.app.state, "model", None)
    if model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded")

    try:
        input_df = pd.DataFrame([payload.model_dump()])
        prediction_raw = model.predict(input_df)

        prediction = prediction_raw[0]
        if isinstance(prediction, np.generic):
            prediction = prediction.item()

        probability = None
        if hasattr(model, "predict_proba"):
            proba_raw = model.predict_proba(input_df)
            if isinstance(proba_raw, np.ndarray):
                if proba_raw.ndim == 2 and proba_raw.shape[1] > 1:
                    probability = float(proba_raw[0, 1])
                else:
                    probability = float(proba_raw[0])

        return InferenceResponse(
            prediction=prediction,
            probability=probability,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
    except mlflow.exceptions.MlflowException as exc:
        raise HTTPException(status_code=500, detail="Inference failed") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Inference failed") from exc
