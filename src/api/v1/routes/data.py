import mlflow
from fastapi import APIRouter, Depends, HTTPException
import pandas as pd

from api.core.dependencies import get_inference_frame_service
from api.schemas.request import InferenceRequest
from api.services.inference_frame_service import InferenceFrameService

router = APIRouter(prefix="/data")


@router.post(
    "/inference-frame",
    tags=["Data"],
    responses={
        422: {"description": "Validation Error"},
        500: {"description": "Inference frame generation failed"},
    },
)
def inference_frame(
    payload: InferenceRequest,
    inference_frame_service: InferenceFrameService = Depends(
        get_inference_frame_service
    ),
) -> dict:
    try:
        frame = inference_frame_service.build_from_payload(payload)
        return {"inference_frame": frame.to_dict(orient="records")}
    except mlflow.exceptions.MlflowException as exc:
        raise HTTPException(
            status_code=500, detail=f"Inference frame generation failed: {exc}"
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Inference frame generation failed: {exc}"
        ) from exc


@router.post(
    "/inference-frame/batch",
    tags=["Data"],
    responses={
        422: {"description": "Validation Error"},
        500: {"description": "Inference frame generation failed"},
    },
)
def inference_frame_batch(
    payloads: list[InferenceRequest],
    inference_frame_service: InferenceFrameService = Depends(
        get_inference_frame_service
    ),
) -> dict:
    """Build model-ready inference frame rows for multiple payloads."""
    try:
        frames = [
            inference_frame_service.build_from_payload(payload) for payload in payloads
        ]
        merged_frame = (
            pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        )
        return {"inference_frame": merged_frame.to_dict(orient="records")}
    except mlflow.exceptions.MlflowException as exc:
        raise HTTPException(
            status_code=500, detail=f"Inference frame generation failed: {exc}"
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Inference frame generation failed: {exc}"
        ) from exc
