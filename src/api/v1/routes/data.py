import mlflow
from fastapi import APIRouter, Depends, HTTPException

from api.core.auth import verify_api_key
from api.core.dependencies import get_inference_frame_service
from api.schemas.request import InferenceRequest
from api.services.inference_frame_service import InferenceFrameService

router = APIRouter(prefix="/data")


@router.post(
    "/inference-frame",
    tags=["Data"],
    responses={
        401: {"description": "Unauthorized: invalid API key"},
        422: {"description": "Validation Error"},
        500: {"description": "Inference frame generation failed"},
    },
)
def inference_frame(
    payload: InferenceRequest,
    inference_frame_service: InferenceFrameService = Depends(get_inference_frame_service),
    _: str = Depends(verify_api_key),
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
