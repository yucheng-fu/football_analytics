import logging

from fastapi import APIRouter, Depends, HTTPException, Request
from api.core.auth import verify_api_key
from api.core.dependencies import get_model_service
from api.schemas.request import InferenceRequest
from api.schemas.response import InferenceResponse
from api.services.model_service import ModelService

router = APIRouter(prefix="/inference")
logger = logging.getLogger(__name__)


@router.get("/health", tags=["Inference"])
def health(request: Request) -> dict[str, bool]:
    service = getattr(request.app.state, "model_service", None)
    model_loaded = service is not None and service.is_model_available()
    return {"model_loaded": model_loaded}


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
    service: ModelService = Depends(get_model_service),
    _: str = Depends(verify_api_key),
) -> InferenceResponse:
    """Run inference for a validated request payload.

    Args:
        payload (InferenceRequest): Inference input payload.
        service (ModelService, optional): Injected model service.
        _ (str, optional): API key dependency placeholder.

    Raises:
        HTTPException: If model is not loaded or inference fails.

    Returns:
        InferenceResponse: Prediction payload.
    """
    if not service.is_model_available():
        raise HTTPException(status_code=500, detail="Model unavailable")

    try:
        return service.predict(payload)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Prediction error: %s", exc)
        raise HTTPException(status_code=500, detail="Internal inference error") from exc


@router.post(
    "/predict/batch",
    tags=["Inference"],
    response_model=list[InferenceResponse],
    responses={
        401: {"description": "Unauthorized: invalid API key"},
        422: {"description": "Validation Error"},
        500: {"description": "Inference failed"},
    },
)
def predict_batch(
    payloads: list[InferenceRequest],
    service: ModelService = Depends(get_model_service),
    _: str = Depends(verify_api_key),
) -> list[InferenceResponse]:
    """Run inference for multiple validated request payloads."""
    if not service.is_model_available():
        raise HTTPException(status_code=500, detail="Model unavailable")

    try:
        return [service.predict(payload) for payload in payloads]
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Batch prediction error: %s", exc)
        raise HTTPException(status_code=500, detail="Internal inference error") from exc
