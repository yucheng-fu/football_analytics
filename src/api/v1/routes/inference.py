from fastapi import APIRouter

router = APIRouter(prefix="/inference", tags=["inference"])


@router.get("/")
def inference():
    return {"prediction": "complete", "confidence": 0.97}
