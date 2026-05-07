from typing import Any
from pydantic import BaseModel


class InferenceResponse(BaseModel):
    prediction: int
    probability: float | None = None
    timestamp: str
