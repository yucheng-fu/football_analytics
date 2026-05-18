from typing import Literal

from pydantic import BaseModel, Field

from api.schemas.enums import Bodypart, PassHeight


class InferenceRequest(BaseModel):
    start_x: float = Field(..., example=60.0)
    start_y: float = Field(..., example=40.0)
    end_x: float = Field(..., example=78.0)
    end_y: float = Field(..., example=35.0)
    length: float = Field(..., example=22.5)
    height: PassHeight = Field(..., example="Ground Pass")
    angle: float = Field(..., example=0.25)
    duration: float = Field(..., example=1.35)
    body_part: Bodypart = Field(..., example="Right Foot")
    under_pressure: Literal[0, 1] = Field(..., example=0)
