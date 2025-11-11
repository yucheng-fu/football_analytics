from dataclasses import dataclass, field
from pydantic import BaseModel, Field
from typing import List, Any, Optional


@dataclass
class OuterCVResults:
    scores: List[float] = field(default_factory=list)
    params: List[dict] = field(default_factory=list)
    features: List[Any] = field(default_factory=list)
    run_ids: List[str] = field(default_factory=list)


class LGBMParams(BaseModel):
    bagging_freq: Optional[int] = 0
    colsample_bytree: Optional[float] = 1.0
    learning_rate: Optional[float] = 0.1
    max_depth: Optional[int] = -1
    min_child_samples: Optional[int] = 20
    num_leaves: Optional[int] = 31
    n_estimators: Optional[int] = 100
    reg_alpha: Optional[float] = 0.0
    reg_lambda: Optional[float] = 0.0
    subsample: Optional[float] = 1.0
