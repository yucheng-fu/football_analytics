from dataclasses import dataclass, field
from typing import List, Any


@dataclass
class OuterCVResults:
    scores: List[float] = field(default_factory=list)
    params: List[dict] = field(default_factory=list)
    features: List[Any] = field(default_factory=list)
    run_ids: List[str] = field(default_factory=list)
