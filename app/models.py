from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator


ActionType = Literal[
    "read_logs",
    "check_metrics",
    "inspect_dependency",
    "restart_service",
    "scale_service",
    "rollback_deployment",
    "update_config",
]


class Action(BaseModel):
    action_type: ActionType
    target: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None


class Observation(BaseModel):
    logs: str
    metrics: Dict[str, float] = Field(
        default_factory=lambda: {"cpu": 0.0, "memory": 0.0, "latency": 0.0}
    )
    alerts: List[str] = Field(default_factory=list)
    step_count: int = Field(ge=0)


class Reward(BaseModel):
    value: float = Field(ge=-1.0, le=1.0)
    components: Dict[str, float] = Field(default_factory=dict)
    reason: str = ""

    @field_validator("value")
    @classmethod
    def round_value(cls, value: float) -> float:
        return round(value, 4)
