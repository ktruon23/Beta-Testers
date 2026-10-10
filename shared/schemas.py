"""The message formats every agent agrees on.

Agents never import each other's code -- they only send these JSON shapes
over HTTP. New Planner Agent fields are optional for backward compatibility.
"""

from typing import Literal, Optional
from pydantic import BaseModel, Field


class Hold(BaseModel):
    id: int
    x: float
    y: float
    type: str = "unknown"
    color: Optional[str] = None


class Climber(BaseModel):
    height_in: float = Field(..., gt=0, description="Climber height in inches")
    wingspan_in: float = Field(..., gt=0, description="Fingertip to fingertip, arms out, in inches")
    reach_factor: float = Field(0.85, gt=0, le=1.5, description="Max single move = wingspan * reach_factor. Tune this with a real climber.")
    level: Literal["beginner", "intermediate", "advanced"] = "intermediate"

    @property
    def reach_in(self) -> float:
        return self.wingspan_in * self.reach_factor


class Move(BaseModel):
    from_id: int
    to_id: int
    distance_in: float
    reach_ratio: float
    dx_in: float
    dy_in: float
    from_type: str
    to_type: str


class DetectRequest(BaseModel):
    image_b64: str
    sat_threshold: int = 90
    min_area_frac: float = 0.00015
    max_area_frac: float = 0.03


class DetectResponse(BaseModel):
    holds: list[Hold]
    image_width: int
    image_height: int
    method: str


class CriticFeedback(BaseModel):
    reasoning: str = ""
    avoid_holds: list[int] = Field(default_factory=list)
    avoid_moves: list[list[int]] = Field(default_factory=list)


class PlannerPushback(BaseModel):
    constraint: dict = Field(default_factory=dict)
    reason: str
    evidence: dict[str, float] = Field(default_factory=dict)
    proposal: str


class PlanAlternative(BaseModel):
    strategy: str
    path: list[int] = Field(default_factory=list)
    total_cost: float = 0.0
    max_reach_ratio: float = 0.0
    move_count: int = 0
    worst_hold_type: str = "unknown"
    ok: bool = True
    error: Optional[str] = None


class PlanRequest(BaseModel):
    holds: list[Hold]
    start_id: int
    finish_id: int
    inches_per_pixel: float = Field(..., gt=0)
    climber: Climber
    avoid_holds: list[int] = Field(default_factory=list)
    avoid_moves: list[list[int]] = Field(default_factory=list)
    critic_feedback: Optional[CriticFeedback] = None


class PlanResponse(BaseModel):
    ok: bool
    path: list[int] = Field(default_factory=list)
    moves: list[Move] = Field(default_factory=list)
    reach_in: float
    total_cost: float = 0.0
    error: Optional[str] = None
    strategy: str = "balanced"
    reasoning: str = ""
    alternatives: list[PlanAlternative] = Field(default_factory=list)
    pushback: Optional[PlannerPushback] = None
    source: Literal["claude", "deterministic"] = "deterministic"


class CritiqueRequest(BaseModel):
    plan: PlanResponse
    holds: list[Hold]
    climber: Climber
    start_id: int
    finish_id: int
    round: int
    max_rounds: int
    planner_pushback: Optional[PlannerPushback] = None


class CritiqueResponse(BaseModel):
    approved: bool
    reasoning: str
    avoid_holds: list[int] = Field(default_factory=list)
    avoid_moves: list[list[int]] = Field(default_factory=list)
    source: str


class CoachRequest(BaseModel):
    plan: PlanResponse
    holds: list[Hold]
    climber: Climber


class BetaStep(BaseModel):
    move: int
    instruction: str


class CoachResponse(BaseModel):
    summary: str
    steps: list[BetaStep]
    tips: list[str] = Field(default_factory=list)
    source: str


class SolveRequest(BaseModel):
    holds: list[Hold]
    start_id: Optional[int] = None
    finish_id: Optional[int] = None
    inches_per_pixel: float = Field(..., gt=0)
    climber: Climber
    max_rounds: int = Field(4, ge=1, le=8)


class TraceEntry(BaseModel):
    step: int
    sender: str
    receiver: str
    summary: str
    payload: dict = Field(default_factory=dict)


class SolveResponse(BaseModel):
    ok: bool
    status: str
    holds: list[Hold]
    start_id: Optional[int] = None
    finish_id: Optional[int] = None
    plan: Optional[PlanResponse] = None
    critic_approved: bool = False
    beta: Optional[CoachResponse] = None
    trace: list[TraceEntry] = Field(default_factory=list)
    rounds: int = 0
