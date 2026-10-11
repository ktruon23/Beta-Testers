"""The message formats every agent agrees on.

Agents never import each other's code -- they only send these JSON shapes
over HTTP. New Planner Agent fields are optional for backward compatibility.
"""

from typing import Literal, Optional
from pydantic import BaseModel, Field, field_validator

# Hold types the agents understand. The camera usually can't tell them apart, so
# any hold Vision can't classify is treated as DEFAULT_HOLD_TYPE unless the user says otherwise.
HoldType = Literal["jug", "pinch", "pocket", "crimp", "sloper"]
HOLD_TYPES: tuple[str, ...] = ("jug", "pinch", "pocket", "crimp", "sloper")
DEFAULT_HOLD_TYPE = "jug"  # the standard gym hold; most routes are set mostly on jugs


def _clean_hold_types(v: dict) -> dict:
    """Lowercase user-typed hold types ("Crimp" -> "crimp") before validation."""
    return {k: t.strip().lower() if isinstance(t, str) else t for k, t in (v or {}).items()}


class Hold(BaseModel):
    id: int
    x: float
    y: float
    type: str = "unknown"
    color: Optional[str] = None


class Climber(BaseModel):
    # Upper bounds catch centimeters typed into an inches field (180 cm would be a 153 in reach).
    height_in: float = Field(..., gt=0, le=108, description="Climber height in inches")
    wingspan_in: float = Field(..., gt=0, le=120, description="Fingertip to fingertip, arms out, in inches")
    reach_factor: float = Field(0.85, gt=0, le=1.5, description="Max single move = wingspan * reach_factor. Tune this with a real climber.")
    level: Literal["beginner", "intermediate", "expert"] = "intermediate"

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
    min_conf: float = Field(0.5, ge=0, le=1, description="Drop detections below this confidence")
    # Legacy OpenCV color-blob settings; unused by the Roboflow detector.
    sat_threshold: int = 90
    min_area_frac: float = 0.00015
    max_area_frac: float = 0.03


class DetectResponse(BaseModel):
    holds: list[Hold]
    image_width: int
    image_height: int
    method: str
    inches_per_pixel: Optional[float] = None  # estimated from median hold size
    scale_method: Optional[str] = None        # "median-hold-size" | "fallback-wall-height"
    colors: dict[str, int] = Field(default_factory=dict)  # hold count per color, for the route-color picker


class RouteDifficulty(BaseModel):
    """How hard the route is for THIS climber, judged from hold types and reach."""
    grade: str                      # "V0".."V10"
    score: float                    # 0-10 before rounding to a grade
    label: str                      # easy / moderate / hard / very hard
    hardest_move: Optional[Move] = None
    hold_counts: dict[str, int] = Field(default_factory=dict)  # hold type -> holds on the route
    assumed_holds: list[int] = Field(default_factory=list)     # route holds whose type was assumed, not given
    default_hold_type: str = DEFAULT_HOLD_TYPE
    reasoning: str = ""
    source: str = "rule-based"


class DifficultyRequest(BaseModel):
    plan: "PlanResponse"
    holds: list[Hold]
    climber: Climber
    assumed_holds: list[int] = Field(default_factory=list)
    default_hold_type: HoldType = DEFAULT_HOLD_TYPE


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
    user_id: str = "guest"  # whose attempt history the Coach looks up


class FeedbackRequest(BaseModel):
    """A climber reporting whether one move worked, for the Coach's memory."""
    user_id: str = "guest"
    move: Move
    success: bool
    notes: str = ""


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
    # Hold types the user knows (hold id -> type). Every other unclassified hold uses default_hold_type.
    hold_types: dict[int, HoldType] = Field(default_factory=dict)
    default_hold_type: HoldType = DEFAULT_HOLD_TYPE

    _clean = field_validator("hold_types", mode="before")(_clean_hold_types)

    @field_validator("default_hold_type", mode="before")
    @classmethod
    def _lower_default(cls, v):
        return v.strip().lower() if isinstance(v, str) else v


class GripPick(BaseModel):
    """A grip the climber set by tapping a hold on the photo (image pixels)."""
    x: float
    y: float
    type: HoldType

    @field_validator("type", mode="before")
    @classmethod
    def _lower(cls, v):
        return v.strip().lower() if isinstance(v, str) else v


class SolvePhotoRequest(BaseModel):
    """Photo in, route out: Vision -> color filter -> Planner/Critic loop -> Coach."""
    image_b64: str
    climber: Climber
    route_color: Optional[str] = None           # keep only holds of this color; None = all holds
    start_id: Optional[int] = None              # ids refer to the filtered hold list
    finish_id: Optional[int] = None
    # Or pick them on the photo: (x, y) in image pixels, snapped to the nearest hold on the route.
    start_point: Optional[tuple[float, float]] = None
    finish_point: Optional[tuple[float, float]] = None
    inches_per_pixel: Optional[float] = Field(None, gt=0)  # overrides Vision's estimate
    min_conf: float = Field(0.5, ge=0, le=1)
    max_rounds: int = Field(4, ge=1, le=8)
    hold_types: dict[int, HoldType] = Field(default_factory=dict)  # ids refer to the filtered hold list
    grips: list[GripPick] = Field(default_factory=list)  # set on the photo; snapped to the hold under each tap
    default_hold_type: HoldType = DEFAULT_HOLD_TYPE

    _clean = field_validator("hold_types", mode="before")(_clean_hold_types)

    @field_validator("default_hold_type", mode="before")
    @classmethod
    def _lower_default(cls, v):
        return v.strip().lower() if isinstance(v, str) else v


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
    difficulty: Optional[RouteDifficulty] = None
    beta: Optional[CoachResponse] = None
    trace: list[TraceEntry] = Field(default_factory=list)
    rounds: int = 0
    orchestrator_source: str = "fixed"  # "claude" or "fixed" -- which pipeline ran


DifficultyRequest.model_rebuild()
