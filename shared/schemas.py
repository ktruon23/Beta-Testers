"""
The message formats every agent agrees on.

Agents never import each other's code -- they only send these JSON shapes
over HTTP. If you change a field here, every agent that sends or reads it
needs to know.
"""

from typing import Optional
from pydantic import BaseModel, Field


# ---------- Basic pieces ----------

class Hold(BaseModel):
    id: int
    x: float  # pixel coordinates in the image the user sees
    y: float  # (y grows DOWNWARD, like all image coordinates)
    type: str = "unknown"  # jug, crimp, sloper, pinch, pocket, unknown
    color: Optional[str] = None


class Climber(BaseModel):
    height_in: float = Field(..., gt=0, description="Climber height in inches")
    wingspan_in: float = Field(..., gt=0, description="Fingertip to fingertip, arms out, in inches")
    reach_factor: float = Field(
        0.85,
        gt=0,
        le=1.5,
        description="Max single move = wingspan * reach_factor. Tune this with a real climber.",
    )

    @property
    def reach_in(self) -> float:
        return self.wingspan_in * self.reach_factor


class Move(BaseModel):
    from_id: int
    to_id: int
    distance_in: float
    reach_ratio: float  # distance / climber reach. 1.0 = fully stretched
    dx_in: float  # + means right
    dy_in: float  # + means UP (already flipped from image coordinates)
    from_type: str
    to_type: str


# ---------- Vision Agent ----------

class DetectRequest(BaseModel):
    image_b64: str  # PNG/JPEG bytes, base64 encoded
    sat_threshold: int = 90  # how colorful a pixel must be to count as "hold"
    min_area_frac: float = 0.00015
    max_area_frac: float = 0.03


class DetectResponse(BaseModel):
    holds: list[Hold]
    image_width: int
    image_height: int
    method: str


# ---------- Planner Agent ----------

class PlanRequest(BaseModel):
    holds: list[Hold]
    start_id: int
    finish_id: int
    inches_per_pixel: float = Field(..., gt=0)
    climber: Climber
    avoid_holds: list[int] = []
    avoid_moves: list[list[int]] = []  # pairs like [3, 7]


class PlanResponse(BaseModel):
    ok: bool
    path: list[int] = []
    moves: list[Move] = []
    reach_in: float
    total_cost: float = 0.0
    error: Optional[str] = None


# ---------- Critic Agent ----------

class CritiqueRequest(BaseModel):
    plan: PlanResponse
    holds: list[Hold]
    climber: Climber
    start_id: int
    finish_id: int
    round: int
    max_rounds: int


class CritiqueResponse(BaseModel):
    approved: bool
    reasoning: str
    avoid_holds: list[int] = []
    avoid_moves: list[list[int]] = []
    source: str  # "claude" or "rule-based (...)"


# ---------- Coach Agent ----------

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
    tips: list[str] = []
    source: str


# ---------- Orchestrator (what the frontend talks to) ----------

class SolveRequest(BaseModel):
    holds: list[Hold]
    start_id: Optional[int] = None  # defaults: lowest hold
    finish_id: Optional[int] = None  # defaults: highest hold
    inches_per_pixel: float = Field(..., gt=0)
    climber: Climber
    max_rounds: int = Field(4, ge=1, le=8)


class TraceEntry(BaseModel):
    step: int
    sender: str
    receiver: str
    summary: str
    payload: dict = {}


class SolveResponse(BaseModel):
    ok: bool
    status: str
    holds: list[Hold]
    start_id: Optional[int] = None
    finish_id: Optional[int] = None
    plan: Optional[PlanResponse] = None
    critic_approved: bool = False
    beta: Optional[CoachResponse] = None
    trace: list[TraceEntry] = []
    rounds: int = 0
