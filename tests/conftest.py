"""Shared test setup.

Every agent is tested in-process with FastAPI's TestClient: no servers, no
network, no LLM. The LLM env vars are blanked BEFORE any agent is imported, so
shared.llm.backend() is "none" and every agent takes its rule-based path even if
your .env has an ANTHROPIC_API_KEY (load_dotenv never overrides a variable that
is already set).
"""

import os
import sys
import tempfile
from pathlib import Path

for var in ("ANTHROPIC_API_KEY", "LLM_BASE_URL", "LLM_PROVIDER", "LLM_MODEL", "LLM_API_KEY"):
    os.environ[var] = ""
os.environ["COACH_DB"] = os.path.join(tempfile.mkdtemp(), "coach-test.db")

ROOT = Path(__file__).resolve().parent.parent
for p in (ROOT, ROOT / "vision-agent", ROOT / "planner-agent", ROOT / "coach-agent"):
    sys.path.insert(0, str(p))

import pytest  # noqa: E402

from shared.schemas import Climber, Hold, Move  # noqa: E402


def ladder(n=5, spacing_px=100, x=100, types=None):
    """n holds stacked straight up, bottom first (id 0 has the largest y)."""
    types = types or ["jug"] * n
    return [Hold(id=i, x=x, y=1000 - i * spacing_px, type=types[i]) for i in range(n)]


def move(from_id=0, to_id=1, distance=20.0, ratio=0.4, dx=0.0, dy=20.0, from_type="jug", to_type="jug"):
    return Move(from_id=from_id, to_id=to_id, distance_in=distance, reach_ratio=ratio,
                dx_in=dx, dy_in=dy, from_type=from_type, to_type=to_type)


@pytest.fixture
def climber():
    # reach = 68 * 0.85 = 57.8 in
    return Climber(height_in=66, wingspan_in=68)
