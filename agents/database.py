"""
Tiny SQLite store for the Coach agent's memory of each climber.
 
We store features that carry over between walls (hold types, how stretched the
move was) rather than hold IDs, because hold #7 on Monday's photo is not hold #7
on Tuesday's photo.
 
Always pass values with `?` placeholders (never f-strings) so notes typed by a
user can't inject SQL.
"""
 
import os
import sqlite3
from contextlib import closing
from typing import Optional
 
from shared.schemas import Move
 
DB_PATH = os.environ.get("COACH_DB", "coach.db")
 
 
def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # rows behave like dicts: row["success"]
    return conn
 
 
def init() -> None:
    """Create the table if it doesn't exist yet. Safe to call on every startup."""
    with closing(_connect()) as conn:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS attempts (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id     TEXT NOT NULL,
                from_type   TEXT,
                to_type     TEXT,
                reach_ratio REAL,
                dy_in       REAL,
                success     INTEGER NOT NULL,
                notes       TEXT,
                ts          TEXT DEFAULT CURRENT_TIMESTAMP
            )"""
        )
        conn.commit()
 
 
def record(user_id: str, move: Move, success: bool, notes: str) -> None:
    """Save one attempt at one move."""
    with closing(_connect()) as conn:
        conn.execute(
            "INSERT INTO attempts (user_id, from_type, to_type, reach_ratio, dy_in, success, notes) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (user_id, move.from_type, move.to_type, move.reach_ratio, move.dy_in, int(success), notes),
        )
        conn.commit()
 
 
def history(user_id: str, to_type: Optional[str] = None, limit: int = 5) -> dict:
    """Summarize a climber's record (optionally only moves INTO one hold type).
 
    Returns counts plus the most recent failures. Claude gets this small summary
    instead of raw rows, which keeps its context short and focused.
    """
    sql = "SELECT * FROM attempts WHERE user_id = ?"
    params: list = [user_id]
    if to_type:
        sql += " AND to_type = ?"
        params.append(to_type)
    sql += " ORDER BY id DESC"  # newest first
 
    with closing(_connect()) as conn:
        rows = conn.execute(sql, params).fetchall()
 
    failures = [
        {
            "from_type": r["from_type"],
            "to_type": r["to_type"],
            "reach_ratio": round(r["reach_ratio"], 2),
            "notes": r["notes"] or "",
        }
        for r in rows
        if not r["success"]
    ]
    return {
        "attempts": len(rows),
        "successes": sum(r["success"] for r in rows),
        "recent_failures": failures[:limit],
    }