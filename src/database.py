"""
VISIONFORGE AI lightweight investigation database.
SQLite is used so the prototype has a durable audit store without an external service.
"""
from __future__ import annotations
import json, sqlite3
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "evidence" / "visionforge_final.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS investigations(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    target_type TEXT NOT NULL,
    query_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS candidates(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    investigation_id INTEGER,
    created_at TEXT NOT NULL,
    camera_id TEXT,
    track_id INTEGER,
    candidate_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS audit_log(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    action TEXT NOT NULL,
    detail_json TEXT
);
"""

def connect():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.executescript(SCHEMA)
    return conn

def record_investigation(target_type: str, query: dict) -> int:
    with connect() as conn:
        cur = conn.execute(
            "INSERT INTO investigations(created_at,target_type,query_json) VALUES(?,?,?)",
            (datetime.utcnow().isoformat(), target_type, json.dumps(query)),
        )
        conn.commit()
        return int(cur.lastrowid)

def record_candidates(investigation_id: int, candidates: list[dict]) -> None:
    now = datetime.utcnow().isoformat()
    with connect() as conn:
        conn.executemany(
            "INSERT INTO candidates(investigation_id,created_at,camera_id,track_id,candidate_json) VALUES(?,?,?,?,?)",
            [
                (
                    investigation_id, now,
                    c.get("target_camera") or c.get("camera_id"),
                    c.get("target_track") or c.get("track_id"),
                    json.dumps(c),
                )
                for c in candidates
            ],
        )
        conn.commit()

def audit(action: str, detail: dict | None = None) -> None:
    with connect() as conn:
        conn.execute(
            "INSERT INTO audit_log(created_at,action,detail_json) VALUES(?,?,?)",
            (datetime.utcnow().isoformat(), action, json.dumps(detail or {})),
        )
        conn.commit()
