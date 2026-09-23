from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "config" / "camera_topology.json"

def load_topology(path: Path = DEFAULT_CONFIG) -> dict:
    if not path.exists():
        return {"transitions": []}
    return json.loads(path.read_text(encoding="utf-8"))

def transition_rule(config: dict, source: str, target: str) -> dict | None:
    for row in config.get("transitions", []):
        if row.get("source") == source and row.get("target") == target:
            return row
    return None

def validate_transition(config: dict, source: str, target: str, gap_seconds: float) -> bool | None:
    rule = transition_rule(config, source, target)
    if rule is None:
        return None
    return float(rule.get("min_gap_seconds", 0)) <= gap_seconds <= float(rule.get("max_gap_seconds", 1e9))
