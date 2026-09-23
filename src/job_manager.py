from __future__ import annotations
import json, threading, uuid
from datetime import datetime
from pathlib import Path
from typing import Callable, Any

ROOT=Path(__file__).resolve().parent.parent
JOBS=ROOT/"data"/"investigation"/"jobs.json"
_lock=threading.Lock()

def _load():
    if not JOBS.exists(): return []
    try: return json.loads(JOBS.read_text(encoding="utf-8"))
    except Exception: return []

def _save(rows):
    JOBS.parent.mkdir(parents=True, exist_ok=True)
    JOBS.write_text(json.dumps(rows, indent=2), encoding="utf-8")

def submit(name: str, fn: Callable[...,Any], *args, **kwargs):
    job_id=uuid.uuid4().hex
    with _lock:
        rows=_load()
        rows.append({"job_id":job_id,"name":name,"status":"queued","created_at":datetime.utcnow().isoformat()})
        _save(rows)
    def run():
        try:
            with _lock:
                rows=_load()
                for r in rows:
                    if r["job_id"]==job_id: r.update(status="running",started_at=datetime.utcnow().isoformat())
                _save(rows)
            result=fn(*args,**kwargs)
            with _lock:
                rows=_load()
                for r in rows:
                    if r["job_id"]==job_id: r.update(status="completed",finished_at=datetime.utcnow().isoformat(),result=str(result))
                _save(rows)
        except Exception as exc:
            with _lock:
                rows=_load()
                for r in rows:
                    if r["job_id"]==job_id: r.update(status="failed",finished_at=datetime.utcnow().isoformat(),error=repr(exc))
                _save(rows)
    threading.Thread(target=run,daemon=True).start()
    return job_id


def list_jobs():
    with _lock:
        return list(reversed(_load()))


def get_job(job_id: str):
    with _lock:
        for row in _load():
            if row.get("job_id") == job_id:
                return row
    return None
