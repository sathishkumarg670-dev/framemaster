from __future__ import annotations
import json, multiprocessing, queue, threading, uuid
from datetime import datetime
from pathlib import Path
from typing import Callable, Any

ROOT=Path(__file__).resolve().parent.parent
JOBS=ROOT/"data"/"investigation"/"jobs.json"
_lock=threading.Lock()
# Jobs run one at a time on a single worker: each processing job loads YOLO and
# Re-ID models, and parallel CPU jobs only slow each other down.
_queue: "queue.Queue[Callable[[], None]]"=queue.Queue()
_worker: threading.Thread | None=None

def _child(conn, fn, args, kwargs):
    try: conn.send(("ok", fn(*args, **kwargs)))
    except BaseException as exc: conn.send(("error", repr(exc)))
    finally: conn.close()

def _run_in_process(fn: Callable[...,Any], args, kwargs):
    """Run fn on the main thread of a fresh process.

    Keeps a native crash in torch/OpenCV from killing the API server (and any
    ESP32 recorder running in it), releases model memory after every job, and
    avoids torch.load crashing in non-main threads (seen on Python 3.15 + torch 2.14).
    fn must be a module-level function and args picklable.
    """
    ctx=multiprocessing.get_context("spawn")
    parent,child=ctx.Pipe(duplex=False)
    proc=ctx.Process(target=_child,args=(child,fn,args,kwargs),name="vf-job")
    proc.start()
    child.close()
    try:
        status,value=parent.recv()
    except EOFError:
        proc.join()
        raise RuntimeError(f"job process crashed (exit code {proc.exitcode})")
    finally:
        parent.close()
    proc.join()
    if status=="error": raise RuntimeError(value)
    return value

def _work():
    while True:
        task=_queue.get()
        try: task()
        finally: _queue.task_done()

def _ensure_worker():
    global _worker
    if _worker is None or not _worker.is_alive():
        _worker=threading.Thread(target=_work,name="vf-job-worker",daemon=True)
        _worker.start()

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
            result=_run_in_process(fn,args,kwargs)
            # The pipeline reports stage failures in its summary instead of raising.
            if isinstance(result,dict) and result.get("status")=="failed":
                raise RuntimeError(result.get("error") or "job reported failure")
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
    with _lock:
        _ensure_worker()
    _queue.put(run)
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
