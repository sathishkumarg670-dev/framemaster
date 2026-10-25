"""
VISIONFORGE AI - ESP32-CAM recorder.

Pulls the MJPEG stream from an ESP32-CAM node (hardware/esp32cam_stream_v1 or v2),
stores it as fixed-length MP4 segments in data/videos/, and queues each stored
segment for the normal processing pipeline (tracking -> crops -> attributes -> Re-ID).

Hardware-driven choices:
  * ESP32 fps is variable (WiFi jitter, ~5-15 fps at VGA). Frames are timestamped
    on arrival and each segment is written at its measured average fps, with
    frames placed by timestamp, so pipeline timestamps match real time.
  * Segments are written to data/videos/.recording/ and moved into data/videos/
    only when complete, so a half-written file is never discovered or processed.
  * Segments without motion are still stored, but are not auto-processed: on a
    CPU-only PC that skips YOLO/Re-ID on empty scenes. They can be processed
    manually from the dashboard.

CLI:
    python -m src.esp32_recorder --url http://192.168.1.50/stream
    python -m src.esp32_recorder --port COM9          (discover the IP over serial)
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import threading
import time
import urllib.request
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from src.job_manager import submit as submit_job


PROJECT_ROOT = Path(__file__).resolve().parent.parent
VIDEO_DIR = PROJECT_ROOT / "data" / "videos"
TEMP_DIR = VIDEO_DIR / ".recording"
SEGMENT_LOG = PROJECT_ROOT / "data" / "esp32" / "segments.jsonl"

SOI = b"\xff\xd8"
EOI = b"\xff\xd9"

MOTION_SIZE = (160, 120)
MOTION_PIXEL_DELTA = 25        # grey-level change that counts as a changed pixel
MOTION_AREA_FRACTION = 0.01    # share of changed pixels that counts as motion
MOTION_INTERVAL_SECONDS = 0.5
MAX_BUFFER_BYTES = 2_000_000   # drop garbage if no complete JPEG appears
MIN_SEGMENT_SECONDS = 1.0


def iter_mjpeg_frames(stream, stop_event: threading.Event, chunk_size: int = 16384):
    """Yield (arrival_time, jpeg_bytes) from any MJPEG HTTP stream.

    Scans for JPEG start/end markers instead of parsing multipart headers, so it
    also works with the stock CameraWebServer example and other MJPEG sources.
    """
    buffer = bytearray()
    # read1 returns as soon as any bytes arrive; read(n) would block until n
    # bytes, batching several small QVGA frames under one arrival timestamp.
    read = getattr(stream, "read1", stream.read)
    while not stop_event.is_set():
        chunk = read(chunk_size)
        if not chunk:
            raise ConnectionError("stream closed by camera")
        buffer.extend(chunk)

        while True:
            start = buffer.find(SOI)
            if start < 0:
                buffer.clear()
                break
            end = buffer.find(EOI, start + 2)
            if end < 0:
                if start > 0:
                    del buffer[:start]
                break
            yield time.time(), bytes(buffer[start:end + 2])
            del buffer[:end + 2]

        if len(buffer) > MAX_BUFFER_BYTES:
            buffer.clear()


def _decode(jpeg: bytes):
    return cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)


def write_segment(frames: list[tuple[float, bytes]], destination: Path) -> dict:
    """Write timestamped JPEG frames as a constant-fps MP4 and measure motion."""
    duration = frames[-1][0] - frames[0][0]
    fps = (len(frames) - 1) / duration if duration > 0 else 1.0
    fps = float(min(30.0, max(1.0, round(fps, 2))))

    first = None
    for _, jpeg in frames:
        first = _decode(jpeg)
        if first is not None:
            break
    if first is None:
        raise ValueError("no decodable JPEG frames in segment")

    height, width = first.shape[:2]
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    temp_path = TEMP_DIR / destination.name
    writer = cv2.VideoWriter(str(temp_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"OpenCV could not open a video writer for {temp_path}")

    t0 = frames[0][0]
    output_count = int(duration * fps) + 1
    source_index = 0
    decoded_index = -1
    current = first
    previous_small = None
    previous_time = t0
    motion_peak = 0.0
    written = 0

    try:
        for k in range(output_count):
            target_time = t0 + k / fps
            # Latest source frame at or before this output slot; repeats the
            # previous frame across WiFi gaps, drops extras in bursts.
            while source_index + 1 < len(frames) and frames[source_index + 1][0] <= target_time:
                source_index += 1

            if source_index != decoded_index:
                decoded_index = source_index
                image = _decode(frames[source_index][1])
                if image is not None:
                    if image.shape[:2] != (height, width):
                        image = cv2.resize(image, (width, height))
                    current = image

                    # Compare against a frame ~0.5 s older: consecutive frames
                    # at ~10 fps barely differ when a person moves slowly.
                    frame_time = frames[source_index][0]
                    if previous_small is None or frame_time - previous_time >= MOTION_INTERVAL_SECONDS:
                        small = cv2.cvtColor(cv2.resize(current, MOTION_SIZE), cv2.COLOR_BGR2GRAY)
                        small = cv2.GaussianBlur(small, (5, 5), 0)
                        if previous_small is not None:
                            changed = cv2.absdiff(small, previous_small) > MOTION_PIXEL_DELTA
                            motion_peak = max(motion_peak, float(changed.mean()))
                        previous_small, previous_time = small, frame_time

            writer.write(current)
            written += 1
    finally:
        writer.release()

    os.replace(temp_path, destination)

    return {
        "filename": destination.name,
        "path": str(destination),
        "fps": fps,
        "width": width,
        "height": height,
        "source_frames": len(frames),
        "written_frames": written,
        "duration_seconds": round(duration, 2),
        "motion_peak": round(motion_peak, 4),
        "motion_detected": motion_peak >= MOTION_AREA_FRACTION,
        "size_bytes": destination.stat().st_size,
    }


class Esp32Recorder:
    def __init__(
        self,
        stream_url: str,
        node_id: str = "esp32cam",
        segment_seconds: int = 60,
        auto_process: bool = True,
        process_static_segments: bool = False,
        frame_stride: int = 1,
    ):
        self.stream_url = stream_url
        self.node_id = node_id.strip().lower().replace(" ", "_") or "esp32cam"
        self.segment_seconds = max(5, int(segment_seconds))
        self.auto_process = auto_process
        self.process_static_segments = process_static_segments
        self.frame_stride = max(1, int(frame_stride))

        self._stop = threading.Event()
        self._segments: "queue.Queue[list[tuple[float, bytes]] | None]" = queue.Queue()
        self._reader: threading.Thread | None = None
        self._writer: threading.Thread | None = None
        self._response = None
        self._lock = threading.Lock()

        self.state = "idle"
        self.last_error: str | None = None
        self.connected_at: float | None = None
        self.frames_received = 0
        self.current_segment_frames = 0
        self.recent_fps = 0.0
        self.segments: list[dict] = []

    # ---------------- lifecycle ----------------

    def start(self):
        if self._reader and self._reader.is_alive():
            return self
        self._stop.clear()
        self.state = "connecting"
        self._writer = threading.Thread(target=self._write_loop, name=f"esp32-writer-{self.node_id}", daemon=True)
        self._reader = threading.Thread(target=self._read_loop, name=f"esp32-reader-{self.node_id}", daemon=True)
        self._writer.start()
        self._reader.start()
        return self

    def stop(self, timeout: float = 15.0):
        self._stop.set()
        response = self._response
        if response is not None:
            try:
                response.close()   # unblocks a pending socket read
            except Exception:
                pass
        if self._reader:
            self._reader.join(timeout)
        self._segments.put(None)
        if self._writer:
            self._writer.join(timeout)
        self.state = "stopped"

    def is_running(self) -> bool:
        return bool(self._reader and self._reader.is_alive())

    # ---------------- reading ----------------

    def _read_loop(self):
        backoff = 2.0
        while not self._stop.is_set():
            frames: list[tuple[float, bytes]] = []
            try:
                self._response = urllib.request.urlopen(self.stream_url, timeout=10)
                self.state = "recording"
                self.connected_at = time.time()
                self.last_error = None
                backoff = 2.0
                segment_start = None
                window_start, window_frames = time.time(), 0

                for arrival, jpeg in iter_mjpeg_frames(self._response, self._stop):
                    if segment_start is None:
                        segment_start = arrival
                    frames.append((arrival, jpeg))
                    self.frames_received += 1
                    self.current_segment_frames = len(frames)

                    window_frames += 1
                    if arrival - window_start >= 2.0:
                        self.recent_fps = round(window_frames / (arrival - window_start), 2)
                        window_start, window_frames = arrival, 0

                    if arrival - segment_start >= self.segment_seconds:
                        self._segments.put(frames)
                        frames, segment_start = [], None
                        self.current_segment_frames = 0
            except Exception as exc:
                if not self._stop.is_set():
                    self.state = "reconnecting"
                    self.last_error = f"{type(exc).__name__}: {exc}"
            finally:
                if self._response is not None:
                    try:
                        self._response.close()
                    except Exception:
                        pass
                    self._response = None
                # Keep whatever was recorded before a disconnect or stop.
                if frames:
                    self._segments.put(frames)
                self.current_segment_frames = 0
                self.recent_fps = 0.0

            if not self._stop.is_set():
                self._stop.wait(backoff)
                backoff = min(backoff * 2, 15.0)

    # ---------------- writing + processing ----------------

    def _write_loop(self):
        while True:
            frames = self._segments.get()
            if frames is None:
                return
            # Too short to track anyone (e.g. the tail captured right at stop).
            if len(frames) < 2 or frames[-1][0] - frames[0][0] < MIN_SEGMENT_SECONDS:
                continue
            started = datetime.fromtimestamp(frames[0][0])
            name = f"esp32_{self.node_id}_{started:%Y%m%d_%H%M%S}.mp4"
            record = {"node_id": self.node_id, "started_at": started.isoformat(timespec="seconds")}
            try:
                record.update(write_segment(frames, VIDEO_DIR / name))
                record["video_id"] = Path(name).stem
                record["processing"] = self._queue_processing(record)
            except Exception as exc:
                record.update(filename=name, error=f"{type(exc).__name__}: {exc}")
            self._log_segment(record)

    def _queue_processing(self, record: dict) -> dict:
        if not self.auto_process:
            return {"status": "not_requested"}
        if not record["motion_detected"] and not self.process_static_segments:
            return {"status": "skipped_no_motion"}
        try:
            from src.pipeline.orchestrator import process_video as run_pipeline
        except Exception as exc:
            return {"status": "unavailable", "error": f"{type(exc).__name__}: {exc}"}
        job_id = submit_job(
            "process_video",
            run_pipeline,
            Path(record["path"]),
            device="cpu",
            frame_stride=self.frame_stride,
        )
        return {"status": "queued", "job_id": job_id}

    def _log_segment(self, record: dict):
        with self._lock:
            self.segments.append(record)
            del self.segments[:-20]
        SEGMENT_LOG.parent.mkdir(parents=True, exist_ok=True)
        with SEGMENT_LOG.open("a", encoding="utf-8") as log:
            log.write(json.dumps(record) + "\n")

    # ---------------- status ----------------

    def status(self) -> dict:
        with self._lock:
            segments = list(reversed(self.segments))
        return {
            "node_id": self.node_id,
            "stream_url": self.stream_url,
            "state": self.state if self.is_running() or self.state == "stopped" else "stopped",
            "running": self.is_running(),
            "last_error": self.last_error,
            "frames_received": self.frames_received,
            "current_segment_frames": self.current_segment_frames,
            "recent_fps": self.recent_fps,
            "segment_seconds": self.segment_seconds,
            "auto_process": self.auto_process,
            "process_static_segments": self.process_static_segments,
            "frame_stride": self.frame_stride,
            "pending_segments": self._segments.qsize(),
            "segments": segments,
        }


# ---------------- registry used by the API ----------------

_recorders: dict[str, Esp32Recorder] = {}
_registry_lock = threading.Lock()


def start_recorder(stream_url: str, node_id: str = "esp32cam", **options) -> Esp32Recorder:
    with _registry_lock:
        recorder = Esp32Recorder(stream_url, node_id, **options)
        existing = _recorders.get(recorder.node_id)
        if existing and existing.is_running():
            raise RuntimeError(f"Recorder for {recorder.node_id} is already running")
        _recorders[recorder.node_id] = recorder
    return recorder.start()


def stop_recorder(node_id: str) -> Esp32Recorder | None:
    with _registry_lock:
        recorder = _recorders.get(node_id)
    if recorder:
        recorder.stop()
    return recorder


def all_status() -> list[dict]:
    with _registry_lock:
        recorders = list(_recorders.values())
    return [recorder.status() for recorder in recorders]


def main():
    parser = argparse.ArgumentParser(description="Record an ESP32-CAM MJPEG stream into data/videos")
    parser.add_argument("--url", help="MJPEG stream URL, e.g. http://192.168.1.50/stream")
    parser.add_argument("--port", help="discover the stream URL from the node's serial output (e.g. COM9)")
    parser.add_argument("--node-id", default=None)
    parser.add_argument("--segment-seconds", type=int, default=60)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--no-process", action="store_true", help="store segments only")
    parser.add_argument("--process-static", action="store_true", help="also process segments without motion")
    args = parser.parse_args()

    url, node_id = args.url, args.node_id
    if not url:
        if not args.port:
            parser.error("give --url or --port")
        from src.hardware_bridge import discover_esp32
        found = discover_esp32(args.port)
        url, node_id = found["stream_url"], node_id or found["node_id"]
        print(f"[esp32] discovered {found['node_id']} at {found['ip']}")

    recorder = start_recorder(
        url,
        node_id or "esp32cam",
        segment_seconds=args.segment_seconds,
        auto_process=not args.no_process,
        process_static_segments=args.process_static,
        frame_stride=args.frame_stride,
    )
    print(f"[esp32] recording {url} -> {VIDEO_DIR} in {recorder.segment_seconds}s segments. Ctrl+C to stop.")
    printed: set[str] = set()
    try:
        while True:
            time.sleep(2)
            status = recorder.status()
            print(f"\r[esp32] {status['state']:<12} {status['recent_fps']:>5.1f} fps  "
                  f"segment frames {status['current_segment_frames']:<5}", end="", flush=True)
            for segment in reversed(status["segments"]):
                if segment.get("filename") in printed:
                    continue
                printed.add(segment.get("filename"))
                print(f"\n[esp32] stored {segment.get('filename')}  "
                      f"motion={segment.get('motion_detected')}  processing={segment.get('processing') or segment.get('error')}")
    except KeyboardInterrupt:
        print("\n[esp32] stopping, flushing last segment...")
        recorder.stop()
        # Jobs run on a daemon worker; wait so queued processing is not cut off.
        from src.job_manager import _queue
        _queue.join()


if __name__ == "__main__":
    main()
