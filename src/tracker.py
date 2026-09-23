"""
VISIONFORGE AI - Single-Pass Tracker
Performs person detection and BoT-SORT tracking ONCE on a video stream,
emitting structured frame-level observations.
"""

from pathlib import Path
import time
import cv2
from ultralytics import YOLO

from src.pipeline.models import Observation, VideoTrackingResult


def get_video_id(video_path: Path) -> str:
    return video_path.stem.lower().replace(" ", "_")


def run_single_pass_tracking(
    video_path: Path,
    model_path: str = "yolo11n.pt",
    conf_thresh: float = 0.35,
    frame_stride: int = 1,
    max_frames: int = None,
    device: str = "cpu",
    output_observations_path: Path = None,
    log_interval: int = 250,
) -> VideoTrackingResult:
    """
    Executes detection and tracking in a single pass.
    Saves and returns VideoTrackingResult with frame observations.
    """
    video_path = Path(video_path)
    video_id = get_video_id(video_path)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0
    cap.release()

    model = YOLO(model_path)

    print(f"\n[VISIONFORGE AI] Starting single-pass tracking: {video_path.name}")
    print(f"Total Frames: {total_frames} | FPS: {fps:.2f} | Device: {device}")

    start_time = time.time()

    # Track with persist=True for temporal consistency
    results = model.track(
        source=str(video_path),
        tracker="botsort.yaml",
        classes=[0],  # Person
        conf=conf_thresh,
        device=device,
        stream=True,
        persist=True,
        verbose=False,
    )

    observations = []
    unique_tracks = set()
    frame_count = 0

    for result in results:
        frame_count += 1

        if frame_stride > 1 and (frame_count - 1) % frame_stride != 0:
            continue

        if max_frames and frame_count > max_frames:
            break

        timestamp = (frame_count - 1) / fps

        if result.boxes is not None and result.boxes.id is not None:
            boxes = result.boxes.xyxy.cpu().tolist()
            ids = result.boxes.id.cpu().tolist()
            confs = result.boxes.conf.cpu().tolist() if result.boxes.conf is not None else [1.0] * len(ids)

            for box, track_id, conf in zip(boxes, ids, confs):
                t_id = int(track_id)
                unique_tracks.add(t_id)

                obs = Observation(
                    frame_index=frame_count - 1,
                    timestamp_seconds=round(timestamp, 3),
                    track_id=t_id,
                    bbox=[int(b) for b in box],
                    confidence=round(float(conf), 4),
                    class_id=0,
                )
                observations.append(obs)

        if frame_count % log_interval == 0:
            print(f"Tracking frame {frame_count}/{total_frames} | Unique tracks: {len(unique_tracks)} | Observations: {len(observations)}")

    elapsed = time.time() - start_time
    print(f"Tracking complete in {elapsed:.2f}s. Processed {frame_count} frames, found {len(unique_tracks)} tracks, {len(observations)} observations.")

    tracking_result = VideoTrackingResult(
        video_id=video_id,
        video_filename=video_path.name,
        fps=round(fps, 2),
        total_frames=total_frames,
        processed_frames=frame_count,
        unique_tracks=sorted(list(unique_tracks)),
        total_observations=len(observations),
        observations=observations,
    )

    if output_observations_path:
        tracking_result.save_json(Path(output_observations_path))

    return tracking_result
