"""
VISIONFORGE AI - Observation-Driven Crop Extractor
Reads existing VideoTrackingResult observations and extracts person crops
directly from video frames without re-running any tracker.
"""

from pathlib import Path
from typing import Dict, List
import time
import cv2

from src.pipeline.models import (
    VideoTrackingResult,
    Observation,
    CropRecord,
    CropMetadataCollection,
)


def extract_crops_from_observations(
    video_path: Path,
    tracking_result: VideoTrackingResult,
    output_dir: Path,
    frame_interval: int = 15,
    min_width: int = 40,
    min_height: int = 80,
    max_crops_per_track: int = 25,
) -> CropMetadataCollection:
    """
    Extracts person crops purely from recorded observations.
    Zero detection or tracking models are executed here.
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Group observations by frame_index for single-pass sequential video reading
    obs_by_frame: Dict[int, List[Observation]] = {}
    for obs in tracking_result.observations:
        # Sample by frame interval to avoid excessive redundant frames per track
        if obs.frame_index % frame_interval == 0:
            obs_by_frame.setdefault(obs.frame_index, []).append(obs)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    start_time = time.time()
    extracted_crops: List[CropRecord] = []
    crops_per_track: Dict[int, int] = {}
    unique_tracks = set()

    current_frame_idx = 0
    total_frames = tracking_result.total_frames

    print(f"\n[VISIONFORGE AI] Extracting crops from observations: {video_path.name}")
    print(f"Target frames with observations: {len(obs_by_frame)}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if current_frame_idx in obs_by_frame:
            observations = obs_by_frame[current_frame_idx]
            frame_h, frame_w = frame.shape[:2]

            for obs in observations:
                track_id = obs.track_id

                if crops_per_track.get(track_id, 0) >= max_crops_per_track:
                    continue

                x1, y1, x2, y2 = obs.bbox
                x1 = max(0, min(frame_w, x1))
                y1 = max(0, min(frame_h, y1))
                x2 = max(0, min(frame_w, x2))
                y2 = max(0, min(frame_h, y2))

                w = x2 - x1
                h = y2 - y1

                if w < min_width or h < min_height:
                    continue

                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                crop_filename = f"track_{track_id:04d}_frame_{current_frame_idx:06d}.jpg"
                crop_path = output_dir / crop_filename
                cv2.imwrite(str(crop_path), crop)

                crop_record = CropRecord(
                    crop_id=f"{tracking_result.video_id}_{track_id:04d}_{current_frame_idx:06d}",
                    crop_filename=crop_filename,
                    crop_path=str(crop_path),
                    video_id=tracking_result.video_id,
                    track_id=track_id,
                    frame_index=current_frame_idx,
                    timestamp_seconds=obs.timestamp_seconds,
                    bbox=[x1, y1, x2, y2],
                    width=w,
                    height=h,
                )
                extracted_crops.append(crop_record)
                crops_per_track[track_id] = crops_per_track.get(track_id, 0) + 1
                unique_tracks.add(track_id)

        current_frame_idx += 1
        if current_frame_idx > total_frames:
            break

    cap.release()
    elapsed = time.time() - start_time

    print(f"Crops extracted: {len(extracted_crops)} across {len(unique_tracks)} tracks in {elapsed:.2f}s.")

    collection = CropMetadataCollection(
        video_id=tracking_result.video_id,
        video_filename=video_path.name,
        total_crops=len(extracted_crops),
        unique_tracks=sorted(list(unique_tracks)),
        crops=extracted_crops,
    )

    collection.save_json(output_dir / "crop_metadata.json")
    return collection
