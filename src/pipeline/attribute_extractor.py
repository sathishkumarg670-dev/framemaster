"""
VISIONFORGE AI - Exact Crop Attribute Extractor
Extracts clothing color heuristic attributes strictly from generated crop files,
aggregating attributes on a per-track basis.
"""

from pathlib import Path
from typing import Dict, List
import json
import cv2
import numpy as np

from src.pipeline.models import (
    CropMetadataCollection,
    TrackAttributes,
)

COLOR_NAMES = {
    "black": np.array([40, 40, 40]),
    "white": np.array([220, 220, 220]),
    "gray": np.array([128, 128, 128]),
    "red": np.array([180, 50, 50]),
    "green": np.array([50, 150, 70]),
    "blue": np.array([60, 100, 190]),
    "yellow": np.array([210, 190, 60]),
    "orange": np.array([210, 130, 50]),
    "brown": np.array([120, 80, 50]),
    "purple": np.array([130, 70, 150]),
    "pink": np.array([200, 100, 150]),
}


def color_distance(pixel: np.ndarray, reference: np.ndarray) -> float:
    return float(np.linalg.norm(pixel - reference))


def classify_color(image_region: np.ndarray) -> str:
    if image_region is None or image_region.size == 0:
        return "unknown"

    pixels = image_region.reshape(-1, 3).astype(np.float32)
    brightness = pixels.mean(axis=1)
    filtered = pixels[(brightness > 20) & (brightness < 245)]

    if len(filtered) == 0:
        filtered = pixels

    mean_color = filtered.mean(axis=0)
    distances = {
        name: color_distance(mean_color, reference)
        for name, reference in COLOR_NAMES.items()
    }
    return min(distances, key=distances.get)


def extract_attributes_from_crop_img(crop_bgr: np.ndarray) -> Dict[str, str]:
    h, w = crop_bgr.shape[:2]
    if h < 20 or w < 10:
        return {"upper_body_color": "unknown", "lower_body_color": "unknown"}

    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)

    upper_y1 = int(h * 0.18)
    upper_y2 = int(h * 0.55)
    upper_x1 = int(w * 0.15)
    upper_x2 = int(w * 0.85)

    lower_y1 = int(h * 0.55)
    lower_y2 = int(h * 0.90)
    lower_x1 = int(w * 0.15)
    lower_x2 = int(w * 0.85)

    upper_region = crop_rgb[upper_y1:upper_y2, upper_x1:upper_x2]
    lower_region = crop_rgb[lower_y1:lower_y2, lower_x1:lower_x2]

    return {
        "upper_body_color": classify_color(upper_region),
        "lower_body_color": classify_color(lower_region),
    }


def extract_attributes_for_crops(
    crop_collection: CropMetadataCollection,
    output_dir: Path,
) -> Dict[int, TrackAttributes]:
    """
    Extracts attributes for all crops in the collection and aggregates per track.
    Saves person_attributes.json in output_dir.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    track_upper_colors: Dict[int, List[str]] = {}
    track_lower_colors: Dict[int, List[str]] = {}

    print(f"\n[VISIONFORGE AI] Extracting attributes for {crop_collection.total_crops} crops")

    for crop_rec in crop_collection.crops:
        crop_path = Path(crop_rec.crop_path)
        if not crop_path.exists():
            continue

        crop_bgr = cv2.imread(str(crop_path))
        if crop_bgr is None:
            continue

        attrs = extract_attributes_from_crop_img(crop_bgr)
        track_id = crop_rec.track_id

        track_upper_colors.setdefault(track_id, []).append(attrs["upper_body_color"])
        track_lower_colors.setdefault(track_id, []).append(attrs["lower_body_color"])

    track_attributes: Dict[int, TrackAttributes] = {}

    for track_id in crop_collection.unique_tracks:
        uppers = track_upper_colors.get(track_id, ["unknown"])
        lowers = track_lower_colors.get(track_id, ["unknown"])

        # Mode calculation for dominant color
        dominant_upper = max(set(uppers), key=uppers.count) if uppers else "unknown"
        dominant_lower = max(set(lowers), key=lowers.count) if lowers else "unknown"

        # Color frequency distribution
        total = len(uppers)
        dist = {}
        for c in set(uppers + lowers):
            dist[c] = round((uppers.count(c) + lowers.count(c)) / (2 * total), 3)

        track_attributes[track_id] = TrackAttributes(
            track_id=track_id,
            upper_body_color=dominant_upper,
            lower_body_color=dominant_lower,
            color_distribution=dist,
            sample_crops=total,
        )

    output_file = output_dir / "person_attributes.json"
    data_to_save = {
        "video_id": crop_collection.video_id,
        "total_tracks": len(track_attributes),
        "tracks": {str(k): v.model_dump() for k, v in track_attributes.items()},
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data_to_save, f, indent=2)

    print(f"Attributes extraction complete. Saved {len(track_attributes)} track profiles to {output_file.name}")
    return track_attributes
