"""
VISIONFORGE AI
Dynamic Person Attribute Extraction

Reads person crops for any video and generates
appearance attributes for each tracked person.

Current prototype attributes:
    - upper clothing color
    - lower clothing color
    - basic color statistics

Important:
    This is a heuristic prototype, not a trained clothing
    attribute recognition model.
"""

from pathlib import Path
import argparse
import json
import math
import time

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent

CROPS_DIRECTORY = PROJECT_ROOT / "data" / "crops"
ATTRIBUTES_DIRECTORY = PROJECT_ROOT / "data" / "attributes"


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


def get_video_id(video_name):
    return Path(video_name).stem.lower().replace(" ", "_")


def color_distance(pixel, reference):
    return float(np.linalg.norm(pixel - reference))


def classify_color(image_region):
    if image_region is None or image_region.size == 0:
        return "unknown"

    pixels = image_region.reshape(-1, 3).astype(np.float32)

    # Ignore extremely dark and extremely bright outliers
    brightness = pixels.mean(axis=1)

    filtered = pixels[
        (brightness > 20)
        & (brightness < 245)
    ]

    if len(filtered) == 0:
        filtered = pixels

    mean_color = filtered.mean(axis=0)

    distances = {
        name: color_distance(
            mean_color,
            reference,
        )
        for name, reference in COLOR_NAMES.items()
    }

    return min(
        distances,
        key=distances.get,
    )


def extract_attributes_from_crop(crop):
    height, width = crop.shape[:2]

    if height < 20 or width < 10:
        return {
            "upper_body_color": "unknown",
            "lower_body_color": "unknown",
        }

    # Person crop layout:
    #
    # 0% ───────── head
    # 30% ──────── upper body
    # 60% ──────── lower body
    # 100% ─────── feet
    #
    # Use central regions to reduce background influence.

    upper_start = int(height * 0.25)
    upper_end = int(height * 0.55)

    lower_start = int(height * 0.55)
    lower_end = int(height * 0.90)

    horizontal_margin = int(width * 0.15)

    x1 = horizontal_margin
    x2 = width - horizontal_margin

    upper_region = crop[
        upper_start:upper_end,
        x1:x2,
    ]

    lower_region = crop[
        lower_start:lower_end,
        x1:x2,
    ]

    return {
        "upper_body_color": classify_color(
            upper_region
        ),
        "lower_body_color": classify_color(
            lower_region
        ),
    }


def load_crop_metadata(video_id):
    metadata_file = (
        CROPS_DIRECTORY
        / video_id
        / "crop_metadata.json"
    )

    if not metadata_file.exists():
        raise FileNotFoundError(
            f"Crop metadata not found: {metadata_file}"
        )

    with open(
        metadata_file,
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def process_video(video_name):
    video_id = get_video_id(video_name)

    print()
    print("=" * 70)
    print("VISIONFORGE AI — DYNAMIC ATTRIBUTE EXTRACTION")
    print("=" * 70)
    print("Video:", video_name)
    print("Video ID:", video_id)
    print("=" * 70)

    data = load_crop_metadata(video_id)

    crops = data.get("crops", [])

    if not crops:
        print("No person crops found.")
        return

    output_directory = (
        ATTRIBUTES_DIRECTORY / video_id
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = (
        output_directory
        / "person_attributes.json"
    )

    start_time = time.time()

    attributes = []

    track_summary = {}

    processed = 0
    failed = 0

    for crop_info in crops:

        crop_path = Path(
            crop_info["crop_path"]
        )

        # Handle paths stored by another Windows
        # environment by falling back to filename.
        if not crop_path.exists():
            crop_path = (
                CROPS_DIRECTORY
                / video_id
                / crop_path.name
            )

        image = cv2.imread(
            str(crop_path)
        )

        if image is None:
            failed += 1
            continue

        try:
            crop_attributes = (
                extract_attributes_from_crop(
                    image
                )
            )

            record = {
                "video_id": video_id,
                "video_filename": video_name,
                "track_id": crop_info["track_id"],
                "frame_index": crop_info[
                    "frame_index"
                ],
                "timestamp_seconds": crop_info.get(
                    "timestamp_seconds"
                ),
                "crop_path": str(crop_path),
                "attributes": crop_attributes,
            }

            attributes.append(record)

            track_id = int(
                crop_info["track_id"]
            )

            if track_id not in track_summary:
                track_summary[track_id] = {
                    "observations": 0,
                    "upper_colors": [],
                    "lower_colors": [],
                }

            track_summary[track_id][
                "observations"
            ] += 1

            track_summary[track_id][
                "upper_colors"
            ].append(
                crop_attributes[
                    "upper_body_color"
                ]
            )

            track_summary[track_id][
                "lower_colors"
            ].append(
                crop_attributes[
                    "lower_body_color"
                ]
            )

            processed += 1

        except Exception:
            failed += 1

        if processed % 100 == 0:
            print(
                "Processed crops:",
                processed,
                "/",
                len(crops),
            )

    track_profiles = []

    for track_id, summary in sorted(
        track_summary.items()
    ):

        upper_counts = {}

        for color in summary[
            "upper_colors"
        ]:
            upper_counts[color] = (
                upper_counts.get(color, 0) + 1
            )

        lower_counts = {}

        for color in summary[
            "lower_colors"
        ]:
            lower_counts[color] = (
                lower_counts.get(color, 0) + 1
            )

        dominant_upper = max(
            upper_counts,
            key=upper_counts.get,
        )

        dominant_lower = max(
            lower_counts,
            key=lower_counts.get,
        )

        track_profiles.append(
            {
                "video_id": video_id,
                "track_id": track_id,
                "observations": summary[
                    "observations"
                ],
                "appearance": {
                    "upper_body_color": (
                        dominant_upper
                    ),
                    "lower_body_color": (
                        dominant_lower
                    ),
                },
                "color_observations": {
                    "upper": upper_counts,
                    "lower": lower_counts,
                },
            }
        )

    result = {
        "video_id": video_id,
        "video_filename": video_name,
        "total_crops": len(crops),
        "processed_crops": processed,
        "failed_crops": failed,
        "unique_tracks": len(track_profiles),
        "track_profiles": track_profiles,
        "crop_attributes": attributes,
        "generated_at": time.strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
        "processing_time_seconds": round(
            time.time() - start_time,
            2,
        ),
    }

    with open(
        output_file,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            result,
            file,
            indent=2,
        )

    print()
    print("=" * 70)
    print("ATTRIBUTE EXTRACTION COMPLETE")
    print("=" * 70)
    print("Video:", video_name)
    print("Crops:", len(crops))
    print("Processed:", processed)
    print("Failed:", failed)
    print("Unique tracks:", len(track_profiles))
    print("Output:", output_file)
    print(
        "Time:",
        result["processing_time_seconds"],
        "seconds",
    )
    print("=" * 70)


def main():

    parser = argparse.ArgumentParser(
        description=(
            "VISIONFORGE dynamic "
            "attribute extraction"
        )
    )

    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Video filename",
    )

    args = parser.parse_args()

    process_video(
        args.video
    )


if __name__ == "__main__":
    main()