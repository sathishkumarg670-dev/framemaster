"""
VISIONFORGE AI
Dynamic Person Crop Extraction

Takes any CCTV video from data/videos and extracts
person crops using YOLO + BoT-SORT tracking.

Usage:

    python src/extract_dynamic_crops.py --video camera_02.avi

    python src/extract_dynamic_crops.py --all
"""

from pathlib import Path
import argparse
import json
import time

import cv2
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parent.parent

VIDEO_DIRECTORY = PROJECT_ROOT / "data" / "videos"
CROPS_DIRECTORY = PROJECT_ROOT / "data" / "crops"

MODEL_NAME = "yolo11n.pt"

FRAME_INTERVAL = 15

MIN_WIDTH = 40
MIN_HEIGHT = 80


SUPPORTED_EXTENSIONS = {
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".webm",
    ".m4v",
}


def discover_videos():
    VIDEO_DIRECTORY.mkdir(parents=True, exist_ok=True)

    return [
        path
        for path in sorted(VIDEO_DIRECTORY.iterdir())
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]


def get_video_id(video_path):
    return video_path.stem.lower().replace(" ", "_")


def extract_crops(video_path, model):
    video_id = get_video_id(video_path)

    output_directory = CROPS_DIRECTORY / video_id
    output_directory.mkdir(parents=True, exist_ok=True)

    metadata_file = output_directory / "crop_metadata.json"

    print()
    print("=" * 70)
    print("VISIONFORGE AI — DYNAMIC CROP EXTRACTION")
    print("=" * 70)
    print("Video:", video_path.name)
    print("Video ID:", video_id)
    print("Output:", output_directory)
    print("Frame interval:", FRAME_INTERVAL)
    print("=" * 70)

    start_time = time.time()

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(
            f"Unable to open video: {video_path}"
        )

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    fps = cap.get(cv2.CAP_PROP_FPS)

    print("Total frames:", total_frames)
    print("FPS:", round(fps, 2))
    print()

    results = model.track(
        source=str(video_path),
        tracker="botsort.yaml",
        classes=[0],
        conf=0.35,
        device="cpu",
        stream=True,
        persist=True,
        verbose=False,
    )

    metadata = []

    frame_index = 0
    crop_count = 0
    track_ids = set()

    for result in results:

        if frame_index % FRAME_INTERVAL != 0:
            frame_index += 1
            continue

        if result.orig_img is None:
            frame_index += 1
            continue

        frame = result.orig_img

        if (
            result.boxes is None
            or result.boxes.id is None
        ):
            frame_index += 1
            continue

        boxes = result.boxes.xyxy.cpu().tolist()
        ids = result.boxes.id.cpu().tolist()

        for box, track_id in zip(boxes, ids):

            x1, y1, x2, y2 = [
                int(value)
                for value in box
            ]

            x1 = max(0, x1)
            y1 = max(0, y1)

            x2 = min(frame.shape[1], x2)
            y2 = min(frame.shape[0], y2)

            width = x2 - x1
            height = y2 - y1

            if width < MIN_WIDTH:
                continue

            if height < MIN_HEIGHT:
                continue

            track_id = int(track_id)

            track_ids.add(track_id)

            crop = frame[y1:y2, x1:x2]

            if crop.size == 0:
                continue

            filename = (
                f"track_{track_id:04d}"
                f"_frame_{frame_index:06d}.jpg"
            )

            output_path = (
                output_directory / filename
            )

            cv2.imwrite(
                str(output_path),
                crop,
            )

            metadata.append(
                {
                    "video_id": video_id,
                    "video_filename": video_path.name,
                    "track_id": track_id,
                    "frame_index": frame_index,
                    "timestamp_seconds": (
                        frame_index / fps
                        if fps > 0
                        else None
                    ),
                    "bbox": [
                        x1,
                        y1,
                        x2,
                        y2,
                    ],
                    "width": width,
                    "height": height,
                    "crop_path": str(output_path),
                }
            )

            crop_count += 1

        frame_index += 1

        if frame_index % 250 == 0:
            print(
                "Frames:",
                frame_index,
                "/",
                total_frames,
                "| Crops:",
                crop_count,
                "| Tracks:",
                len(track_ids),
            )

    cap.release()

    with open(
        metadata_file,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            {
                "video_id": video_id,
                "video_filename": video_path.name,
                "total_frames": total_frames,
                "fps": fps,
                "frame_interval": FRAME_INTERVAL,
                "total_crops": crop_count,
                "unique_tracks": len(track_ids),
                "tracks": sorted(track_ids),
                "crops": metadata,
            },
            file,
            indent=2,
        )

    elapsed = time.time() - start_time

    print()
    print("=" * 70)
    print("CROP EXTRACTION COMPLETE")
    print("=" * 70)
    print("Video:", video_path.name)
    print("Frames processed:", frame_index)
    print("Person crops:", crop_count)
    print("Unique tracks:", len(track_ids))
    print("Metadata:", metadata_file)
    print("Time:", round(elapsed, 2), "seconds")
    print("=" * 70)

    return {
        "video_id": video_id,
        "video_filename": video_path.name,
        "total_crops": crop_count,
        "unique_tracks": len(track_ids),
        "metadata_file": str(metadata_file),
        "elapsed_seconds": round(elapsed, 2),
    }


def process_selected_video(video_name):

    videos = discover_videos()

    selected = None

    for video in videos:

        if video.name.lower() == video_name.lower():
            selected = video
            break

    if selected is None:

        print()
        print("Video not found:", video_name)
        print()
        print("Available videos:")

        for video in videos:
            print("-", video.name)

        return

    print()
    print("Selected:", selected.name)

    model = YOLO(MODEL_NAME)

    extract_crops(
        selected,
        model,
    )


def process_all_videos():

    videos = discover_videos()

    if not videos:

        print("No CCTV videos found.")

        return

    print()
    print("Videos to process:", len(videos))

    for video in videos:
        print("-", video.name)

    model = YOLO(MODEL_NAME)

    results = []

    for index, video in enumerate(
        videos,
        start=1,
    ):

        print()
        print(
            f"PROCESSING {index}/{len(videos)}"
        )

        result = extract_crops(
            video,
            model,
        )

        results.append(result)

    summary_file = (
        PROJECT_ROOT
        / "data"
        / "crop_processing_summary.json"
    )

    with open(
        summary_file,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            {
                "total_videos": len(results),
                "videos": results,
            },
            file,
            indent=2,
        )

    print()
    print("All crop extraction completed.")
    print("Summary:", summary_file)


def main():

    parser = argparse.ArgumentParser(
        description=(
            "VISIONFORGE dynamic person "
            "crop extraction"
        )
    )

    parser.add_argument(
        "--video",
        type=str,
        help="Extract crops from one video",
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Extract crops from all videos",
    )

    args = parser.parse_args()

    if args.video:

        process_selected_video(
            args.video
        )

    elif args.all:

        process_all_videos()

    else:

        print()
        print(
            "Use --video <filename> "
            "or --all"
        )


if __name__ == "__main__":
    main()