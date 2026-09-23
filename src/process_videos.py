"""
VISIONFORGE AI
Dynamic CCTV Video Processing Engine

Usage:

    python src/process_videos.py --list

    python src/process_videos.py --video camera_02.avi

    python src/process_videos.py --all
"""

from pathlib import Path
import argparse
import json
import time

from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parent.parent

VIDEO_DIRECTORY = PROJECT_ROOT / "data" / "videos"
RUNS_DIRECTORY = PROJECT_ROOT / "runs"

SUPPORTED_EXTENSIONS = {
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".webm",
    ".m4v",
}

MODEL_NAME = "yolo11n.pt"


def discover_videos():
    VIDEO_DIRECTORY.mkdir(parents=True, exist_ok=True)

    videos = []

    for path in sorted(VIDEO_DIRECTORY.iterdir()):
        if not path.is_file():
            continue

        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        videos.append(path)

    return videos


def get_video_id(video_path):
    return video_path.stem.lower().replace(" ", "_")


def prepare_directories(video_id):
    paths = {
        "detection": RUNS_DIRECTORY / "detection" / video_id,
        "tracking": RUNS_DIRECTORY / "tracking" / video_id,
        "crops": PROJECT_ROOT / "data" / "crops" / video_id,
        "attributes": PROJECT_ROOT / "data" / "attributes" / video_id,
        "reid": PROJECT_ROOT / "data" / "reid" / video_id,
    }

    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)

    return paths


def run_detection(video_path, video_id, model):
    output_directory = RUNS_DIRECTORY / "detection"
    output_directory.mkdir(parents=True, exist_ok=True)

    print()
    print("=" * 60)
    print("PERSON DETECTION")
    print("=" * 60)
    print("Video:", video_path.name)
    print("Model:", MODEL_NAME)
    print("Device: CPU")
    print()

    start_time = time.time()

    results = model.predict(
        source=str(video_path),
        project=str(output_directory),
        name=video_id,
        exist_ok=True,
        classes=[0],
        conf=0.35,
        device="cpu",
        stream=True,
        verbose=False,
        save=True,
    )

    frame_count = 0

    for _ in results:
        frame_count += 1

        if frame_count % 250 == 0:
            print("Detection frames:", frame_count)

    elapsed = time.time() - start_time

    print()
    print("Detection complete.")
    print("Frames processed:", frame_count)
    print("Time:", round(elapsed, 2), "seconds")

    return {
        "frames_processed": frame_count,
        "elapsed_seconds": round(elapsed, 2),
        "output_directory": str(output_directory / video_id),
    }


def run_tracking(video_path, video_id, model):
    output_directory = RUNS_DIRECTORY / "tracking"
    output_directory.mkdir(parents=True, exist_ok=True)

    print()
    print("=" * 60)
    print("PERSON TRACKING")
    print("=" * 60)
    print("Video:", video_path.name)
    print("Tracker: BoT-SORT")
    print("Device: CPU")
    print()

    start_time = time.time()

    results = model.track(
        source=str(video_path),
        tracker="botsort.yaml",
        project=str(output_directory),
        name=video_id,
        exist_ok=True,
        classes=[0],
        conf=0.35,
        device="cpu",
        stream=True,
        verbose=False,
        save=True,
    )

    frame_count = 0
    track_ids = set()

    for result in results:
        frame_count += 1

        if result.boxes is not None and result.boxes.id is not None:
            ids = result.boxes.id.cpu().tolist()

            for track_id in ids:
                track_ids.add(int(track_id))

        if frame_count % 250 == 0:
            print(
                "Tracking frames:",
                frame_count,
                "| Unique tracks:",
                len(track_ids),
            )

    elapsed = time.time() - start_time

    print()
    print("Tracking complete.")
    print("Frames processed:", frame_count)
    print("Unique tracks:", len(track_ids))
    print("Time:", round(elapsed, 2), "seconds")

    return {
        "frames_processed": frame_count,
        "unique_tracks": len(track_ids),
        "elapsed_seconds": round(elapsed, 2),
        "output_directory": str(output_directory / video_id),
    }


def process_video(video_path, model):
    video_id = get_video_id(video_path)

    print()
    print("#" * 70)
    print("VISIONFORGE AI")
    print("DYNAMIC CCTV PROCESSOR")
    print("#" * 70)
    print("Video:", video_path.name)
    print("Video ID:", video_id)
    print("Source:", video_path)
    print("#" * 70)

    paths = prepare_directories(video_id)

    result = {
        "video_id": video_id,
        "filename": video_path.name,
        "source": str(video_path),
        "status": "processing",
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "detection": None,
        "tracking": None,
    }

    try:
        detection_result = run_detection(
            video_path,
            video_id,
            model,
        )

        result["detection"] = detection_result

        tracking_result = run_tracking(
            video_path,
            video_id,
            model,
        )

        result["tracking"] = tracking_result

        result["status"] = "completed"

    except Exception as error:
        result["status"] = "failed"
        result["error"] = str(error)

        print()
        print("ERROR:")
        print(error)

    result["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")

    metadata_file = paths["reid"] / "processing_metadata.json"

    with open(metadata_file, "w", encoding="utf-8") as file:
        json.dump(
            result,
            file,
            indent=2,
        )

    return result


def process_selected_video(video_name):
    videos = discover_videos()

    selected_video = None

    for video in videos:
        if video.name.lower() == video_name.lower():
            selected_video = video
            break

    if selected_video is None:
        print()
        print("Video not found:", video_name)
        print()
        print("Available videos:")

        for video in videos:
            print("-", video.name)

        return

    print()
    print("Selected video:", selected_video.name)

    model = YOLO(MODEL_NAME)

    result = process_video(
        selected_video,
        model,
    )

    print()
    print("=" * 70)
    print("FINAL RESULT")
    print("=" * 70)
    print("Video:", result["filename"])
    print("Status:", result["status"])

    if result["status"] == "completed":
        print(
            "Unique tracks:",
            result["tracking"]["unique_tracks"],
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

    print()

    model = YOLO(MODEL_NAME)

    results = []

    for index, video in enumerate(videos, start=1):
        print()
        print("=" * 70)
        print(
            f"PROCESSING VIDEO {index}/{len(videos)}"
        )
        print("=" * 70)

        result = process_video(
            video,
            model,
        )

        results.append(result)

    summary_file = (
        PROJECT_ROOT
        / "data"
        / "processing_summary.json"
    )

    with open(summary_file, "w", encoding="utf-8") as file:
        json.dump(
            {
                "total_videos": len(videos),
                "completed": sum(
                    1
                    for result in results
                    if result["status"] == "completed"
                ),
                "failed": sum(
                    1
                    for result in results
                    if result["status"] == "failed"
                ),
                "videos": results,
            },
            file,
            indent=2,
        )

    print()
    print("=" * 70)
    print("PROCESSING SUMMARY")
    print("=" * 70)

    for result in results:
        print(
            result["filename"],
            "->",
            result["status"],
        )

    print()
    print("Summary:", summary_file)


def main():
    parser = argparse.ArgumentParser(
        description="VISIONFORGE dynamic CCTV processor"
    )

    parser.add_argument(
        "--list",
        action="store_true",
        help="List available CCTV videos",
    )

    parser.add_argument(
        "--video",
        type=str,
        help="Process one specific CCTV video",
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all CCTV videos",
    )

    args = parser.parse_args()

    if args.list:
        videos = discover_videos()

        print()
        print("VISIONFORGE CCTV DATABASE")
        print("=" * 50)
        print("Videos discovered:", len(videos))
        print()

        for index, video in enumerate(videos, start=1):
            print(f"{index}. {video.name}")

        return

    if args.video:
        process_selected_video(args.video)
        return

    if args.all:
        process_all_videos()
        return

    print()
    print("VISIONFORGE AI — Dynamic CCTV Processor")
    print()
    print("Choose an operation:")
    print()
    print("  --list")
    print("  --video <filename>")
    print("  --all")
    print()


if __name__ == "__main__":
    main()