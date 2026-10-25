from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

from .tracker import run_single_pass_tracking
from .crop_extractor import extract_crops_from_observations
from .attribute_extractor import extract_attributes_for_crops
from .reid_extractor import generate_track_embeddings
from .models import TrackProfile


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

DATA_DIR = PROJECT_ROOT / "data"
VIDEO_DIR = DATA_DIR / "videos"

UNIFIED_DIR = DATA_DIR / "unified"

TRACKING_DIR = UNIFIED_DIR / "tracking"
CROPS_DIR = UNIFIED_DIR / "crops"
ATTRIBUTES_DIR = UNIFIED_DIR / "attributes"
REID_DIR = UNIFIED_DIR / "reid"
PROFILES_DIR = UNIFIED_DIR / "profiles"


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}


def ensure_directories():
    for directory in [
        UNIFIED_DIR,
        TRACKING_DIR,
        CROPS_DIR,
        ATTRIBUTES_DIR,
        REID_DIR,
        PROFILES_DIR,
    ]:
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )


def clean_unified_outputs():
    if UNIFIED_DIR.exists():
        print()
        print("Cleaning unified M11 outputs...")
        shutil.rmtree(UNIFIED_DIR)

    ensure_directories()

    print("Unified outputs cleaned.")
    print()


def discover_videos():
    """Discover every supported CCTV video in the database."""
    if not VIDEO_DIR.exists():
        return []
    return sorted(
        [
            path for path in VIDEO_DIR.iterdir()
            if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
        ],
        key=lambda path: path.name.lower(),
    )


def resolve_video(video_name: str):
    requested = Path(video_name)

    if requested.exists():
        return requested.resolve()

    direct = VIDEO_DIR / video_name

    if direct.exists():
        return direct.resolve()

    for video in discover_videos():
        if video.name.lower() == video_name.lower():
            return video.resolve()

    raise FileNotFoundError(
        f"Video not found: {video_name}\n"
        f"Expected inside: {VIDEO_DIR}"
    )


def save_json(path: Path, data):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
            indent=2,
            ensure_ascii=False,
        )


def print_dataset():
    videos = discover_videos()

    print()
    print("=" * 70)
    print("VISIONFORGE AI — ACTIVE DATASET")
    print("=" * 70)

    if not videos:
        print("No CCTV videos found in the video database.")
        print(f"Expected directory: {VIDEO_DIR}")
        print("=" * 70)
        print()
        return

    for index, video in enumerate(
        videos,
        start=1,
    ):
        size_mb = (
            video.stat().st_size
            / (1024 * 1024)
        )

        print(
            f"{index}. {video.name} "
            f"({size_mb:.2f} MB)"
        )

    print()
    print(f"Video count: {len(videos)}")
    print("=" * 70)
    print()


def middle_crop_per_track(crop_metadata):
    """
    Representative crop per track when Re-ID is skipped.

    Same choice as the Re-ID stage: the middle crop of the track.
    """

    crops_by_track = {}

    for crop in crop_metadata.crops:
        crops_by_track.setdefault(
            crop.track_id,
            [],
        ).append(crop)

    representative_crops = {}

    for track_id, crops in crops_by_track.items():
        crops.sort(
            key=lambda item: item.frame_index
        )
        representative_crops[track_id] = (
            crops[len(crops) // 2].crop_path
        )

    return representative_crops


def build_canonical_profiles(
    video_id,
    tracking_result,
    crop_metadata,
    attributes,
    embeddings,
    representative_crops,
):
    """
    Combine tracking, crop, attribute and Re-ID data
    into canonical TrackProfile objects.
    """

    observations_by_track = {}

    for observation in tracking_result.observations:
        observations_by_track.setdefault(
            observation.track_id,
            [],
        ).append(observation)

    crop_count_by_track = {}

    for crop in crop_metadata.crops:
        crop_count_by_track[crop.track_id] = (
            crop_count_by_track.get(
                crop.track_id,
                0,
            )
            + 1
        )

    profiles = []

    for track_id, observations in observations_by_track.items():

        if not observations:
            continue

        observations.sort(
            key=lambda item: item.frame_index
        )

        first_observation = observations[0]
        last_observation = observations[-1]

        first_frame = (
            first_observation.frame_index
        )

        last_frame = (
            last_observation.frame_index
        )

        first_time = (
            first_observation.timestamp_seconds
        )

        last_time = (
            last_observation.timestamp_seconds
        )

        duration = max(
            0.0,
            last_time - first_time,
        )

        track_attributes = None

        if isinstance(attributes, dict):
            track_attributes = attributes.get(
                track_id
            )

        track_embedding = None

        if isinstance(embeddings, dict):
            track_embedding = embeddings.get(
                track_id
            )

        representative_crop = None

        if isinstance(
            representative_crops,
            dict,
        ):
            representative_crop = (
                representative_crops.get(
                    track_id
                )
            )

        embedding_dimension = 512

        if track_embedding:
            embedding_dimension = len(
                track_embedding
            )

        profile = TrackProfile(
            video_id=video_id,
            track_id=track_id,
            first_seen_frame=first_frame,
            last_seen_frame=last_frame,
            first_seen_sec=first_time,
            last_seen_sec=last_time,
            duration_sec=duration,
            total_observations=len(
                observations
            ),
            total_crops=crop_count_by_track.get(
                track_id,
                0,
            ),
            attributes=track_attributes,
            embedding=track_embedding,
            embedding_dimension=embedding_dimension,
            representative_crop=representative_crop,
        )

        profile.update_readiness()
        profiles.append(profile)

    return profiles


def process_video(
    video_path: Path,
    device="cpu",
    max_frames=None,
    frame_stride=1,
):
    video_path = Path(video_path)

    video_id = (
        video_path.stem
        .lower()
        .replace(" ", "_")
    )

    print()
    print("=" * 70)
    print(
        f"M11 PROCESSING: {video_path.name}"
    )
    print("=" * 70)

    print(f"Video ID: {video_id}")
    print(f"Device: {device}")
    print(f"Frame stride: {frame_stride}")
    print(f"Max frames: {max_frames}")
    print()

    start_time = time.time()

    tracking_dir = TRACKING_DIR / video_id
    crops_dir = CROPS_DIR / video_id
    attributes_dir = ATTRIBUTES_DIR / video_id
    reid_dir = REID_DIR / video_id
    profiles_dir = PROFILES_DIR / video_id

    for directory in [
        tracking_dir,
        crops_dir,
        attributes_dir,
        reid_dir,
        profiles_dir,
    ]:
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    summary = {
        "video_id": video_id,
        "video_filename": video_path.name,
        "video_path": str(video_path),
        "status": "started",
        "stages": {},
    }

    try:

        # ========================================================
        # M11.1 TRACKING
        # ========================================================

        print(
            "[M11.1] Running person detection + tracking..."
        )

        tracking_json = (
            tracking_dir
            / "tracking_observations.json"
        )

        tracking_start = time.time()

        tracking_result = (
            run_single_pass_tracking(
                video_path=video_path,
                device=device,
                frame_stride=frame_stride,
                max_frames=max_frames,
                output_observations_path=tracking_json,
            )
        )

        tracking_seconds = (
            time.time() - tracking_start
        )

        summary["stages"]["tracking"] = {
            "status": "completed",
            "seconds": round(
                tracking_seconds,
                2,
            ),
            "total_frames": (
                tracking_result.total_frames
            ),
            "processed_frames": (
                tracking_result.processed_frames
            ),
            "unique_tracks": len(
                tracking_result.unique_tracks
            ),
            "total_observations": (
                tracking_result.total_observations
            ),
        }

        print()
        print("Tracking complete:")
        print(
            f"  Frames: "
            f"{tracking_result.processed_frames}"
        )
        print(
            f"  Tracks: "
            f"{len(tracking_result.unique_tracks)}"
        )
        print(
            f"  Observations: "
            f"{tracking_result.total_observations}"
        )
        print()

        # ========================================================
        # M11.2 CROPS
        # ========================================================

        print(
            "[M11.2] Extracting exact person crops..."
        )

        crop_start = time.time()

        crop_metadata = (
            extract_crops_from_observations(
                video_path=video_path,
                tracking_result=tracking_result,
                output_dir=crops_dir,
            )
        )

        crop_seconds = (
            time.time() - crop_start
        )

        summary["stages"]["crops"] = {
            "status": "completed",
            "seconds": round(
                crop_seconds,
                2,
            ),
            "total_crops": (
                crop_metadata.total_crops
            ),
            "unique_tracks": (
                crop_metadata.unique_tracks
            ),
        }

        print()
        print("Crop extraction complete:")
        print(
            f"  Crops: "
            f"{crop_metadata.total_crops}"
        )
        print(
            f"  Tracks: "
            f"{len(crop_metadata.unique_tracks)}"
        )
        print()

        # ========================================================
        # M11.3 ATTRIBUTES
        # ========================================================

        print(
            "[M11.3] Extracting appearance attributes..."
        )

        attribute_start = time.time()

        attributes = (
            extract_attributes_for_crops(
                crop_collection=crop_metadata,
                output_dir=attributes_dir,
            )
        )

        attribute_seconds = (
            time.time() - attribute_start
        )

        summary["stages"]["attributes"] = {
            "status": "completed",
            "seconds": round(
                attribute_seconds,
                2,
            ),
            "track_count": len(
                attributes
            ),
        }

        print()
        print(
            "Attribute extraction complete:"
        )
        print(
            f"  Track profiles: "
            f"{len(attributes)}"
        )
        print()

        # ========================================================
        # M11.4 RE-ID
        # ========================================================

        print(
            "[M11.4] Generating Market1501 OSNet embeddings..."
        )

        checkpoint = (
            PROJECT_ROOT
            / "models"
            / "reid"
            / "osnet_x1_0_market1501.pt"
        )

        if not checkpoint.exists():

            # Without the checkpoint, still save TRACK_ONLY profiles so the
            # video's people and their attributes reach investigator search.
            print(
                "Re-ID checkpoint not found, skipping embeddings:\n"
                f"  {checkpoint}"
            )

            embeddings = {}

            representative_crops = middle_crop_per_track(
                crop_metadata
            )

            summary["stages"]["reid"] = {
                "status": "skipped",
                "reason": (
                    "checkpoint not found: "
                    f"{checkpoint}"
                ),
            }

        else:

            reid_json = (
                reid_dir
                / "market1501_track_embeddings.json"
            )

            reid_start = time.time()

            embeddings, representative_crops = (
                generate_track_embeddings(
                    crop_metadata=crop_metadata,
                    checkpoint_path=checkpoint,
                    output_json=reid_json,
                    device=device,
                    batch_size=16,
                )
            )

            reid_seconds = (
                time.time() - reid_start
            )

            summary["stages"]["reid"] = {
                "status": "completed",
                "seconds": round(
                    reid_seconds,
                    2,
                ),
                "track_count": len(
                    embeddings
                ),
                "embedding_dimension": 512,
            }

            print()
            print("Re-ID extraction complete:")
            print(
                f"  Track embeddings: "
                f"{len(embeddings)}"
            )
            print()

        # ========================================================
        # M11.5 CANONICAL PROFILES
        # ========================================================

        print(
            "[M11.5] Building canonical track profiles..."
        )

        profile_start = time.time()

        profiles = build_canonical_profiles(
            video_id=video_id,
            tracking_result=tracking_result,
            crop_metadata=crop_metadata,
            attributes=attributes,
            embeddings=embeddings,
            representative_crops=representative_crops,
        )

        profiles_json = (
            profiles_dir
            / "canonical_track_profiles.json"
        )

        save_json(
            profiles_json,
            [
                profile.model_dump()
                for profile in profiles
            ],
        )

        profile_seconds = (
            time.time() - profile_start
        )

        summary["stages"]["profiles"] = {
            "status": "completed",
            "seconds": round(
                profile_seconds,
                2,
            ),
            "profile_count": len(
                profiles
            ),
            "output": str(
                profiles_json
            ),
        }

        print()
        print(
            "Canonical profiles complete:"
        )
        print(
            f"  Profiles: "
            f"{len(profiles)}"
        )
        print(
            f"  Saved: "
            f"{profiles_json}"
        )
        print()

        # ========================================================
        # FINAL VIDEO SUMMARY
        # ========================================================

        total_seconds = (
            time.time() - start_time
        )

        summary["status"] = "completed"

        summary["total_seconds"] = round(
            total_seconds,
            2,
        )

        summary_path = (
            profiles_dir
            / "processing_summary.json"
        )

        save_json(
            summary_path,
            summary,
        )

        print("=" * 70)
        print(
            f"COMPLETED: {video_path.name}"
        )
        print(
            f"Tracks: {len(profiles)}"
        )
        print(
            f"Total time: "
            f"{total_seconds:.2f} seconds"
        )
        print("=" * 70)
        print()

        return summary

    except Exception as error:

        total_seconds = (
            time.time() - start_time
        )

        summary["status"] = "failed"
        summary["error"] = str(error)
        summary["total_seconds"] = round(
            total_seconds,
            2,
        )

        error_path = (
            profiles_dir
            / "processing_summary.json"
        )

        save_json(
            error_path,
            summary,
        )

        print()
        print("=" * 70)
        print(
            f"FAILED: {video_path.name}"
        )
        print(
            f"ERROR: {error}"
        )
        print("=" * 70)
        print()

        return summary


def build_dataset_summary(
    summaries,
):
    completed = [
        item
        for item in summaries
        if item.get("status")
        == "completed"
    ]

    failed = [
        item
        for item in summaries
        if item.get("status")
        == "failed"
    ]

    total_tracks = 0
    total_observations = 0
    total_crops = 0
    total_profiles = 0
    total_seconds = 0.0

    for summary in completed:

        tracking = (
            summary
            .get("stages", {})
            .get("tracking", {})
        )

        crops = (
            summary
            .get("stages", {})
            .get("crops", {})
        )

        profiles = (
            summary
            .get("stages", {})
            .get("profiles", {})
        )

        total_tracks += tracking.get(
            "unique_tracks",
            0,
        )

        total_observations += tracking.get(
            "total_observations",
            0,
        )

        total_crops += crops.get(
            "total_crops",
            0,
        )

        total_profiles += profiles.get(
            "profile_count",
            0,
        )

        total_seconds += summary.get(
            "total_seconds",
            0.0,
        )

    return {
        "project": "VISIONFORGE AI",
        "milestone": "M11",
        "status": (
            "completed"
            if not failed
            else "completed_with_failures"
        ),
        "dataset_video_count": len(
            summaries
        ),
        "completed_video_count": len(
            completed
        ),
        "failed_video_count": len(
            failed
        ),
        "total_tracks": total_tracks,
        "total_observations": (
            total_observations
        ),
        "total_crops": total_crops,
        "total_canonical_profiles": (
            total_profiles
        ),
        "total_processing_seconds": round(
            total_seconds,
            2,
        ),
        "videos": summaries,
    }


def process_all(
    device="cpu",
    max_frames=None,
    frame_stride=1,
    clean=False,
):

    ensure_directories()

    if clean:
        clean_unified_outputs()

    videos = discover_videos()

    print_dataset()

    if not videos:
        print(
            "No active dataset videos available."
        )
        return

    summaries = []

    dataset_start = time.time()

    for index, video in enumerate(
        videos,
        start=1,
    ):

        print()
        print(
            f"DATASET VIDEO "
            f"{index}/{len(videos)}"
        )

        print(
            f"Starting: {video.name}"
        )

        result = process_video(
            video_path=video,
            device=device,
            max_frames=max_frames,
            frame_stride=frame_stride,
        )

        summaries.append(result)

    dataset_summary = (
        build_dataset_summary(
            summaries
        )
    )

    dataset_summary[
        "dataset_total_seconds"
    ] = round(
        time.time()
        - dataset_start,
        2,
    )

    summary_path = (
        UNIFIED_DIR
        / "dataset_processing_summary.json"
    )

    save_json(
        summary_path,
        dataset_summary,
    )

    print()
    print("=" * 70)
    print(
        "M11 DATASET PROCESSING COMPLETE"
    )
    print("=" * 70)

    print(
        f"Videos processed: "
        f"{dataset_summary['completed_video_count']}/"
        f"{dataset_summary['dataset_video_count']}"
    )

    print(
        f"Failed videos: "
        f"{dataset_summary['failed_video_count']}"
    )

    print(
        f"Total tracks: "
        f"{dataset_summary['total_tracks']}"
    )

    print(
        f"Total observations: "
        f"{dataset_summary['total_observations']}"
    )

    print(
        f"Total crops: "
        f"{dataset_summary['total_crops']}"
    )

    print(
        f"Canonical profiles: "
        f"{dataset_summary['total_canonical_profiles']}"
    )

    print(
        f"Dataset time: "
        f"{dataset_summary['dataset_total_seconds']:.2f} seconds"
    )

    print()
    print(
        "Summary saved to:"
    )
    print(summary_path)
    print("=" * 70)
    print()


def main():

    parser = argparse.ArgumentParser(
        description=(
            "VISIONFORGE AI "
            "M11 unified "
            "multi-camera pipeline"
        )
    )

    parser.add_argument(
        "--list",
        action="store_true",
        help="List all discovered CCTV videos.",
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all discovered CCTV videos.",
    )

    parser.add_argument(
        "--video",
        type=str,
        help="Process one video.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Inference device.",
    )

    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum frames for testing.",
    )

    parser.add_argument(
        "--frame-stride",
        type=int,
        default=1,
        help="Process every Nth frame.",
    )

    parser.add_argument(
        "--clean",
        action="store_true",
        help="Clean unified M11 outputs first.",
    )

    args = parser.parse_args()

    ensure_directories()

    if args.list:
        print_dataset()
        return

    if args.video:

        video = resolve_video(
            args.video
        )

        if args.clean:
            clean_unified_outputs()

        process_video(
            video_path=video,
            device=args.device,
            max_frames=args.max_frames,
            frame_stride=args.frame_stride,
        )

        return

    if args.all:

        process_all(
            device=args.device,
            max_frames=args.max_frames,
            frame_stride=args.frame_stride,
            clean=args.clean,
        )

        return

    parser.print_help()


if __name__ == "__main__":
    main()