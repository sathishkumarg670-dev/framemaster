import cv2
import json
import sqlite3
from pathlib import Path

PROJECT_ROOT = Path.cwd()

DB_PATH = PROJECT_ROOT / "data" / "evidence" / "visionforge.db"

VIDEOS = {
    "camera_01": PROJECT_ROOT / "data" / "videos" / "camera_01.avi",
    "camera_02": PROJECT_ROOT / "data" / "videos" / "camera_02.avi",
}

OUTPUT_DIR = PROJECT_ROOT / "data" / "evidence" / "frames"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

connection = sqlite3.connect(DB_PATH)
cursor = connection.cursor()

cursor.execute("""
SELECT
    camera_01_track,
    camera_02_track,
    similarity
FROM cross_camera_matches
ORDER BY similarity DESC
LIMIT 20
""")

matches = cursor.fetchall()

print("Candidate matches:", len(matches))

# Collect tracks that need evidence frames.
tracks = {
    "camera_01": set(),
    "camera_02": set()
}

for camera_01_track, camera_02_track, similarity in matches:
    tracks["camera_01"].add(camera_01_track)
    tracks["camera_02"].add(camera_02_track)

print("Camera 1 tracks:", len(tracks["camera_01"]))
print("Camera 2 tracks:", len(tracks["camera_02"]))


def extract_frames(camera_id, track_ids):

    video_path = VIDEOS[camera_id]

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print("Could not open:", video_path)
        return []

    saved = []

    frame_number = 0

    while True:

        success, frame = cap.read()

        if not success:
            break

        frame_number += 1

        # Sample every 15 frames.
        if frame_number % 15 != 0:
            continue

        # We use the previously extracted crop filenames
        # to locate frames belonging to each track.
        crop_dir = (
            PROJECT_ROOT
            / "data"
            / "crops"
            / camera_id
        )

        for track_id in track_ids:

            pattern = (
                f"person_{int(track_id):03d}_"
                f"frame_{frame_number:05d}.jpg"
            )

            crop_path = crop_dir / pattern

            if not crop_path.exists():
                continue

            output_name = (
                f"{camera_id}_"
                f"track_{int(track_id):03d}_"
                f"frame_{frame_number:05d}.jpg"
            )

            output_path = OUTPUT_DIR / output_name

            cv2.imwrite(
                str(output_path),
                frame
            )

            timestamp = (
                cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            )

            saved.append({
                "camera_id": camera_id,
                "track_id": int(track_id),
                "frame": frame_number,
                "timestamp_seconds": round(timestamp, 3),
                "image": str(output_path)
            })

        if frame_number % 500 == 0:
            print(
                f"{camera_id}: "
                f"{frame_number} frames scanned | "
                f"Evidence saved: {len(saved)}"
            )

    cap.release()

    return saved


evidence = []

evidence.extend(
    extract_frames(
        "camera_01",
        tracks["camera_01"]
    )
)

evidence.extend(
    extract_frames(
        "camera_02",
        tracks["camera_02"]
    )
)

for item in evidence:

    cursor.execute("""
    INSERT INTO evidence_events
    (
        camera_id,
        track_id,
        frame,
        timestamp_seconds,
        evidence_type,
        description
    )
    VALUES (?, ?, ?, ?, ?, ?)
    """, (
        item["camera_id"],
        item["track_id"],
        item["frame"],
        item["timestamp_seconds"],
        "person_frame",
        "Representative person evidence frame"
    ))

connection.commit()
connection.close()

metadata_path = (
    OUTPUT_DIR / "evidence_frames.json"
)

with open(metadata_path, "w") as f:
    json.dump(evidence, f, indent=2)

print()
print("DONE")
print("Evidence frames:", len(evidence))
print("Metadata:", metadata_path)
print("Evidence folder:", OUTPUT_DIR)