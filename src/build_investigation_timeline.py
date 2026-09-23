import json
import sqlite3
from pathlib import Path

DB_PATH = Path("data/evidence/visionforge.db")

OUTPUT = Path(
    "data/evidence/investigation_timeline.json"
)

connection = sqlite3.connect(DB_PATH)
connection.row_factory = sqlite3.Row
cursor = connection.cursor()

cursor.execute("""
SELECT
    m.camera_01_track,
    m.camera_02_track,
    m.similarity,
    e.camera_id,
    e.track_id,
    e.frame,
    e.timestamp_seconds,
    e.evidence_type,
    e.description
FROM cross_camera_matches m
LEFT JOIN evidence_events e
ON (
    (e.camera_id = 'camera_01'
     AND e.track_id = m.camera_01_track)
    OR
    (e.camera_id = 'camera_02'
     AND e.track_id = m.camera_02_track)
)
ORDER BY
    m.similarity DESC,
    e.timestamp_seconds ASC
""")

rows = cursor.fetchall()

timeline = {}

for row in rows:

    key = (
        row["camera_01_track"],
        row["camera_02_track"]
    )

    if key not in timeline:

        timeline[key] = {
            "camera_01_track": row["camera_01_track"],
            "camera_02_track": row["camera_02_track"],
            "similarity": row["similarity"],
            "evidence": []
        }

    if row["camera_id"] is not None:

        timeline[key]["evidence"].append({
            "camera_id": row["camera_id"],
            "track_id": row["track_id"],
            "frame": row["frame"],
            "timestamp_seconds": row["timestamp_seconds"],
            "evidence_type": row["evidence_type"],
            "description": row["description"]
        })

timeline = list(timeline.values())

timeline.sort(
    key=lambda x: x["similarity"],
    reverse=True
)

OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True
)

with open(OUTPUT, "w") as f:
    json.dump(timeline, f, indent=2)

connection.close()

print()
print("DONE")
print("Timeline entries:", len(timeline))
print("Output:", OUTPUT)

print()
print("Top 10 investigation candidates:")
print()

for item in timeline[:10]:

    print(
        f"Camera 1 Track {item['camera_01_track']:03d} "
        f"<-> Camera 2 Track {item['camera_02_track']:03d} | "
        f"Similarity: {item['similarity']:.4f} | "
        f"Evidence: {len(item['evidence'])}"
    )