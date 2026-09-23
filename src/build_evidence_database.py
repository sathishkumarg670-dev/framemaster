import json
import sqlite3
from pathlib import Path

MATCHES = Path("data/reid/cross_camera_matches.json")
DB_PATH = Path("data/evidence/visionforge.db")

DB_PATH.parent.mkdir(parents=True, exist_ok=True)

with open(MATCHES, "r") as f:
    matches = json.load(f)

connection = sqlite3.connect(DB_PATH)
cursor = connection.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS camera_tracks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    camera_id TEXT NOT NULL,
    track_id INTEGER NOT NULL,
    observations INTEGER,
    UNIQUE(camera_id, track_id)
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS cross_camera_matches (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    camera_01_track INTEGER NOT NULL,
    camera_02_track INTEGER NOT NULL,
    similarity REAL NOT NULL,
    camera_01_observations INTEGER,
    camera_02_observations INTEGER
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS evidence_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    camera_id TEXT NOT NULL,
    track_id INTEGER NOT NULL,
    frame INTEGER,
    timestamp_seconds REAL,
    evidence_type TEXT,
    description TEXT
)
""")

camera_01_tracks = {}
camera_02_tracks = {}

for match in matches:
    camera_01_tracks[match["camera_01_track"]] = (
        match["camera_01_observations"]
    )

    camera_02_tracks[match["camera_02_track"]] = (
        match["camera_02_observations"]
    )

for track_id, observations in camera_01_tracks.items():
    cursor.execute("""
    INSERT OR IGNORE INTO camera_tracks
    (camera_id, track_id, observations)
    VALUES (?, ?, ?)
    """, ("camera_01", track_id, observations))

for track_id, observations in camera_02_tracks.items():
    cursor.execute("""
    INSERT OR IGNORE INTO camera_tracks
    (camera_id, track_id, observations)
    VALUES (?, ?, ?)
    """, ("camera_02", track_id, observations))

for match in matches:
    cursor.execute("""
    INSERT INTO cross_camera_matches
    (
        camera_01_track,
        camera_02_track,
        similarity,
        camera_01_observations,
        camera_02_observations
    )
    VALUES (?, ?, ?, ?, ?)
    """, (
        match["camera_01_track"],
        match["camera_02_track"],
        match["similarity"],
        match["camera_01_observations"],
        match["camera_02_observations"]
    ))

connection.commit()

print()
print("DONE")
print("Database:", DB_PATH)

cursor.execute("SELECT COUNT(*) FROM camera_tracks")
print("Camera tracks:", cursor.fetchone()[0])

cursor.execute("SELECT COUNT(*) FROM cross_camera_matches")
print("Cross-camera matches:", cursor.fetchone()[0])

connection.close()