from pathlib import Path
import json
from collections import defaultdict

# ==========================================
# VISIONFORGE AI - M4.5
# Re-ID Dataset Preparation
# ==========================================

PROJECT_ROOT = Path.cwd()

CROP_DIR = PROJECT_ROOT / "data" / "crops" / "camera_01"
PROFILE_FILE = (
    PROJECT_ROOT
    / "data"
    / "attributes"
    / "camera_01"
    / "appearance_profiles.json"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "reid" / "camera_01"
OUTPUT_FILE = OUTPUT_DIR / "reid_index.json"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("===================================")
print("       VISIONFORGE AI")
print("       RE-ID DATA PREPARATION")
print("===================================")
print()

# ------------------------------------------
# Load appearance profiles
# ------------------------------------------

with open(PROFILE_FILE, "r", encoding="utf-8") as file:
    profiles = json.load(file)

profile_map = {
    profile["track_id"]: profile
    for profile in profiles
}

print(f"Appearance profiles : {len(profiles)}")

# ------------------------------------------
# Find all person crops
# ------------------------------------------

crop_files = sorted(CROP_DIR.glob("*.jpg"))

print(f"Person crops        : {len(crop_files)}")
print()

# ------------------------------------------
# Group crops by track ID
# ------------------------------------------

track_crops = defaultdict(list)

for crop_path in crop_files:

    # Expected:
    # person_002_frame_00015.jpg

    parts = crop_path.stem.split("_")

    try:
        track_id = int(parts[1])
        frame_number = int(parts[3])
    except (IndexError, ValueError):
        continue

    track_crops[track_id].append({
        "image": str(
            crop_path.relative_to(PROJECT_ROOT)
        ),
        "frame": frame_number
    })

# ------------------------------------------
# Build Re-ID index
# ------------------------------------------

reid_index = []

for track_id in sorted(track_crops):

    profile = profile_map.get(track_id)

    if profile is None:
        continue

    crops = sorted(
        track_crops[track_id],
        key=lambda item: item["frame"]
    )

    entry = {
        "track_id": track_id,

        "appearance": profile.get(
            "appearance", {}
        ),

        "objects": profile.get(
            "objects", []
        ),

        "observation_summary": profile.get(
            "observation_summary", {}
        ),

        "crops": crops
    }

    reid_index.append(entry)

# ------------------------------------------
# Save index
# ------------------------------------------

with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
    json.dump(
        reid_index,
        file,
        indent=2
    )

# ------------------------------------------
# Summary
# ------------------------------------------

total_indexed_crops = sum(
    len(entry["crops"])
    for entry in reid_index
)

print("===================================")
print("Re-ID preparation completed!")
print("===================================")
print(f"Tracks indexed     : {len(reid_index)}")
print(f"Crops indexed      : {total_indexed_crops}")
print(f"Output file        : {OUTPUT_FILE}")
print()

print("Track summary:")

for entry in reid_index[:10]:

    print(
        f"  Track {entry['track_id']:03d} "
        f"→ {len(entry['crops'])} crops"
    )

print()
print("Ready for Re-ID embedding generation.")