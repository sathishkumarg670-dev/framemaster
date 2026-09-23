from pathlib import Path
import json
from collections import Counter

# ==========================================
# VISIONFORGE AI - M4.4
# Person Appearance Profile Builder
# ==========================================

PROJECT_ROOT = Path.cwd()

ATTR_DIR = PROJECT_ROOT / "data" / "attributes" / "camera_01"

COLOR_FILE = ATTR_DIR / "person_attributes.json"
OBJECT_FILE = ATTR_DIR / "objects" / "person_objects.json"

OUTPUT_FILE = ATTR_DIR / "appearance_profiles.json"

print("===================================")
print("       VISIONFORGE AI")
print("       APPEARANCE PROFILE BUILDER")
print("===================================")
print()

# ------------------------------------------
# Load clothing attributes
# ------------------------------------------

with open(COLOR_FILE, "r", encoding="utf-8") as file:
    color_data = json.load(file)

# ------------------------------------------
# Load detected objects
# ------------------------------------------

with open(OBJECT_FILE, "r", encoding="utf-8") as file:
    object_data = json.load(file)

print(f"Clothing records : {len(color_data)}")
print(f"Object records   : {len(object_data)}")
print()

# ------------------------------------------
# Group records by track ID
# ------------------------------------------

profiles = {}

for record in color_data:

    track_id = record.get("track_id")

    if track_id is None:
        continue

    if track_id not in profiles:
        profiles[track_id] = {
            "track_id": track_id,
            "frames": [],
            "upper_body_colors": [],
            "lower_body_colors": [],
            "objects": []
        }

    profiles[track_id]["frames"].append(
        record.get("frame")
    )

    attributes = record.get("attributes", {})

    upper = attributes.get("upper_body_color")
    lower = attributes.get("lower_body_color")

    if upper:
        profiles[track_id]["upper_body_colors"].append(upper)

    if lower:
        profiles[track_id]["lower_body_colors"].append(lower)


# ------------------------------------------
# Add detected objects
# ------------------------------------------

for record in object_data:

    track_id = record.get("track_id")

    if track_id is None:
        continue

    if track_id not in profiles:
        profiles[track_id] = {
            "track_id": track_id,
            "frames": [],
            "upper_body_colors": [],
            "lower_body_colors": [],
            "objects": []
        }

    for obj in record.get("objects", []):

        object_name = obj.get("object")

        if object_name:
            profiles[track_id]["objects"].append(
                object_name
            )


# ------------------------------------------
# Convert raw observations into profiles
# ------------------------------------------

final_profiles = []

for track_id, profile in sorted(profiles.items()):

    upper_colors = profile["upper_body_colors"]
    lower_colors = profile["lower_body_colors"]
    objects = profile["objects"]

    # Most frequently observed colors
    upper_counter = Counter(upper_colors)
    lower_counter = Counter(lower_colors)

    dominant_upper = (
        upper_counter.most_common(1)[0][0]
        if upper_counter
        else "unknown"
    )

    dominant_lower = (
        lower_counter.most_common(1)[0][0]
        if lower_counter
        else "unknown"
    )

    # Most frequently detected objects
    object_counter = Counter(objects)

    detected_objects = []

    for name, count in object_counter.most_common():
        detected_objects.append({
            "name": name,
            "observations": count
        })

    # Build final profile
    final_profile = {
        "track_id": track_id,

        "observation_summary": {
            "frames_observed": len(profile["frames"]),
            "first_frame": min(profile["frames"])
            if profile["frames"] else None,
            "last_frame": max(profile["frames"])
            if profile["frames"] else None
        },

        "appearance": {
            "upper_body_color": dominant_upper,
            "lower_body_color": dominant_lower
        },

        "objects": detected_objects
    }

    final_profiles.append(final_profile)


# ------------------------------------------
# Save appearance profiles
# ------------------------------------------

with open(OUTPUT_FILE, "w", encoding="utf-8") as file:

    json.dump(
        final_profiles,
        file,
        indent=2
    )


print("===================================")
print("Appearance profile creation complete!")
print("===================================")
print(f"Unique tracks : {len(final_profiles)}")
print(f"Output file   : {OUTPUT_FILE}")
print()

# Show a small preview
print("Profile preview:")
print()

for profile in final_profiles[:10]:

    print(
        f"Track {profile['track_id']:03d} | "
        f"Upper: {profile['appearance']['upper_body_color']} | "
        f"Lower: {profile['appearance']['lower_body_color']} | "
        f"Frames: {profile['observation_summary']['frames_observed']}"
    )

print()
print("Profiles are ready for Re-ID preparation.")