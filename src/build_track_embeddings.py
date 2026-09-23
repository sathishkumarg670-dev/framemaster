from pathlib import Path
import json
import numpy as np

# ==========================================
# VISIONFORGE AI - M5.3
# Track-Level Re-ID Embeddings
# ==========================================

PROJECT_ROOT = Path.cwd()

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "reid"
    / "camera_01"
    / "embeddings.json"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "reid"
    / "camera_01"
    / "track_embeddings.json"
)

print("===================================")
print("       VISIONFORGE AI")
print("       TRACK-LEVEL RE-ID")
print("===================================")
print()

# ------------------------------------------
# Load crop embeddings
# ------------------------------------------

with open(INPUT_FILE, "r", encoding="utf-8") as file:
    data = json.load(file)

print(f"Crop embeddings loaded : {len(data)}")
print()

# ------------------------------------------
# Group embeddings by track ID
# ------------------------------------------

tracks = {}

for item in data:

    track_id = item.get("track_id")

    if track_id is None:
        continue

    embedding = item.get("embedding")

    if not embedding:
        continue

    if track_id not in tracks:
        tracks[track_id] = []

    tracks[track_id].append(
        np.array(embedding, dtype=np.float32)
    )

print(f"Unique tracks found : {len(tracks)}")
print()

# ------------------------------------------
# Build representative embedding
# ------------------------------------------

track_embeddings = []

for track_id in sorted(tracks):

    vectors = tracks[track_id]

    # Average all crop embeddings
    mean_embedding = np.mean(
        vectors,
        axis=0
    )

    # L2 normalize
    norm = np.linalg.norm(mean_embedding)

    if norm > 0:
        mean_embedding = mean_embedding / norm

    track_embeddings.append({
        "track_id": track_id,
        "num_crops": len(vectors),
        "embedding_dimension": len(mean_embedding),
        "embedding": mean_embedding.tolist()
    })

# ------------------------------------------
# Save
# ------------------------------------------

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        track_embeddings,
        file
    )

# ------------------------------------------
# Summary
# ------------------------------------------

print("===================================")
print("Track embedding generation complete!")
print("===================================")
print(f"Tracks processed : {len(track_embeddings)}")
print("Embedding size   : 512")
print(f"Output file      : {OUTPUT_FILE}")
print()

print("Track summary:")

for item in track_embeddings[:10]:

    print(
        f"  Track {item['track_id']:03d} "
        f"→ {item['num_crops']} crops "
        f"→ {item['embedding_dimension']}D"
    )

print()
print("Representative embeddings are ready!")