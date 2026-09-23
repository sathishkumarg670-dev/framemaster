import json
import numpy as np
from pathlib import Path

CAMERA_01 = Path(
    "data/reid/camera_01/market1501_track_embeddings.json"
)

CAMERA_02 = Path(
    "data/reid/camera_02/market1501_track_embeddings.json"
)

OUTPUT = Path(
    "data/reid/cross_camera_matches.json"
)

with open(CAMERA_01, "r") as f:
    camera_01_tracks = json.load(f)

with open(CAMERA_02, "r") as f:
    camera_02_tracks = json.load(f)

matches = []

for track_01 in camera_01_tracks:

    embedding_01 = np.array(
        track_01["embedding"],
        dtype=np.float32
    )

    for track_02 in camera_02_tracks:

        embedding_02 = np.array(
            track_02["embedding"],
            dtype=np.float32
        )

        similarity = float(
            np.dot(embedding_01, embedding_02)
        )

        matches.append({
            "camera_01_track": track_01["track_id"],
            "camera_02_track": track_02["track_id"],
            "similarity": similarity,
            "camera_01_observations": track_01["num_crops"],
            "camera_02_observations": track_02["num_crops"]
        })

matches.sort(
    key=lambda x: x["similarity"],
    reverse=True
)

OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True
)

with open(OUTPUT, "w") as f:
    json.dump(matches, f, indent=2)

print()
print("DONE")
print("Camera 1 tracks:", len(camera_01_tracks))
print("Camera 2 tracks:", len(camera_02_tracks))
print("Comparisons:", len(matches))

print()
print("Top 20 candidate cross-camera matches:")
print()

for item in matches[:20]:
    print(
        f"Camera 1 Track {item['camera_01_track']:03d} "
        f"<-> Camera 2 Track {item['camera_02_track']:03d} | "
        f"Similarity: {item['similarity']:.4f} | "
        f"Observations: "
        f"{item['camera_01_observations']} + "
        f"{item['camera_02_observations']}"
    )

print()
print("Output:", OUTPUT)