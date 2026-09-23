import json
import numpy as np
from pathlib import Path

INPUT = Path("data/reid/camera_01/selected_market1501_embeddings.json")
OUTPUT = Path("data/reid/camera_01/market1501_track_embeddings.json")

with open(INPUT, "r") as f:
    data = json.load(f)

tracks = {}

for item in data:
    crop_path = Path(item["crop"])
    parts = crop_path.stem.split("_")

    track_id = parts[1]
    embedding = np.array(item["embedding"], dtype=np.float32)

    tracks.setdefault(track_id, []).append(embedding)

track_embeddings = []

for track_id, embeddings in sorted(tracks.items()):
    matrix = np.stack(embeddings)

    mean_embedding = matrix.mean(axis=0)

    norm = np.linalg.norm(mean_embedding)

    if norm > 0:
        mean_embedding = mean_embedding / norm

    track_embeddings.append({
        "track_id": int(track_id),
        "num_crops": len(embeddings),
        "embedding": mean_embedding.tolist()
    })

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

with open(OUTPUT, "w") as f:
    json.dump(track_embeddings, f)

print()
print("DONE")
print("Tracks:", len(track_embeddings))
print("Input crops:", len(data))
print("Embedding dimension:", len(track_embeddings[0]["embedding"]))
print("Output:", OUTPUT)