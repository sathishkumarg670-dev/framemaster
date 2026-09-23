import json
import numpy as np
from pathlib import Path

INPUT = Path("data/reid/camera_01/market1501_track_embeddings.json")
OUTPUT = Path("data/reid/camera_01/market1501_similarity_report.json")

with open(INPUT, "r") as f:
    tracks = json.load(f)

results = []

for i in range(len(tracks)):
    a = np.array(tracks[i]["embedding"], dtype=np.float32)

    for j in range(i + 1, len(tracks)):
        b = np.array(tracks[j]["embedding"], dtype=np.float32)

        similarity = float(np.dot(a, b))

        results.append({
            "track_a": tracks[i]["track_id"],
            "track_b": tracks[j]["track_id"],
            "similarity": similarity
        })

results.sort(key=lambda x: x["similarity"], reverse=True)

with open(OUTPUT, "w") as f:
    json.dump(results, f, indent=2)

print()
print("DONE")
print("Tracks:", len(tracks))
print("Comparisons:", len(results))
print()
print("Top 10 candidate similarities:")

for item in results[:10]:
    print(
        f"Track {item['track_a']:03d} <-> "
        f"Track {item['track_b']:03d} | "
        f"Similarity: {item['similarity']:.4f}"
    )

print()
print("Output:", OUTPUT)