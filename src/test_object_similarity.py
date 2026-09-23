import json
import numpy as np
from pathlib import Path

INPUT = Path("data/reid/camera_01/object_embeddings.json")
OUTPUT = Path("data/reid/camera_01/object_similarity_report.json")

with open(INPUT, "r") as f:
    objects = json.load(f)

results = []

for i in range(len(objects)):
    a = np.array(objects[i]["embedding"], dtype=np.float32)

    for j in range(i + 1, len(objects)):
        b = np.array(objects[j]["embedding"], dtype=np.float32)

        similarity = float(np.dot(a, b))

        results.append({
            "object_a": objects[i]["object"],
            "frame_a": objects[i]["frame"],
            "crop_a": objects[i]["crop"],
            "object_b": objects[j]["object"],
            "frame_b": objects[j]["frame"],
            "crop_b": objects[j]["crop"],
            "similarity": similarity
        })

results.sort(
    key=lambda x: x["similarity"],
    reverse=True
)

with open(OUTPUT, "w") as f:
    json.dump(results, f, indent=2)

print()
print("DONE")
print("Objects:", len(objects))
print("Comparisons:", len(results))
print()
print("Top candidate similarities:")

for item in results[:10]:
    print(
        f"{item['object_a']} (frame {item['frame_a']}) <-> "
        f"{item['object_b']} (frame {item['frame_b']}) | "
        f"Similarity: {item['similarity']:.4f}"
    )

print()
print("Output:", OUTPUT)