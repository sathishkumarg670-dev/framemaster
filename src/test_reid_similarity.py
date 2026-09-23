from pathlib import Path
import json
import numpy as np

# ==========================================
# VISIONFORGE AI - M5.4
# Re-ID Cosine Similarity Test
# ==========================================

PROJECT_ROOT = Path.cwd()

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "reid"
    / "camera_01"
    / "track_embeddings.json"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "reid"
    / "camera_01"
    / "similarity_report.json"
)

print("===================================")
print("       VISIONFORGE AI")
print("       RE-ID SIMILARITY TEST")
print("===================================")
print()

# ------------------------------------------
# Load track embeddings
# ------------------------------------------

with open(INPUT_FILE, "r", encoding="utf-8") as file:
    tracks = json.load(file)

print(f"Track embeddings loaded : {len(tracks)}")
print()

# ------------------------------------------
# Convert embeddings to NumPy
# ------------------------------------------

track_vectors = {}

track_info = {}

for item in tracks:

    track_id = item["track_id"]

    vector = np.array(
        item["embedding"],
        dtype=np.float32
    )

    track_vectors[track_id] = vector

    track_info[track_id] = {
        "num_crops": item["num_crops"]
    }

# ------------------------------------------
# Calculate pairwise cosine similarity
# ------------------------------------------

comparisons = []

track_ids = sorted(track_vectors.keys())

for i in range(len(track_ids)):

    track_a = track_ids[i]

    vector_a = track_vectors[track_a]

    for j in range(i + 1, len(track_ids)):

        track_b = track_ids[j]

        vector_b = track_vectors[track_b]

        # Since vectors were L2-normalized,
        # dot product = cosine similarity.
        similarity = float(
            np.dot(vector_a, vector_b)
        )

        comparisons.append({
            "track_a": track_a,
            "track_b": track_b,
            "similarity": round(similarity, 6),
            "track_a_crops": track_info[track_a]["num_crops"],
            "track_b_crops": track_info[track_b]["num_crops"]
        })

# ------------------------------------------
# Sort by similarity
# ------------------------------------------

comparisons.sort(
    key=lambda item: item["similarity"],
    reverse=True
)

# ------------------------------------------
# Save complete report
# ------------------------------------------

report = {
    "camera": "camera_01",
    "tracks": len(track_ids),
    "comparisons": len(comparisons),
    "results": comparisons
}

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        report,
        file,
        indent=2
    )

# ------------------------------------------
# Display top matches
# ------------------------------------------

print("Top 15 most visually similar track pairs:")
print()

for result in comparisons[:15]:

    print(
        f"Track {result['track_a']:03d} "
        f"<-> "
        f"Track {result['track_b']:03d} "
        f"| Similarity: {result['similarity']:.4f} "
        f"| Crops: "
        f"{result['track_a_crops']} + "
        f"{result['track_b_crops']}"
    )

print()

print("===================================")
print("Similarity test completed!")
print("===================================")
print(f"Tracks compared : {len(track_ids)}")
print(f"Pair comparisons : {len(comparisons)}")
print(f"Report           : {OUTPUT_FILE}")
print()

print(
    "NOTE: High similarity does NOT prove "
    "that two tracks are the same person."
)
print(
    "This test is a Re-ID sanity check before "
    "cross-camera matching."
)