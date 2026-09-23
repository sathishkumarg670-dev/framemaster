import json
from pathlib import Path
from collections import defaultdict

INPUT = Path(
    "data/reid/cross_camera_matches.json"
)

OUTPUT = Path(
    "data/reid/cross_camera_candidate_report.json"
)

with open(INPUT, "r") as f:
    matches = json.load(f)

grouped = defaultdict(list)

for match in matches:
    grouped[match["camera_01_track"]].append(match)

report = []

for camera_01_track, candidates in sorted(grouped.items()):

    candidates.sort(
        key=lambda x: x["similarity"],
        reverse=True
    )

    report.append({
        "camera_01_track": camera_01_track,
        "candidate_count": len(candidates),
        "top_candidates": candidates[:5]
    })

OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True
)

with open(OUTPUT, "w") as f:
    json.dump(report, f, indent=2)

print()
print("DONE")
print("Camera 1 tracks:", len(report))
print("Candidates per track: 5")
print("Output:", OUTPUT)

print()
print("Top candidate for each Camera 1 track:")
print()

for item in report:

    top = item["top_candidates"][0]

    print(
        f"Camera 1 Track {item['camera_01_track']:03d} "
        f"-> Camera 2 Track "
        f"{top['camera_02_track']:03d} | "
        f"Similarity: {top['similarity']:.4f}"
    )