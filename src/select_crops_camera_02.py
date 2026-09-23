from pathlib import Path
import collections
import shutil

SOURCE = Path("data/crops/camera_02")
DEST = Path("data/reid/camera_02/selected_crops")

DEST.mkdir(parents=True, exist_ok=True)

groups = collections.defaultdict(list)

files = sorted(SOURCE.glob("*.jpg"))

for path in files:
    parts = path.stem.split("_")

    if len(parts) < 4:
        continue

    track_id = parts[1]
    groups[track_id].append(path)

selected = []

for track_id, crops in groups.items():

    step = max(1, len(crops) // 5)

    samples = crops[::step][:5]

    selected.extend(samples)

for path in selected:
    shutil.copy2(path, DEST / path.name)

print()
print("DONE")
print("Total crops:", len(files))
print("Tracks:", len(groups))
print("Selected crops:", len(selected))
print("Output:", DEST)