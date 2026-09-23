from ultralytics import YOLO
from pathlib import Path
from collections import Counter

INPUT_VIDEO = Path("data/videos/camera_01.avi")

model = YOLO("yolo11n.pt")

results = model.predict(
    source=str(INPUT_VIDEO),
    conf=0.15,
    device="cpu",
    stream=True,
    verbose=False,
)

counts = Counter()
frames = 0

for result in results:
    frames += 1

    if result.boxes is None:
        continue

    for cls in result.boxes.cls.tolist():
        class_id = int(cls)
        class_name = model.names[class_id]
        counts[class_name] += 1

    if frames % 300 == 0:
        print(f"Processed frames: {frames}")

print()
print("DONE")
print("Frames:", frames)
print()
print("Detected classes:")

for name, count in counts.most_common():
    print(f"{name}: {count}")