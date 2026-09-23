from ultralytics import YOLO
from pathlib import Path

PROJECT_ROOT = Path.cwd()

INPUT_VIDEO = PROJECT_ROOT / "data" / "videos" / "camera_02.avi"
OUTPUT_DIR = PROJECT_ROOT / "runs" / "tracking"

model = YOLO("yolo11n.pt")

results = model.track(
    source=str(INPUT_VIDEO),
    tracker="botsort.yaml",
    save=True,
    project=str(OUTPUT_DIR),
    name="camera_02",
    exist_ok=True,
    conf=0.35,
    device="cpu",
    classes=[0],
    stream=True,
    verbose=True
)

for result in results:
    pass

print()
print("DONE")
print("Camera 2 person tracking completed.")
print("Output:", OUTPUT_DIR / "camera_02")