from ultralytics import YOLO
from pathlib import Path

PROJECT_ROOT = Path.cwd()

INPUT_VIDEO = PROJECT_ROOT / "data" / "videos" / "camera_02.avi"
OUTPUT_DIR = PROJECT_ROOT / "runs" / "detection"

model = YOLO("yolo11n.pt")

model.predict(
    source=str(INPUT_VIDEO),
    save=True,
    project=str(OUTPUT_DIR),
    name="camera_02",
    exist_ok=True,
    conf=0.35,
    device="cpu",
    classes=[0],
    verbose=True,
)

print()
print("DONE")
print("Camera 2 person detection completed.")
print("Output:", OUTPUT_DIR / "camera_02")