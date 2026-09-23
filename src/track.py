from ultralytics import YOLO
from pathlib import Path

# ==========================================
# VISIONFORGE AI - M3.1 Person Tracking
# ==========================================

PROJECT_ROOT = Path.cwd()

INPUT_VIDEO = PROJECT_ROOT / "data" / "videos" / "camera_01.avi"
OUTPUT_DIR = PROJECT_ROOT / "runs" / "tracking"

# Load YOLO11 Nano
model = YOLO("yolo11n.pt")

print("===================================")
print("       VISIONFORGE AI")
print("       PERSON TRACKING")
print("===================================")
print(f"Input video : {INPUT_VIDEO}")
print(f"Output folder: {OUTPUT_DIR}")
print()
print("Tracking persons only...")
print()

# YOLO class 0 = person
results = model.track(
    source=str(INPUT_VIDEO),
    tracker="botsort.yaml",
    save=True,
    project=str(OUTPUT_DIR),
    name="camera_01",
    exist_ok=True,
    conf=0.35,
    device="cpu",
    classes=[0],
    stream=True,
    verbose=True,
)

# Consume the generator so every frame is processed
for result in results:
    pass
print()
print("===================================")
print("Person tracking completed!")
print("===================================")
print(f"Check: {OUTPUT_DIR / 'camera_01'}")