from ultralytics import YOLO
from pathlib import Path

# ==============================
# VISIONFORGE AI - M1 Detection
# ==============================

INPUT_VIDEO = Path("data/videos/camera_01.avi")

# Load the YOLO11 nano model
model = YOLO("yolo11n.pt")

print("===================================")
print("       VISIONFORGE AI")
print("       CCTV Detection")
print("===================================")
print(f"Input video: {INPUT_VIDEO}")
print("Starting detection...")
print()

# Run YOLO detection
model.predict(
    source=str(INPUT_VIDEO),
    save=True,
    project="runs/detection",
    name="camera_01",
    exist_ok=True,
    conf=0.35,
    device="cpu",
    verbose=True,
)

print()
print("===================================")
print("Detection completed!")
print("===================================")
print("Check the runs/detection/camera_01 folder.")