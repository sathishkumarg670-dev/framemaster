from ultralytics import YOLO
from pathlib import Path
import cv2

# ==========================================
# VISIONFORGE AI - M4.1 Person Crop Extraction
# ==========================================

PROJECT_ROOT = Path.cwd()

INPUT_VIDEO = PROJECT_ROOT / "data" / "videos" / "camera_01.avi"
OUTPUT_DIR = PROJECT_ROOT / "data" / "crops" / "camera_01"

# Create output folder
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load YOLO11 Nano
model = YOLO("yolo11n.pt")

print("===================================")
print("       VISIONFORGE AI")
print("       PERSON CROP EXTRACTION")
print("===================================")
print(f"Input video : {INPUT_VIDEO}")
print(f"Crop folder : {OUTPUT_DIR}")
print()
print("Extracting person crops...")
print()

# Process video with tracking
results = model.track(
    source=str(INPUT_VIDEO),
    tracker="botsort.yaml",
    classes=[0],          # Person only
    conf=0.35,
    device="cpu",
    stream=True,
    persist=True,
    verbose=False,
)

crop_count = 0
frame_count = 0

# Extract crops from tracked persons
for result in results:

    frame_count += 1

    if result.orig_img is None:
        continue

    # Only extract every 15th frame
    if frame_count % 15 != 0:
        continue

    boxes = result.boxes

    if boxes is None or len(boxes) == 0:
        continue

    frame = result.orig_img

    for i, box in enumerate(boxes):

        # Get tracking ID
        if box.id is not None:
            track_id = int(box.id.item())
        else:
            track_id = i

        # Bounding box coordinates
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

        # Make sure coordinates stay inside the frame
        h, w = frame.shape[:2]

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        # Ignore extremely small detections
        crop_width = x2 - x1
        crop_height = y2 - y1

        if crop_width < 40 or crop_height < 80:
            continue

        # Extract person crop
        crop = frame[y1:y2, x1:x2]

        if crop.size == 0:
            continue

        # Save crop
        filename = (
            f"person_{track_id:03d}_"
            f"frame_{frame_count:05d}.jpg"
        )

        output_path = OUTPUT_DIR / filename

        cv2.imwrite(str(output_path), crop)

        crop_count += 1

    # Progress message
    if frame_count % 300 == 0:
        print(
            f"Processed frames: {frame_count} | "
            f"Crops saved: {crop_count}"
        )

print()
print("===================================")
print("Person crop extraction completed!")
print("===================================")
print(f"Frames processed : {frame_count}")
print(f"Crops saved      : {crop_count}")
print(f"Output folder    : {OUTPUT_DIR}")
print()