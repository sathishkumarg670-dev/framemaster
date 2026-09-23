from ultralytics import YOLO
from pathlib import Path
import cv2
import json

INPUT_VIDEO = Path("data/videos/camera_01.avi")
OUTPUT_DIR = Path("data/crops/camera_01/objects")
OUTPUT_JSON = Path("data/attributes/camera_01/objects/detected_object_crops.json")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)

model = YOLO("yolo11n.pt")

TARGET_CLASSES = {
    "backpack",
    "handbag",
    "tie",
    "suitcase",
    "umbrella"
}

results = model.predict(
    source=str(INPUT_VIDEO),
    conf=0.15,
    device="cpu",
    stream=True,
    verbose=False
)

records = []
frame_count = 0
crop_count = 0

for result in results:
    frame_count += 1

    if result.orig_img is None or result.boxes is None:
        continue

    frame = result.orig_img
    h, w = frame.shape[:2]

    for i, box in enumerate(result.boxes):

        class_id = int(box.cls.item())
        class_name = model.names[class_id]

        if class_name not in TARGET_CLASSES:
            continue

        confidence = float(box.conf.item())

        x1, y1, x2, y2 = map(
            int,
            box.xyxy[0].tolist()
        )

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        object_width = x2 - x1
        object_height = y2 - y1

        if object_width < 10 or object_height < 10:
            continue

        crop = frame[y1:y2, x1:x2]

        if crop.size == 0:
            continue

        filename = (
            f"{class_name}_"
            f"frame_{frame_count:05d}_"
            f"det_{i:03d}.jpg"
        )

        output_path = OUTPUT_DIR / filename

        cv2.imwrite(str(output_path), crop)

        records.append({
            "object": class_name,
            "frame": frame_count,
            "confidence": confidence,
            "bbox": [x1, y1, x2, y2],
            "crop": str(output_path)
        })

        crop_count += 1

    if frame_count % 300 == 0:
        print(
            f"Processed frames: {frame_count} | "
            f"Object crops: {crop_count}"
        )

with open(OUTPUT_JSON, "w") as f:
    json.dump(records, f, indent=2)

print()
print("DONE")
print("Frames processed:", frame_count)
print("Object crops:", crop_count)
print("Output folder:", OUTPUT_DIR)
print("Metadata:", OUTPUT_JSON)