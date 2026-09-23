from ultralytics import YOLO
from pathlib import Path
import cv2
import json

PROJECT_ROOT = Path.cwd()

INPUT_VIDEO = PROJECT_ROOT / "data" / "videos" / "camera_01.avi"
OUTPUT_DIR = PROJECT_ROOT / "data" / "crops" / "camera_01" / "objects"
OUTPUT_JSON = PROJECT_ROOT / "data" / "attributes" / "camera_01" / "objects" / "object_crops.json"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)

model = YOLO("yolo11n.pt")

TARGET_CLASSES = {
    24: "backpack",
    25: "umbrella",
    26: "handbag",
    27: "tie",
    28: "suitcase",
}

results = model.track(
    source=str(INPUT_VIDEO),
    tracker="botsort.yaml",
    conf=0.25,
    device="cpu",
    stream=True,
    persist=True,
    verbose=False,
)

records = []
frame_count = 0
crop_count = 0

for result in results:
    frame_count += 1

    if result.orig_img is None:
        continue

    if frame_count % 5 != 0:
        continue

    boxes = result.boxes

    if boxes is None or len(boxes) == 0:
        continue

    frame = result.orig_img
    h, w = frame.shape[:2]

    for i, box in enumerate(boxes):

        class_id = int(box.cls.item())

        if class_id not in TARGET_CLASSES:
            continue

        object_name = TARGET_CLASSES[class_id]

        if box.id is not None:
            track_id = int(box.id.item())
        else:
            track_id = i

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

        if object_width < 20 or object_height < 20:
            continue

        crop = frame[y1:y2, x1:x2]

        if crop.size == 0:
            continue

        filename = (
            f"{object_name}_"
            f"{track_id:03d}_"
            f"frame_{frame_count:05d}.jpg"
        )

        output_path = OUTPUT_DIR / filename

        cv2.imwrite(str(output_path), crop)

        records.append({
            "object": object_name,
            "track_id": track_id,
            "frame": frame_count,
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