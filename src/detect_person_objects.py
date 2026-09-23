from pathlib import Path
import cv2
import json
from ultralytics import YOLO

# ==========================================
# VISIONFORGE AI - M4.3
# Person Object Attribute Detection
# ==========================================

PROJECT_ROOT = Path.cwd()

CROP_DIR = PROJECT_ROOT / "data" / "crops" / "camera_01"
OUTPUT_DIR = PROJECT_ROOT / "data" / "attributes" / "camera_01" / "objects"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "person_objects.json"

# YOLO11 Nano
model = YOLO("yolo11n.pt")

print("===================================")
print("       VISIONFORGE AI")
print("       PERSON OBJECT ATTRIBUTES")
print("===================================")
print(f"Crop folder : {CROP_DIR}")
print(f"Output file : {OUTPUT_FILE}")
print()
print("Detecting backpacks, handbags and other objects...")
print()

# COCO classes useful for person appearance
TARGET_CLASSES = {
    24: "backpack",
    26: "handbag",
    28: "suitcase",
    25: "umbrella",
    27: "tie",
}

results_data = []

image_files = sorted(CROP_DIR.glob("*.jpg"))

print(f"Found {len(image_files)} person crops.")
print()

for index, image_path in enumerate(image_files, start=1):

    image = cv2.imread(str(image_path))

    if image is None:
        continue

    # Detect objects inside the person crop
    result = model.predict(
        source=image,
        conf=0.25,
        device="cpu",
        verbose=False
    )[0]

    detected_objects = []

    if result.boxes is not None:

        for box in result.boxes:

            class_id = int(box.cls.item())
            confidence = float(box.conf.item())

            if class_id in TARGET_CLASSES:

                detected_objects.append({
                    "object": TARGET_CLASSES[class_id],
                    "confidence": round(confidence, 3)
                })

    # Extract track ID and frame number
    parts = image_path.stem.split("_")

    track_id = None
    frame_number = None

    try:
        track_id = int(parts[1])
        frame_number = int(parts[3])
    except (IndexError, ValueError):
        pass

    person_data = {
        "image": image_path.name,
        "track_id": track_id,
        "frame": frame_number,
        "objects": detected_objects
    }

    results_data.append(person_data)

    if index % 100 == 0:
        print(f"Processed crops: {index}/{len(image_files)}")


# Save results
with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
    json.dump(results_data, file, indent=2)


# Summary
object_counts = {}

for person in results_data:

    for obj in person["objects"]:

        name = obj["object"]

        object_counts[name] = object_counts.get(name, 0) + 1


print()
print("===================================")
print("Object attribute extraction completed!")
print("===================================")
print(f"Crops processed : {len(results_data)}")
print()

if object_counts:

    print("Detected objects:")

    for name, count in object_counts.items():
        print(f"  {name}: {count}")

else:

    print("No target objects detected.")

print()
print(f"JSON file: {OUTPUT_FILE}")
print()