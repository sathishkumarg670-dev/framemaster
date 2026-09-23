from pathlib import Path
import cv2
import json
import numpy as np

# ==========================================
# VISIONFORGE AI - M4.2
# Clothing Color Attribute Extraction
# ==========================================

PROJECT_ROOT = Path.cwd()

CROP_DIR = PROJECT_ROOT / "data" / "crops" / "camera_01"
OUTPUT_DIR = PROJECT_ROOT / "data" / "attributes" / "camera_01"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "person_attributes.json"

print("===================================")
print("       VISIONFORGE AI")
print("       CLOTHING ATTRIBUTES")
print("===================================")
print(f"Crop folder : {CROP_DIR}")
print(f"Output file : {OUTPUT_FILE}")
print()
print("Extracting clothing colors...")
print()


def classify_color(bgr):
    """
    Convert a BGR color into a simple human-readable color.
    """

    b, g, r = bgr

    # Brightness
    brightness = (int(r) + int(g) + int(b)) / 3

    # Very dark
    if brightness < 50:
        return "black"

    # Very bright
    if brightness > 200 and max(b, g, r) - min(b, g, r) < 35:
        return "white"

    # Gray
    if max(b, g, r) - min(b, g, r) < 30:
        if brightness < 110:
            return "dark_gray"
        return "gray"

    # Color classification
    if r > g * 1.35 and r > b * 1.35:
        return "red"

    if g > r * 1.25 and g > b * 1.15:
        return "green"

    if b > r * 1.25 and b > g * 1.10:
        return "blue"

    if r > 130 and g > 90 and b < 100:
        return "brown"

    if r > 150 and g > 120 and b < 100:
        return "yellow"

    if r > 120 and b > 100 and g < 110:
        return "purple"

    return "mixed"


def dominant_color(image):
    """
    Estimate the dominant color while ignoring
    very dark/light background pixels.
    """

    pixels = image.reshape(-1, 3)

    # Remove extremely dark pixels
    brightness = pixels.mean(axis=1)
    pixels = pixels[(brightness > 25) & (brightness < 245)]

    if len(pixels) == 0:
        pixels = image.reshape(-1, 3)

    # Median is more stable than a single pixel
    color = np.median(pixels, axis=0)

    return classify_color(color)


def extract_regions(image):
    """
    Split a person crop into approximate upper-body
    and lower-body regions.
    """

    height, width = image.shape[:2]

    # Ignore a small portion around the head
    upper_start = int(height * 0.20)
    upper_end = int(height * 0.55)

    lower_start = int(height * 0.55)
    lower_end = int(height * 0.95)

    upper = image[upper_start:upper_end, :]
    lower = image[lower_start:lower_end, :]

    return upper, lower


results = []

image_files = sorted(CROP_DIR.glob("*.jpg"))

print(f"Found {len(image_files)} person crops.")
print()

for index, image_path in enumerate(image_files, start=1):

    image = cv2.imread(str(image_path))

    if image is None:
        continue

    upper_body, lower_body = extract_regions(image)

    upper_color = dominant_color(upper_body)
    lower_color = dominant_color(lower_body)

    # Extract track ID from filename:
    # person_001_frame_00015.jpg
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
        "attributes": {
            "upper_body_color": upper_color,
            "lower_body_color": lower_color
        }
    }

    results.append(person_data)

    if index % 100 == 0:
        print(f"Processed crops: {index}/{len(image_files)}")


# Save JSON
with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
    json.dump(results, file, indent=2)


print()
print("===================================")
print("Attribute extraction completed!")
print("===================================")
print(f"Crops processed : {len(results)}")
print(f"JSON file       : {OUTPUT_FILE}")
print()