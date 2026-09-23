from pathlib import Path
import json
import cv2
import numpy as np
import torch
import torchreid
from PIL import Image

# ==========================================
# VISIONFORGE AI - M5.2
# Batch Re-ID Embedding Generation
# ==========================================

PROJECT_ROOT = Path.cwd()

CROP_DIR = PROJECT_ROOT / "data" / "crops" / "camera_01"
OUTPUT_DIR = PROJECT_ROOT / "data" / "reid" / "camera_01"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "embeddings.json"

DEVICE = torch.device("cpu")

# Process several images at once.
# 16 is reasonable for CPU and small 128x256 images.
BATCH_SIZE = 16

print("===================================")
print("       VISIONFORGE AI")
print("       RE-ID EMBEDDING GENERATOR")
print("===================================")
print()

print("Loading OSNet...")

model = torchreid.models.build_model(
    name="osnet_x1_0",
    num_classes=1000,
    pretrained=True
)

model = model.to(DEVICE)
model.eval()

print("OSNet loaded successfully!")
print()

# ------------------------------------------
# Build preprocessing transform
# ------------------------------------------

transform = torchreid.data.transforms.build_transforms(
    height=256,
    width=128,
    is_train=False
)[1]

# ------------------------------------------
# Find crops
# ------------------------------------------

image_files = sorted(CROP_DIR.glob("*.jpg"))

if not image_files:
    raise FileNotFoundError(
        f"No person crops found in {CROP_DIR}"
    )

print(f"Person crops found: {len(image_files)}")
print(f"Batch size        : {BATCH_SIZE}")
print(f"Device             : {DEVICE}")
print()
print("Generating embeddings...")
print()

embeddings = []

# ------------------------------------------
# Process batches
# ------------------------------------------

for start in range(0, len(image_files), BATCH_SIZE):

    batch_files = image_files[
        start:start + BATCH_SIZE
    ]

    batch_tensors = []
    valid_files = []

    for image_path in batch_files:

        image = cv2.imread(str(image_path))

        if image is None:
            print(f"Warning: could not read {image_path.name}")
            continue

        # OpenCV BGR → RGB
        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        pil_image = Image.fromarray(image)

        tensor = transform(pil_image)

        batch_tensors.append(tensor)
        valid_files.append(image_path)

    if not batch_tensors:
        continue

    batch = torch.stack(batch_tensors).to(DEVICE)

    # --------------------------------------
    # Generate embeddings
    # --------------------------------------

    with torch.no_grad():

        features = model(batch)

        features = features.view(
            features.size(0),
            -1
        )

        # L2 normalization
        features = torch.nn.functional.normalize(
            features,
            p=2,
            dim=1
        )

    features = features.cpu().numpy()

    # --------------------------------------
    # Store results
    # --------------------------------------

    for image_path, feature in zip(
        valid_files,
        features
    ):

        parts = image_path.stem.split("_")

        track_id = None
        frame_number = None

        try:
            track_id = int(parts[1])
            frame_number = int(parts[3])
        except (IndexError, ValueError):
            pass

        embeddings.append({
            "image": str(
                image_path.relative_to(PROJECT_ROOT)
            ),
            "track_id": track_id,
            "frame": frame_number,
            "embedding": feature.tolist()
        })

    processed = min(
        start + len(batch_files),
        len(image_files)
    )

    print(
        f"Processed: {processed}/{len(image_files)} "
        f"({processed / len(image_files) * 100:.1f}%)"
    )

# ------------------------------------------
# Save embeddings
# ------------------------------------------

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        embeddings,
        file
    )

print()
print("===================================")
print("Embedding generation completed!")
print("===================================")
print(f"Crops processed : {len(embeddings)}")
print(f"Embedding size  : 512")
print(f"Output file     : {OUTPUT_FILE}")
print()

# ------------------------------------------
# Verification
# ------------------------------------------

if embeddings:

    first_embedding = embeddings[0]["embedding"]

    print("Verification:")
    print(
        f"First embedding values: "
        f"{first_embedding[:5]}"
    )

    print(
        f"Embedding length: "
        f"{len(first_embedding)}"
    )

print()
print("Re-ID embeddings are ready!")