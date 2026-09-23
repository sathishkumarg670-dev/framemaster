"""
VISIONFORGE AI
Dynamic Person Re-ID Embedding Generator

Reads person crops from:

    data/crops/<video_id>/

and generates 512-dimensional appearance embeddings
using the proper Torchreid OSNet Market1501 model.

Outputs:

    data/reid/<video_id>/market1501_embeddings.json
    data/reid/<video_id>/market1501_track_embeddings.json

Usage:

    python src/generate_dynamic_reid.py --video camera_02.avi
"""

from pathlib import Path
import argparse
import json
import time

import cv2
import numpy as np
import torch
import torchreid


PROJECT_ROOT = Path(__file__).resolve().parent.parent

CROPS_DIRECTORY = PROJECT_ROOT / "data" / "crops"
REID_DIRECTORY = PROJECT_ROOT / "data" / "reid"

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "reid"
    / "osnet_x1_0_market1501.pt"
)

IMAGE_SIZE = (256, 128)

BATCH_SIZE = 8


def get_video_id(video_name):
    return Path(video_name).stem.lower().replace(
        " ", "_"
    )


def load_model():
    print()
    print("=" * 70)
    print("LOADING MARKET1501 OSNET")
    print("=" * 70)
    print("Checkpoint:", MODEL_PATH)
    print("Device: CPU")
    print()

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Re-ID checkpoint not found: {MODEL_PATH}"
        )

    model = torchreid.models.build_model(
        name="osnet_x1_0",
        num_classes=751,
        loss="softmax",
        pretrained=False,
        use_gpu=False,
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=False,
    )

    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    cleaned_state_dict = {}

    for key, value in state_dict.items():

        if key.startswith("module."):
            key = key[7:]

        cleaned_state_dict[key] = value

    model.load_state_dict(
        cleaned_state_dict,
        strict=True,
    )

    model.eval()

    print("Model loaded successfully.")
    print(
        "Parameters:",
        sum(
            parameter.numel()
            for parameter in model.parameters()
        ),
    )

    return model


def preprocess_image(image_path):
    image = cv2.imread(
        str(image_path)
    )

    if image is None:
        return None

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB,
    )

    image = cv2.resize(
        image,
        (
            IMAGE_SIZE[1],
            IMAGE_SIZE[0],
        ),
    )

    image = image.astype(
        np.float32
    ) / 255.0

    mean = np.array(
        [0.485, 0.456, 0.406],
        dtype=np.float32,
    )

    std = np.array(
        [0.229, 0.224, 0.225],
        dtype=np.float32,
    )

    image = (
        image - mean
    ) / std

    image = torch.from_numpy(
        image
    ).permute(
        2,
        0,
        1,
    )

    return image


def l2_normalize(vector):
    norm = np.linalg.norm(vector)

    if norm == 0:
        return vector

    return vector / norm


def generate_embeddings(video_name):

    video_id = get_video_id(video_name)

    crop_directory = (
        CROPS_DIRECTORY / video_id
    )

    output_directory = (
        REID_DIRECTORY / video_id
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = (
        output_directory
        / "market1501_embeddings.json"
    )

    if not crop_directory.exists():
        raise FileNotFoundError(
            f"Crop directory not found: {crop_directory}"
        )

    crop_files = sorted(
        crop_directory.glob("*.jpg")
    )

    if not crop_files:
        raise RuntimeError(
            f"No person crops found in {crop_directory}"
        )

    print()
    print("=" * 70)
    print("VISIONFORGE AI — DYNAMIC RE-ID")
    print("=" * 70)
    print("Video:", video_name)
    print("Video ID:", video_id)
    print("Crops:", len(crop_files))
    print("Batch size:", BATCH_SIZE)
    print("=" * 70)

    model = load_model()

    start_time = time.time()

    embeddings = []

    failed = 0

    for start in range(
        0,
        len(crop_files),
        BATCH_SIZE,
    ):

        batch_files = crop_files[
            start:start + BATCH_SIZE
        ]

        batch_images = []
        valid_files = []

        for crop_file in batch_files:

            image = preprocess_image(
                crop_file
            )

            if image is None:
                failed += 1
                continue

            batch_images.append(image)
            valid_files.append(crop_file)

        if not batch_images:
            continue

        batch = torch.stack(
            batch_images
        )

        with torch.no_grad():

            features = model(
                batch
            )

        features = features.cpu().numpy()

        for crop_file, feature in zip(
            valid_files,
            features,
        ):

            feature = l2_normalize(
                feature.astype(
                    np.float32
                )
            )

            embeddings.append(
                {
                    "video_id": video_id,
                    "video_filename": video_name,
                    "crop_filename": crop_file.name,
                    "crop_path": str(crop_file),
                    "track_id": int(
                        crop_file.stem.split(
                            "_"
                        )[1]
                    ),
                    "frame_index": int(
                        crop_file.stem.split(
                            "_"
                        )[3]
                    ),
                    "embedding_dimension": int(
                        len(feature)
                    ),
                    "embedding": feature.tolist(),
                }
            )

        processed = min(
            start + BATCH_SIZE,
            len(crop_files),
        )

        if processed % 40 == 0 or processed == len(
            crop_files
        ):
            print(
                "Processed:",
                processed,
                "/",
                len(crop_files),
            )

    result = {
        "video_id": video_id,
        "video_filename": video_name,
        "model": "osnet_x1_0_market1501",
        "checkpoint": str(MODEL_PATH),
        "embedding_dimension": 512,
        "total_crops": len(crop_files),
        "generated_embeddings": len(
            embeddings
        ),
        "failed_crops": failed,
        "generated_at": time.strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
        "processing_time_seconds": round(
            time.time() - start_time,
            2,
        ),
        "embeddings": embeddings,
    }

    with open(
        output_file,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            result,
            file,
            indent=2,
        )

    print()
    print("=" * 70)
    print("RE-ID EMBEDDING GENERATION COMPLETE")
    print("=" * 70)
    print("Crops:", len(crop_files))
    print(
        "Embeddings:",
        len(embeddings),
    )
    print("Failed:", failed)
    print("Dimension:", 512)
    print("Output:", output_file)
    print(
        "Time:",
        result["processing_time_seconds"],
        "seconds",
    )
    print("=" * 70)

    return result


def build_track_embeddings(video_name):

    video_id = get_video_id(video_name)

    embeddings_file = (
        REID_DIRECTORY
        / video_id
        / "market1501_embeddings.json"
    )

    output_file = (
        REID_DIRECTORY
        / video_id
        / "market1501_track_embeddings.json"
    )

    if not embeddings_file.exists():
        raise FileNotFoundError(
            f"Embedding file not found: {embeddings_file}"
        )

    with open(
        embeddings_file,
        "r",
        encoding="utf-8",
    ) as file:

        data = json.load(file)

    grouped = {}

    for item in data["embeddings"]:

        track_id = int(
            item["track_id"]
        )

        grouped.setdefault(
            track_id,
            [],
        ).append(
            np.array(
                item["embedding"],
                dtype=np.float32,
            )
        )

    track_embeddings = []

    for track_id, vectors in sorted(
        grouped.items()
    ):

        matrix = np.vstack(vectors)

        mean_embedding = matrix.mean(
            axis=0
        )

        mean_embedding = l2_normalize(
            mean_embedding
        )

        track_embeddings.append(
            {
                "video_id": video_id,
                "video_filename": video_name,
                "track_id": track_id,
                "observation_count": len(
                    vectors
                ),
                "embedding_dimension": 512,
                "embedding": mean_embedding.tolist(),
            }
        )

    result = {
        "video_id": video_id,
        "video_filename": video_name,
        "model": data["model"],
        "embedding_dimension": 512,
        "track_count": len(
            track_embeddings
        ),
        "source_observations": len(
            data["embeddings"]
        ),
        "track_embeddings": track_embeddings,
    }

    with open(
        output_file,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            result,
            file,
            indent=2,
        )

    print()
    print("=" * 70)
    print("TRACK-LEVEL RE-ID PROFILES")
    print("=" * 70)
    print("Tracks:", len(track_embeddings))
    print(
        "Source observations:",
        len(data["embeddings"]),
    )
    print("Output:", output_file)
    print("=" * 70)


def main():

    parser = argparse.ArgumentParser(
        description=(
            "VISIONFORGE dynamic "
            "person Re-ID"
        )
    )

    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Video filename",
    )

    args = parser.parse_args()

    generate_embeddings(
        args.video
    )

    build_track_embeddings(
        args.video
    )


if __name__ == "__main__":
    main()