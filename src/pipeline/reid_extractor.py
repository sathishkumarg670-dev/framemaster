"""
VISIONFORGE AI
Unified Re-ID Extractor

Uses:
    Torchreid OSNet
    Market1501 checkpoint

Input:
    CropMetadataCollection

Output:
    Track-level 512-D embeddings
    Representative crop per track
"""

from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json

import numpy as np
import torch
from PIL import Image
from torchvision import transforms


from src.pipeline.models import CropMetadataCollection


# ============================================================
# CONSTANTS
# ============================================================

EMBEDDING_DIMENSION = 512

IMAGE_WIDTH = 128
IMAGE_HEIGHT = 256

DEFAULT_BATCH_SIZE = 16


# ============================================================
# MODEL
# ============================================================

def load_reid_model(
    checkpoint_path: Path,
    device: str = "cpu",
):

    if not checkpoint_path.exists():

        raise FileNotFoundError(
            f"Re-ID checkpoint not found: "
            f"{checkpoint_path}"
        )

    import torchreid

    model = torchreid.models.build_model(
        name="osnet_x1_0",
        num_classes=751,
        loss="softmax",
        pretrained=False,
        use_gpu=False,
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
    )

    if isinstance(checkpoint, dict):

        if "state_dict" in checkpoint:
            state_dict = checkpoint[
                "state_dict"
            ]

        elif "model" in checkpoint:
            state_dict = checkpoint[
                "model"
            ]

        else:
            state_dict = checkpoint

    else:

        state_dict = checkpoint

    cleaned_state_dict = {}

    for key, value in state_dict.items():

        clean_key = key

        if clean_key.startswith(
            "module."
        ):

            clean_key = clean_key[
                len("module.") :
            ]

        cleaned_state_dict[
            clean_key
        ] = value

    model.load_state_dict(
        cleaned_state_dict,
        strict=True,
    )

    model.eval()

    if device != "cpu":

        model = model.to(device)

    return model


# ============================================================
# PREPROCESSING
# ============================================================

def build_transform():

    return transforms.Compose(
        [
            transforms.Resize(
                (
                    IMAGE_HEIGHT,
                    IMAGE_WIDTH,
                )
            ),

            transforms.ToTensor(),

            transforms.Normalize(
                mean=[
                    0.485,
                    0.456,
                    0.406,
                ],

                std=[
                    0.229,
                    0.224,
                    0.225,
                ],
            ),
        ]
    )


# ============================================================
# IMAGE LOADING
# ============================================================

def load_crop_image(
    crop_path: str,
) -> Optional[Image.Image]:

    path = Path(crop_path)

    if not path.exists():
        return None

    try:

        image = Image.open(
            path
        ).convert("RGB")

        return image

    except Exception:

        return None


# ============================================================
# EMBEDDING NORMALIZATION
# ============================================================

def l2_normalize(
    vector: np.ndarray,
) -> np.ndarray:

    norm = np.linalg.norm(
        vector
    )

    if norm == 0:

        return vector

    return vector / norm


# ============================================================
# TRACK EMBEDDINGS
# ============================================================

def generate_track_embeddings(
    crop_metadata: CropMetadataCollection,
    checkpoint_path: Path,
    output_json: Path,
    device: str = "cpu",
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> Tuple[
    Dict[int, List[float]],
    Dict[int, str],
]:

    print(
        "Loading Re-ID model..."
    )

    model = load_reid_model(
        checkpoint_path=
            Path(checkpoint_path),

        device=device,
    )

    transform = build_transform()

    # --------------------------------------------------------
    # Group crops by track
    # --------------------------------------------------------

    crops_by_track = {}

    for crop in crop_metadata.crops:

        crops_by_track.setdefault(
            crop.track_id,
            [],
        ).append(
            crop
        )

    print(
        "Tracks:",
        len(crops_by_track),
    )

    print(
        "Crops:",
        len(crop_metadata.crops),
    )

    # --------------------------------------------------------
    # Generate crop embeddings
    # --------------------------------------------------------

    crop_embeddings = {}

    failed_crops = 0

    all_crops = list(
        crop_metadata.crops
    )

    for start in range(
        0,
        len(all_crops),
        batch_size,
    ):

        batch_records = all_crops[
            start:
            start + batch_size
        ]

        tensors = []
        valid_records = []

        for crop in batch_records:

            image = load_crop_image(
                crop.crop_path
            )

            if image is None:

                failed_crops += 1

                continue

            try:

                tensor = transform(
                    image
                )

                tensors.append(
                    tensor
                )

                valid_records.append(
                    crop
                )

            except Exception:

                failed_crops += 1

        if not tensors:
            continue

        batch = torch.stack(
            tensors
        )

        if device != "cpu":

            batch = batch.to(
                device
            )

        with torch.no_grad():

            features = model(
                batch
            )

        if isinstance(
            features,
            tuple,
        ):

            features = features[0]

        features = (
            features
            .detach()
            .cpu()
            .numpy()
        )

        for crop, feature in zip(
            valid_records,
            features,
        ):

            feature = l2_normalize(
                feature.astype(
                    np.float32
                )
            )

            crop_embeddings[
                crop.crop_id
            ] = feature

    # --------------------------------------------------------
    # Aggregate crop embeddings by track
    # --------------------------------------------------------

    track_embeddings = {}

    representative_crops = {}

    for track_id, crops in (
        crops_by_track.items()
    ):

        vectors = []

        valid_crop_records = []

        for crop in crops:

            vector = crop_embeddings.get(
                crop.crop_id
            )

            if vector is None:
                continue

            vectors.append(
                vector
            )

            valid_crop_records.append(
                crop
            )

        if not vectors:
            continue

        matrix = np.stack(
            vectors
        )

        track_vector = np.mean(
            matrix,
            axis=0,
        )

        track_vector = l2_normalize(
            track_vector
        )

        track_embeddings[
            track_id
        ] = track_vector.tolist()

        # Representative crop:
        # choose the middle valid observation.
        representative = (
            valid_crop_records[
                len(valid_crop_records) // 2
            ]
        )

        representative_crops[
            track_id
        ] = representative.crop_path

    # --------------------------------------------------------
    # Save output
    # --------------------------------------------------------

    output_json = Path(
        output_json
    )

    output_json.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_data = {
        "model": {
            "name":
                "osnet_x1_0",

            "training_dataset":
                "Market1501",

            "embedding_dimension":
                EMBEDDING_DIMENSION,

            "checkpoint":
                str(checkpoint_path),
        },

        "video_id":
            crop_metadata.video_id,

        "video_filename":
            crop_metadata.video_filename,

        "track_count":
            len(track_embeddings),

        "crop_count":
            len(
                crop_metadata.crops
            ),

        "failed_crops":
            failed_crops,

        "embeddings": [
            {
                "track_id":
                    track_id,

                "embedding":
                    embedding,

                "embedding_dimension":
                    len(embedding),

                "representative_crop":
                    representative_crops.get(
                        track_id
                    ),
            }

            for track_id, embedding
            in sorted(
                track_embeddings.items()
            )
        ],
    }

    with open(
        output_json,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            output_data,
            file,
            indent=2,
        )

    print(
        "Track embeddings generated:",
        len(track_embeddings),
    )

    print(
        "Failed crops:",
        failed_crops,
    )

    print(
        "Embedding dimension:",
        EMBEDDING_DIMENSION,
    )

    print(
        "Saved:",
        output_json,
    )

    return (
        track_embeddings,
        representative_crops,
    )