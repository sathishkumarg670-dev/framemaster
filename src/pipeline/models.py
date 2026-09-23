"""
VISIONFORGE AI - Pipeline Models & Data Schemas
Defines structured schemas for observations, crops, attributes,
canonical track profiles, and investigation readiness.
"""

from typing import List, Dict, Optional
from pydantic import BaseModel, Field
from pathlib import Path
import json


class Observation(BaseModel):
    """A single detected and tracked bounding box instance in a frame."""

    frame_index: int
    timestamp_seconds: float
    track_id: int

    bbox: List[int] = Field(
        description="[x1, y1, x2, y2]"
    )

    confidence: float
    class_id: int = 0


class VideoTrackingResult(BaseModel):
    """Complete detection/tracking result for a single video."""

    video_id: str
    video_filename: str
    fps: float
    total_frames: int
    processed_frames: int

    unique_tracks: List[int]

    total_observations: int

    observations: List[Observation]

    def save_json(self, path: Path):
        path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        with open(
            path,
            "w",
            encoding="utf-8"
        ) as f:
            f.write(
                self.model_dump_json(
                    indent=2
                )
            )

    @classmethod
    def load_json(
        cls,
        path: Path
    ) -> "VideoTrackingResult":

        with open(
            path,
            "r",
            encoding="utf-8"
        ) as f:
            data = json.load(f)

        return cls(**data)


class CropRecord(BaseModel):
    """Metadata describing one extracted person crop."""

    crop_id: str
    crop_filename: str
    crop_path: str

    video_id: str
    track_id: int

    frame_index: int
    timestamp_seconds: float

    bbox: List[int]

    width: int
    height: int


class CropMetadataCollection(BaseModel):
    """Collection of crops extracted for a video."""

    video_id: str
    video_filename: str

    total_crops: int

    unique_tracks: List[int]

    crops: List[CropRecord]

    def save_json(self, path: Path):
        path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        with open(
            path,
            "w",
            encoding="utf-8"
        ) as f:
            f.write(
                self.model_dump_json(
                    indent=2
                )
            )

    @classmethod
    def load_json(
        cls,
        path: Path
    ) -> "CropMetadataCollection":

        with open(
            path,
            "r",
            encoding="utf-8"
        ) as f:
            data = json.load(f)

        return cls(**data)


class TrackAttributes(BaseModel):
    """Aggregated visual appearance attributes for a track."""

    track_id: int

    upper_body_color: str

    lower_body_color: str

    color_distribution: Dict[str, float] = Field(
        default_factory=dict
    )

    sample_crops: int = 0

    backpack: Optional[str] = None
    headwear: Optional[str] = None
    accessories: List[str] = Field(default_factory=list)
    object_type: Optional[str] = None
    object_color: Optional[str] = None
    brand: Optional[str] = None
    distinctive_feature: Optional[str] = None


class TrackProfile(BaseModel):
    """
    Canonical profile for one tracked identity.

    A profile may contain only tracking metadata,
    or may contain complete visual evidence
    suitable for Re-ID investigation.
    """

    video_id: str
    track_id: int

    first_seen_frame: int
    last_seen_frame: int

    first_seen_sec: float
    last_seen_sec: float

    duration_sec: float

    total_observations: int
    total_crops: int

    attributes: Optional[TrackAttributes] = None

    embedding: Optional[List[float]] = None

    embedding_dimension: int = 512

    representative_crop: Optional[str] = None

    # -------------------------------------------------
    # M11 canonical readiness metadata
    # -------------------------------------------------

    has_crops: bool = False

    has_attributes: bool = False

    has_embedding: bool = False

    investigation_ready: bool = False

    profile_state: str = "TRACK_ONLY"

    def model_post_init(self, __context):
        self.update_readiness()

    def update_readiness(self):
        """
        Determine whether this track contains enough information
        to participate in appearance-based investigation.
        """

        self.has_crops = (
            self.total_crops > 0
        )

        self.has_attributes = (
            self.attributes is not None
        )

        self.has_embedding = (
            self.embedding is not None
            and len(self.embedding) > 0
        )

        self.investigation_ready = (
            self.has_crops
            and self.has_attributes
            and self.has_embedding
        )

        if self.investigation_ready:
            self.profile_state = "REID_READY"
        else:
            self.profile_state = "TRACK_ONLY"