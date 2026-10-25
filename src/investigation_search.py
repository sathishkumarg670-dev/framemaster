"""
VISIONFORGE AI - Unified Investigation Search Engine

M10.12

Searches the unified canonical profile dataset and produces
investigation-enriched candidate evidence.

Supported search modes:
    1. Search by clothing attributes
    2. Search by camera
    3. Search by track
    4. Search by reference track
    5. Combined attribute + camera filtering
    6. Cross-camera candidate retrieval
    7. Enriched investigation candidate generation

Important:
    Results are candidate evidence associations.
    They are NOT identity confirmations.

Identity confirmation is intentionally never performed
automatically by this module.
"""

from pathlib import Path
from typing import Dict, List, Optional
import argparse
import json

import numpy as np

from src.pipeline.models import TrackProfile


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PROFILE_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "unified"
    / "profiles"
)


# ============================================================
# PROFILE LOADING
# ============================================================

def discover_profile_files() -> List[Path]:
    """Discover all unified canonical profile files."""

    if not PROFILE_DIRECTORY.exists():
        return []

    return sorted(
        PROFILE_DIRECTORY.glob(
            "*/canonical_track_profiles.json"
        )
    )


def load_profiles_for_camera(
    profile_file: Path
) -> List[TrackProfile]:
    """Load canonical profiles for one camera."""

    with open(
        profile_file,
        "r",
        encoding="utf-8"
    ) as file:
        data = json.load(file)

    profiles = []

    for item in data:

        profile = TrackProfile(
            **item
        )

        if hasattr(
            profile,
            "update_readiness"
        ):
            profile.update_readiness()

        profiles.append(profile)

    return profiles


def load_all_profiles() -> Dict[
    str,
    List[TrackProfile]
]:
    """Load all canonical profiles."""

    profiles_by_camera = {}

    for profile_file in discover_profile_files():

        camera_id = profile_file.parent.name

        profiles_by_camera[
            camera_id
        ] = load_profiles_for_camera(
            profile_file
        )

    return profiles_by_camera


# ============================================================
# PROFILE READINESS
# ============================================================

def is_ready(
    profile: TrackProfile
) -> bool:
    """Return whether profile is ready for Re-ID investigation."""

    if hasattr(
        profile,
        "investigation_ready"
    ):
        return profile.investigation_ready

    return (
        profile.total_crops > 0
        and
        profile.attributes is not None
        and
        profile.embedding is not None
        and
        len(profile.embedding) > 0
    )


# ============================================================
# EMBEDDING SIMILARITY
# ============================================================

def cosine_similarity(
    embedding_a: List[float],
    embedding_b: List[float]
) -> float:
    """Calculate cosine similarity."""

    if embedding_a is None or embedding_b is None:
        return 0.0

    vector_a = np.asarray(
        embedding_a,
        dtype=np.float32
    )

    vector_b = np.asarray(
        embedding_b,
        dtype=np.float32
    )

    if (
        vector_a.size == 0
        or
        vector_b.size == 0
    ):
        return 0.0

    if vector_a.shape != vector_b.shape:
        return 0.0

    norm_a = np.linalg.norm(
        vector_a
    )

    norm_b = np.linalg.norm(
        vector_b
    )

    if (
        norm_a == 0
        or
        norm_b == 0
    ):
        return 0.0

    return float(
        np.dot(
            vector_a,
            vector_b
        )
        /
        (
            norm_a
            *
            norm_b
        )
    )


# ============================================================
# SAFE VALUE HELPERS
# ============================================================

def normalize_sample_crops(value) -> List:
    """
    Normalize sample_crops safely.

    Depending on the canonical profile data, sample_crops may
    contain a list, tuple, string, integer, float, or None.

    Only list/tuple/string values are treated as crop references.
    Numeric values are ignored because they are not crop paths.
    """

    if value is None:
        return []

    if isinstance(value, list):
        return value

    if isinstance(value, tuple):
        return list(value)

    if isinstance(value, str):
        return [value]

    return []


def safe_attribute_value(
    attributes,
    name: str
):
    """Safely retrieve an attribute field."""

    if attributes is None:
        return None

    return getattr(
        attributes,
        name,
        None
    )


# ============================================================
# ATTRIBUTE MATCHING
# ============================================================

def attribute_match(
    profile: TrackProfile,
    upper_body_color: Optional[str] = None,
    lower_body_color: Optional[str] = None
) -> dict:
    """Compare requested clothing attributes with a profile."""

    attributes = profile.attributes

    if attributes is None:

        return {
            "matched": False,
            "score": None,
            "compared": 0,
            "matches": 0
        }

    compared = 0
    matches = 0

    # --------------------------------------------------------
    # Upper body
    # --------------------------------------------------------

    if (
        upper_body_color
        and
        upper_body_color.lower()
        != "unknown"
    ):

        profile_color = safe_attribute_value(
            attributes,
            "upper_body_color"
        )

        if (
            profile_color
            and
            str(profile_color).lower()
            != "unknown"
        ):

            compared += 1

            if (
                str(profile_color).lower()
                ==
                upper_body_color.lower()
            ):
                matches += 1

    # --------------------------------------------------------
    # Lower body
    # --------------------------------------------------------

    if (
        lower_body_color
        and
        lower_body_color.lower()
        != "unknown"
    ):

        profile_color = safe_attribute_value(
            attributes,
            "lower_body_color"
        )

        if (
            profile_color
            and
            str(profile_color).lower()
            != "unknown"
        ):

            compared += 1

            if (
                str(profile_color).lower()
                ==
                lower_body_color.lower()
            ):
                matches += 1

    # --------------------------------------------------------
    # Score
    # --------------------------------------------------------

    if compared == 0:

        score = None

    else:

        score = round(
            matches / compared,
            6
        )

    return {
        "matched": (
            compared > 0
            and
            matches == compared
        ),
        "score": score,
        "compared": compared,
        "matches": matches
    }


# ============================================================
# PROFILE ATTRIBUTE COMPARISON
# ============================================================

def compare_profile_attributes(
    source_profile: TrackProfile,
    target_profile: TrackProfile
) -> dict:
    """Compare clothing attributes between two profiles."""

    source_attributes = source_profile.attributes
    target_attributes = target_profile.attributes

    if (
        source_attributes is None
        or
        target_attributes is None
    ):

        return {
            "score": None,
            "upper_body_match": None,
            "lower_body_match": None,
            "compared": 0,
            "matches": 0
        }

    source_upper = safe_attribute_value(
        source_attributes,
        "upper_body_color"
    )

    source_lower = safe_attribute_value(
        source_attributes,
        "lower_body_color"
    )

    target_upper = safe_attribute_value(
        target_attributes,
        "upper_body_color"
    )

    target_lower = safe_attribute_value(
        target_attributes,
        "lower_body_color"
    )

    compared = 0
    matches = 0

    upper_match = None
    lower_match = None

    # --------------------------------------------------------
    # Upper body
    # --------------------------------------------------------

    if (
        source_upper
        and
        target_upper
        and
        str(source_upper).lower() != "unknown"
        and
        str(target_upper).lower() != "unknown"
    ):

        compared += 1

        upper_match = (
            str(source_upper).lower()
            ==
            str(target_upper).lower()
        )

        if upper_match:
            matches += 1

    # --------------------------------------------------------
    # Lower body
    # --------------------------------------------------------

    if (
        source_lower
        and
        target_lower
        and
        str(source_lower).lower() != "unknown"
        and
        str(target_lower).lower() != "unknown"
    ):

        compared += 1

        lower_match = (
            str(source_lower).lower()
            ==
            str(target_lower).lower()
        )

        if lower_match:
            matches += 1

    # --------------------------------------------------------
    # Score
    # --------------------------------------------------------

    if compared == 0:

        score = None

    else:

        score = round(
            matches / compared,
            6
        )

    return {
        "score": score,
        "upper_body_match": upper_match,
        "lower_body_match": lower_match,
        "compared": compared,
        "matches": matches
    }


# ============================================================
# TEMPORAL ANALYSIS
# ============================================================

def analyze_temporal_transition(
    source_profile: TrackProfile,
    target_profile: TrackProfile
) -> dict:
    """
    Analyze whether the target track occurs after the source track.

    Important:
        These timestamps are video-relative timestamps.
        They are not wall-clock timestamps unless the videos
        are known to be synchronized.
    """

    source_last = source_profile.last_seen_sec
    target_first = target_profile.first_seen_sec

    if (
        source_last is None
        or
        target_first is None
    ):

        return {
            "valid": False,
            "time_gap_sec": None,
            "source_last_seen_sec": source_last,
            "target_first_seen_sec": target_first
        }

    gap = (
        target_first
        -
        source_last
    )

    return {
        "valid": gap >= 0,
        "time_gap_sec": round(
            gap,
            6
        ),
        "source_last_seen_sec": source_last,
        "target_first_seen_sec": target_first
    }


# ============================================================
# CANDIDATE QUALITY
# ============================================================

def calculate_candidate_quality(
    appearance_similarity: float,
    attribute_score: Optional[float],
    temporal_valid: bool
) -> dict:
    """
    Produce descriptive candidate quality metadata.

    This is NOT an identity probability and NOT a confidence
    threshold for identity confirmation.
    """

    appearance_component = max(
        0.0,
        min(
            1.0,
            appearance_similarity
        )
    )

    if attribute_score is None:

        attribute_component = 0.0

    else:

        attribute_component = max(
            0.0,
            min(
                1.0,
                attribute_score
            )
        )

    temporal_component = (
        1.0
        if temporal_valid
        else 0.0
    )

    ranking_score = (
        appearance_component * 0.60
        +
        attribute_component * 0.25
        +
        temporal_component * 0.15
    )

    return {
        "ranking_score": round(
            ranking_score,
            6
        ),
        "appearance_component": round(
            appearance_component,
            6
        ),
        "attribute_component": round(
            attribute_component,
            6
        ),
        "temporal_component": round(
            temporal_component,
            6
        ),
        "interpretation": (
            "candidate_evidence_only"
        )
    }


# ============================================================
# EVIDENCE REFERENCES
# ============================================================

def build_evidence_references(
    source_profile: TrackProfile,
    target_profile: TrackProfile
) -> dict:
    """
    Build safe references to available representative/sample crops.

    The canonical profile schema may contain sample_crops in
    different forms depending on the profile that was generated.

    Supported:
        - list
        - tuple
        - string
        - None

    Numeric/unexpected values are ignored.

    These references are evidence pointers only.
    They are not identity confirmation.
    """

    source_samples = []
    target_samples = []

    # --------------------------------------------------------
    # Source samples
    # --------------------------------------------------------

    if source_profile.attributes:

        source_samples = normalize_sample_crops(
            safe_attribute_value(
                source_profile.attributes,
                "sample_crops"
            )
        )

    # --------------------------------------------------------
    # Target samples
    # --------------------------------------------------------

    if target_profile.attributes:

        target_samples = normalize_sample_crops(
            safe_attribute_value(
                target_profile.attributes,
                "sample_crops"
            )
        )

    return {
        "source_representative_crop":
            source_profile.representative_crop,

        "target_representative_crop":
            target_profile.representative_crop,

        "source_sample_crops":
            source_samples,

        "target_sample_crops":
            target_samples
    }


# ============================================================
# BUILD ENRICHED CANDIDATE
# ============================================================

def build_enriched_candidate(
    source_camera: str,
    source_profile: TrackProfile,
    target_camera: str,
    target_profile: TrackProfile
) -> dict:
    """Build a complete investigation candidate record."""

    similarity = cosine_similarity(
        source_profile.embedding,
        target_profile.embedding
    )

    attributes = compare_profile_attributes(
        source_profile,
        target_profile
    )

    temporal = analyze_temporal_transition(
        source_profile,
        target_profile
    )

    quality = calculate_candidate_quality(
        similarity,
        attributes["score"],
        temporal["valid"]
    )

    evidence = build_evidence_references(
        source_profile,
        target_profile
    )

    source_attributes = None
    target_attributes = None

    if source_profile.attributes:

        source_attributes = {
            "upper_body_color":
                safe_attribute_value(
                    source_profile.attributes,
                    "upper_body_color"
                ),

            "lower_body_color":
                safe_attribute_value(
                    source_profile.attributes,
                    "lower_body_color"
                ),

            "color_distribution":
                safe_attribute_value(
                    source_profile.attributes,
                    "color_distribution"
                )
        }

    if target_profile.attributes:

        target_attributes = {
            "upper_body_color":
                safe_attribute_value(
                    target_profile.attributes,
                    "upper_body_color"
                ),

            "lower_body_color":
                safe_attribute_value(
                    target_profile.attributes,
                    "lower_body_color"
                ),

            "color_distribution":
                safe_attribute_value(
                    target_profile.attributes,
                    "color_distribution"
                )
        }

    return {

        # ----------------------------------------------------
        # Candidate identity
        # ----------------------------------------------------

        "candidate_type":
            "cross_camera_person_association",

        "source_camera":
            source_camera,

        "source_track_id":
            source_profile.track_id,

        "target_camera":
            target_camera,

        "target_track_id":
            target_profile.track_id,

        # ----------------------------------------------------
        # Appearance
        # ----------------------------------------------------

        "appearance_similarity":
            round(
                similarity,
                6
            ),

        # ----------------------------------------------------
        # Attributes
        # ----------------------------------------------------

        "attribute_score":
            attributes["score"],

        "attribute_comparison": {
            "compared":
                attributes["compared"],

            "matches":
                attributes["matches"],

            "upper_body_match":
                attributes["upper_body_match"],

            "lower_body_match":
                attributes["lower_body_match"]
        },

        "source_attributes":
            source_attributes,

        "target_attributes":
            target_attributes,

        # ----------------------------------------------------
        # Temporal
        # ----------------------------------------------------

        "temporal_analysis":
            temporal,

        # ----------------------------------------------------
        # Evidence
        # ----------------------------------------------------

        "evidence":
            evidence,

        # ----------------------------------------------------
        # Candidate ranking
        # ----------------------------------------------------

        "candidate_quality":
            quality,

        # ----------------------------------------------------
        # Profile information
        # ----------------------------------------------------

        "source_profile": {
            "first_seen_sec":
                source_profile.first_seen_sec,

            "last_seen_sec":
                source_profile.last_seen_sec,

            "duration_sec":
                source_profile.duration_sec,

            "total_observations":
                source_profile.total_observations,

            "total_crops":
                source_profile.total_crops
        },

        "target_profile": {
            "first_seen_sec":
                target_profile.first_seen_sec,

            "last_seen_sec":
                target_profile.last_seen_sec,

            "duration_sec":
                target_profile.duration_sec,

            "total_observations":
                target_profile.total_observations,

            "total_crops":
                target_profile.total_crops
        },

        # ----------------------------------------------------
        # Identity decision
        # ----------------------------------------------------

        "identity_confirmed":
            False,

        "identity_confirmation_reason":
            (
                "Automatic identity confirmation is not "
                "performed. This record represents candidate "
                "evidence for investigator review."
            )
    }


# ============================================================
# SEARCH BY ATTRIBUTES
# ============================================================

def search_by_attributes(
    profiles_by_camera: Dict[
        str,
        List[TrackProfile]
    ],
    upper_body_color: Optional[str] = None,
    lower_body_color: Optional[str] = None,
    backpack: Optional[str] = None,
    headwear: Optional[str] = None,
    camera_id: Optional[str] = None,
    ready_only: bool = True
) -> List[dict]:
    """
    Search profiles using investigator-specified clothing
    attributes.

    Matching behavior:

        - If only upper color is supplied:
              upper color must match.

        - If only lower color is supplied:
              lower color must match.

        - If both are supplied:
              BOTH must match.

    This is strict/exact attribute matching.

    Results remain candidate evidence only.
    """

    results = []

    # --------------------------------------------------------
    # Normalize requested attributes
    # --------------------------------------------------------

    requested_upper = (
        upper_body_color.strip().lower()
        if upper_body_color
        else None
    )

    requested_lower = (
        lower_body_color.strip().lower()
        if lower_body_color
        else None
    )

    requested_backpack = (
        backpack.strip().lower()
        if backpack
        else None
    )

    requested_headwear = (
        headwear.strip().lower()
        if headwear
        else None
    )

    # --------------------------------------------------------
    # Search cameras
    # --------------------------------------------------------

    for current_camera, profiles in (
        profiles_by_camera.items()
    ):

        if (
            camera_id
            and
            current_camera != camera_id
        ):
            continue

        # ----------------------------------------------------
        # Search profiles
        # ----------------------------------------------------

        for profile in profiles:

            if (
                ready_only
                and
                not is_ready(profile)
            ):
                continue

            if profile.attributes is None:
                continue

            profile_upper = safe_attribute_value(
                profile.attributes,
                "upper_body_color"
            )

            profile_lower = safe_attribute_value(
                profile.attributes,
                "lower_body_color"
            )

            profile_upper = (
                str(profile_upper).strip().lower()
                if profile_upper
                else None
            )

            profile_lower = (
                str(profile_lower).strip().lower()
                if profile_lower
                else None
            )

            # ------------------------------------------------
            # Upper matching
            # ------------------------------------------------

            upper_match = None

            if requested_upper:

                upper_match = (
                    profile_upper == requested_upper
                )

            # ------------------------------------------------
            # Lower matching
            # ------------------------------------------------

            lower_match = None

            if requested_lower:

                lower_match = (
                    profile_lower == requested_lower
                )

            # ------------------------------------------------
            # Optional accessory matching
            # ------------------------------------------------

            profile_backpack = safe_attribute_value(
                profile.attributes,
                "backpack"
            )
            profile_headwear = safe_attribute_value(
                profile.attributes,
                "headwear"
            )

            backpack_match = None
            headwear_match = None

            if requested_backpack:
                backpack_match = (
                    str(profile_backpack).strip().lower()
                    == requested_backpack
                )

            if requested_headwear:
                headwear_match = (
                    str(profile_headwear).strip().lower()
                    == requested_headwear
                )

            # ------------------------------------------------
            # Strict matching
            # ------------------------------------------------

            if (
                requested_upper
                and
                not upper_match
            ):
                continue

            if (
                requested_lower
                and
                not lower_match
            ):
                continue

            if (
                requested_backpack
                and
                not backpack_match
            ):
                continue

            if (
                requested_headwear
                and
                not headwear_match
            ):
                continue

            # ------------------------------------------------
            # Calculate score
            # ------------------------------------------------

            compared = 0
            matches = 0

            if requested_upper:

                compared += 1

                if upper_match:
                    matches += 1

            if requested_lower:

                compared += 1

                if lower_match:
                    matches += 1

            if requested_backpack:

                compared += 1

                if backpack_match:
                    matches += 1

            if requested_headwear:

                compared += 1

                if headwear_match:
                    matches += 1

            # No attribute requested: browse every profile that has
            # attributes, without claiming a match score.
            attribute_score = (
                round(
                    matches / compared,
                    6
                )
                if compared
                else None
            )

            # ------------------------------------------------
            # Result
            # ------------------------------------------------

            results.append({

                "camera_id":
                    current_camera,

                "track_id":
                    profile.track_id,

                "profile_state":
                    (
                        "REID_READY"
                        if is_ready(profile)
                        else "TRACK_ONLY"
                    ),

                "investigation_ready":
                    is_ready(profile),

                # --------------------------------------------
                # Actual attributes
                # --------------------------------------------

                "upper_body_color":
                    safe_attribute_value(
                        profile.attributes,
                        "upper_body_color"
                    ),

                "lower_body_color":
                    safe_attribute_value(
                        profile.attributes,
                        "lower_body_color"
                    ),

                "backpack":
                    safe_attribute_value(
                        profile.attributes,
                        "backpack"
                    ),

                "headwear":
                    safe_attribute_value(
                        profile.attributes,
                        "headwear"
                    ),

                "requested_attributes": {

                    "upper_body_color":
                        upper_body_color,

                    "lower_body_color":
                        lower_body_color,

                    "backpack":
                        backpack,

                    "headwear":
                        headwear
                },

                # --------------------------------------------
                # Matching information
                # --------------------------------------------

                "upper_body_match":
                    upper_match,

                "lower_body_match":
                    lower_match,

                "attribute_score":
                    attribute_score,

                "matched_attributes":
                    matches,

                "compared_attributes":
                    compared,

                "match_mode":
                    "strict" if compared else "browse",

                # --------------------------------------------
                # Timeline
                # --------------------------------------------

                "first_seen_sec":
                    profile.first_seen_sec,

                "last_seen_sec":
                    profile.last_seen_sec,

                "duration_sec":
                    profile.duration_sec,

                # --------------------------------------------
                # Evidence
                # --------------------------------------------

                "representative_crop":
                    profile.representative_crop,

                # --------------------------------------------
                # Identity
                # --------------------------------------------

                "identity_confirmed":
                    False
            })

    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    results.sort(
        key=lambda item: (
            -(item["attribute_score"] or 0),
            -item["matched_attributes"],
            item["camera_id"],
            item["first_seen_sec"],
        )
    )

    return results


# ============================================================
# ATTRIBUTE SUMMARY
# ============================================================

SUMMARY_ATTRIBUTES = (
    "upper_body_color",
    "lower_body_color",
    "backpack",
    "headwear",
)


def attribute_summary(
    profiles_by_camera: Dict[
        str,
        List[TrackProfile]
    ],
) -> dict:
    """
    Count the attribute values actually present in processed
    profiles, overall and per camera/video.

    An attribute with no values is not produced by the current
    pipeline, so the investigator should not filter on it.
    """

    def empty():
        return {
            name: {}
            for name in SUMMARY_ATTRIBUTES
        }

    overall = empty()
    by_camera = {}

    for camera_id, profiles in profiles_by_camera.items():

        camera_counts = empty()
        profile_count = 0

        for profile in profiles:

            if profile.attributes is None:
                continue

            profile_count += 1

            for name in SUMMARY_ATTRIBUTES:

                value = safe_attribute_value(
                    profile.attributes,
                    name
                )

                if value is None or str(value).strip() == "":
                    continue

                value = str(value).strip().lower()

                for counts in (
                    camera_counts[name],
                    overall[name],
                ):
                    counts[value] = counts.get(value, 0) + 1

        by_camera[camera_id] = {
            "profiles_with_attributes": profile_count,
            "attributes": camera_counts,
        }

    return {
        "profiles_with_attributes": sum(
            item["profiles_with_attributes"]
            for item in by_camera.values()
        ),
        "attributes": overall,
        "available_attributes": [
            name
            for name in SUMMARY_ATTRIBUTES
            if overall[name]
        ],
        "by_camera": by_camera,
    }

# ============================================================
# SEARCH BY CAMERA
# ============================================================

def search_by_camera(
    profiles_by_camera: Dict[
        str,
        List[TrackProfile]
    ],
    camera_id: str,
    ready_only: bool = False
) -> List[dict]:

    profiles = profiles_by_camera.get(
        camera_id,
        []
    )

    results = []

    for profile in profiles:

        if (
            ready_only
            and
            not is_ready(profile)
        ):
            continue

        results.append(
            profile_to_dict(
                camera_id,
                profile
            )
        )

    return results


# ============================================================
# SEARCH BY TRACK
# ============================================================

def search_by_track(
    profiles_by_camera: Dict[
        str,
        List[TrackProfile]
    ],
    camera_id: str,
    track_id: int
) -> Optional[dict]:

    profiles = profiles_by_camera.get(
        camera_id,
        []
    )

    for profile in profiles:

        if profile.track_id == track_id:

            return profile_to_dict(
                camera_id,
                profile
            )

    return None


# ============================================================
# REFERENCE TRACK SEARCH
# ============================================================

def search_by_reference_track(
    profiles_by_camera: Dict[
        str,
        List[TrackProfile]
    ],
    camera_id: str,
    track_id: int,
    target_camera: Optional[str] = None,
    limit: int = 20
) -> List[dict]:
    """
    Find visually similar tracks to a reference track.

    Returns enriched investigation candidates.
    """

    source_profile = None

    for profile in profiles_by_camera.get(
        camera_id,
        []
    ):

        if profile.track_id == track_id:

            source_profile = profile

            break

    if (
        source_profile is None
        or
        not is_ready(source_profile)
        or
        source_profile.embedding is None
    ):
        return []

    results = []

    cameras = (
        [target_camera]
        if target_camera
        else list(
            profiles_by_camera.keys()
        )
    )

    for current_camera in cameras:

        if current_camera == camera_id:
            continue

        for profile in profiles_by_camera.get(
            current_camera,
            []
        ):

            if not is_ready(profile):
                continue

            if profile.embedding is None:
                continue

            candidate = build_enriched_candidate(
                source_camera=camera_id,
                source_profile=source_profile,
                target_camera=current_camera,
                target_profile=profile
            )

            results.append(
                candidate
            )

    # --------------------------------------------------------
    # Ranking
    # --------------------------------------------------------

    results.sort(
        key=lambda item: (
            item[
                "candidate_quality"
            ][
                "ranking_score"
            ],

            item[
                "appearance_similarity"
            ],

            (
                item["attribute_score"]
                if item["attribute_score"]
                is not None
                else -1
            )
        ),
        reverse=True
    )

    return results[:limit]


# ============================================================
# PROFILE SERIALIZATION
# ============================================================

def profile_to_dict(
    camera_id: str,
    profile: TrackProfile
) -> dict:

    attributes = None

    if profile.attributes:

        attributes = {

            "upper_body_color":
                safe_attribute_value(
                    profile.attributes,
                    "upper_body_color"
                ),

            "lower_body_color":
                safe_attribute_value(
                    profile.attributes,
                    "lower_body_color"
                ),

            "color_distribution":
                safe_attribute_value(
                    profile.attributes,
                    "color_distribution"
                ),

            "sample_crops":
                normalize_sample_crops(
                    safe_attribute_value(
                        profile.attributes,
                        "sample_crops"
                    )
                )
        }

    return {

        "camera_id":
            camera_id,

        "track_id":
            profile.track_id,

        "profile_state":
            (
                "REID_READY"
                if is_ready(profile)
                else "TRACK_ONLY"
            ),

        "investigation_ready":
            is_ready(profile),

        "first_seen_frame":
            profile.first_seen_frame,

        "last_seen_frame":
            profile.last_seen_frame,

        "first_seen_sec":
            profile.first_seen_sec,

        "last_seen_sec":
            profile.last_seen_sec,

        "duration_sec":
            profile.duration_sec,

        "total_observations":
            profile.total_observations,

        "total_crops":
            profile.total_crops,

        "attributes":
            attributes,

        "embedding_dimension":
            profile.embedding_dimension,

        "representative_crop":
            profile.representative_crop,

        "identity_confirmed":
            False
    }


# ============================================================
# STATISTICS
# ============================================================

def dataset_statistics(
    profiles_by_camera: Dict[
        str,
        List[TrackProfile]
    ]
) -> dict:

    total = 0
    ready = 0
    track_only = 0

    camera_stats = {}

    for camera_id, profiles in (
        profiles_by_camera.items()
    ):

        camera_ready = sum(
            1
            for profile in profiles
            if is_ready(profile)
        )

        camera_total = len(
            profiles
        )

        camera_track_only = (
            camera_total
            -
            camera_ready
        )

        camera_stats[
            camera_id
        ] = {

            "total_profiles":
                camera_total,

            "reid_ready":
                camera_ready,

            "track_only":
                camera_track_only
        }

        total += camera_total
        ready += camera_ready
        track_only += camera_track_only

    return {

        "camera_count":
            len(profiles_by_camera),

        "total_profiles":
            total,

        "reid_ready":
            ready,

        "track_only":
            track_only,

        "cameras":
            camera_stats
    }


# ============================================================
# CONSOLE
# ============================================================

def print_search_results(
    results: List[dict]
) -> None:

    print()
    print("=" * 70)
    print(
        "VISIONFORGE AI - "
        "INVESTIGATION SEARCH RESULTS"
    )
    print("=" * 70)

    if not results:

        print()
        print("No matching profiles found.")
        return

    for index, result in enumerate(
        results,
        start=1
    ):

        camera = result.get(
            "camera_id",
            result.get(
                "target_camera",
                "unknown"
            )
        )

        track = result.get(
            "track_id",
            result.get(
                "target_track_id",
                "unknown"
            )
        )

        print()
        print(
            f"{index}. "
            f"{camera}/track_{track}"
        )

        print("-" * 70)

        # ----------------------------------------------------
        # PROFILE STATE
        # ----------------------------------------------------

        if "profile_state" in result:

            print(
                "   Profile state:",
                result.get(
                    "profile_state"
                )
            )

        if "investigation_ready" in result:

            print(
                "   Investigation ready:",
                result.get(
                    "investigation_ready"
                )
            )

        # ----------------------------------------------------
        # REFERENCE INFORMATION
        # ----------------------------------------------------

        if "source_camera" in result:

            print(
                "   Source:",
                f"{result['source_camera']}"
                f"/track_{result['source_track_id']}"
            )

        # ----------------------------------------------------
        # APPEARANCE / RE-ID
        # ----------------------------------------------------

        if "appearance_similarity" in result:

            print(
                "   Appearance similarity:",
                result[
                    "appearance_similarity"
                ]
            )

        # ----------------------------------------------------
        # ATTRIBUTES
        # ----------------------------------------------------

        if (
            "upper_body_color" in result
            or
            "lower_body_color" in result
        ):

            print()
            print("   ATTRIBUTES")

            print(
                "   Upper clothing:",
                result.get(
                    "upper_body_color",
                    "unknown"
                )
            )

            print(
                "   Lower clothing:",
                result.get(
                    "lower_body_color",
                    "unknown"
                )
            )

        # ----------------------------------------------------
        # ATTRIBUTE SCORE
        # ----------------------------------------------------

        if "attribute_score" in result:

            print()
            print(
                "   Attribute score:",
                result[
                    "attribute_score"
                ]
            )

            matched = result.get(
                "matched_attributes"
            )

            compared = result.get(
                "compared_attributes"
            )

            if (
                matched is not None
                and
                compared is not None
            ):

                print(
                    "   Attributes matched:",
                    f"{matched}/{compared}"
                )

        # ----------------------------------------------------
        # ATTRIBUTE COMPARISON
        # ----------------------------------------------------

        comparison = result.get(
            "attribute_comparison"
        )

        if comparison:

            print()

            print(
                "   Upper clothing match:",
                comparison.get(
                    "upper_body_match"
                )
            )

            print(
                "   Lower clothing match:",
                comparison.get(
                    "lower_body_match"
                )
            )

        # ----------------------------------------------------
        # TIMELINE
        # ----------------------------------------------------

        first_seen = result.get(
            "first_seen_sec"
        )

        last_seen = result.get(
            "last_seen_sec"
        )

        duration = result.get(
            "duration_sec"
        )

        if (
            first_seen is not None
            or
            last_seen is not None
        ):

            print()
            print("   TIMELINE")

            print(
                "   First seen:",
                first_seen,
                "sec"
            )

            print(
                "   Last seen:",
                last_seen,
                "sec"
            )

            if duration is not None:

                print(
                    "   Duration:",
                    duration,
                    "sec"
                )

        # ----------------------------------------------------
        # TEMPORAL ANALYSIS
        # ----------------------------------------------------

        temporal = result.get(
            "temporal_analysis"
        )

        if temporal:

            print()
            print(
                "   Temporal consistency:",
                temporal.get(
                    "valid"
                )
            )

            print(
                "   Source last seen:",
                temporal.get(
                    "source_last_seen_sec"
                ),
                "sec"
            )

            print(
                "   Target first seen:",
                temporal.get(
                    "target_first_seen_sec"
                ),
                "sec"
            )

            print(
                "   Time gap:",
                temporal.get(
                    "time_gap_sec"
                ),
                "sec"
            )

        # ----------------------------------------------------
        # CANDIDATE QUALITY
        # ----------------------------------------------------

        quality = result.get(
            "candidate_quality"
        )

        if quality:

            print()
            print(
                "   Candidate ranking score:",
                quality.get(
                    "ranking_score"
                )
            )

            print(
                "   Interpretation:",
                quality.get(
                    "interpretation",
                    "candidate_evidence_only"
                )
            )

        # ----------------------------------------------------
        # EVIDENCE
        # ----------------------------------------------------

        evidence = result.get(
            "evidence"
        )

        representative_crop = result.get(
            "representative_crop"
        )

        if evidence or representative_crop:

            print()
            print("   EVIDENCE")

            if representative_crop:

                print(
                    "   Representative crop:",
                    representative_crop
                )

            if evidence:

                source_crop = evidence.get(
                    "source_representative_crop"
                )

                target_crop = evidence.get(
                    "target_representative_crop"
                )

                if source_crop:

                    print(
                        "   Source crop:",
                        source_crop
                    )

                if target_crop:

                    print(
                        "   Target crop:",
                        target_crop
                    )

        # ----------------------------------------------------
        # IDENTITY
        # ----------------------------------------------------

        print()

        print(
            "   Identity confirmed:",
            result.get(
                "identity_confirmed",
                False
            )
        )

        print(
            "   NOTE: Candidate evidence "
            "requires investigator review."
        )

        print("=" * 70)
# ============================================================
# CLI
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "VISIONFORGE AI "
            "Unified Investigation Search"
        )
    )

    parser.add_argument(
        "--camera",
        type=str,
        help="Camera ID, e.g. camera_01"
    )

    parser.add_argument(
        "--track",
        type=int,
        help="Track ID"
    )

    parser.add_argument(
        "--reference-camera",
        type=str,
        help="Reference camera for Re-ID search"
    )

    parser.add_argument(
        "--reference-track",
        type=int,
        help="Reference track for Re-ID search"
    )

    parser.add_argument(
        "--upper",
        type=str,
        help="Upper-body clothing color"
    )

    parser.add_argument(
        "--lower",
        type=str,
        help="Lower-body clothing color"
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Maximum results"
    )
    
    parser.add_argument(
        "--query",
        action="store_true",
        help="Run investigator query mode"
    )

    parser.add_argument(
        "--target-type",
        type=str,
        default="person",
        choices=["person", "object"],
        help="Investigation target type"
    )
    
    args = parser.parse_args()

    profiles_by_camera = (
        load_all_profiles()
    )

    if not profiles_by_camera:

        print(
            "No unified profiles found."
        )

        print(
            "Expected:",
            PROFILE_DIRECTORY
        )

        return

    stats = dataset_statistics(
        profiles_by_camera
    )

    print()
    print("=" * 70)
    print(
        "VISIONFORGE AI - "
        "UNIFIED INVESTIGATION SEARCH"
    )
    print("=" * 70)

    print(
        "Cameras:",
        stats["camera_count"]
    )

    print(
        "Total profiles:",
        stats["total_profiles"]
    )

    print(
        "REID_READY:",
        stats["reid_ready"]
    )

    print(
        "TRACK_ONLY:",
        stats["track_only"]
    )

    # --------------------------------------------------------
    # Reference track search
    # --------------------------------------------------------

    if (
        args.reference_camera
        and
        args.reference_track is not None
    ):

        results = search_by_reference_track(
            profiles_by_camera,
            args.reference_camera,
            args.reference_track,
            target_camera=args.camera,
            limit=args.limit
        )

        print_search_results(
            results
        )

        return

    # --------------------------------------------------------
    # Track search
    # --------------------------------------------------------

    if (
        args.camera
        and
        args.track is not None
    ):

        result = search_by_track(
            profiles_by_camera,
            args.camera,
            args.track
        )

        if result:

            print_search_results(
                [result]
            )

        else:

            print(
                "Track not found."
            )

        return

    # --------------------------------------------------------
    # Attribute search
    # --------------------------------------------------------

    if (
        args.upper
        or
        args.lower
    ):

        results = search_by_attributes(
            profiles_by_camera,
            upper_body_color=args.upper,
            lower_body_color=args.lower,
            camera_id=args.camera
        )

        print_search_results(
            results[:args.limit]
        )

        return

    # --------------------------------------------------------
    # Camera listing
    # --------------------------------------------------------

    if args.camera:

        results = search_by_camera(
            profiles_by_camera,
            args.camera,
            ready_only=False
        )

        print_search_results(
            results[:args.limit]
        )

        return

    # --------------------------------------------------------
    # Default
    # --------------------------------------------------------

    print()
    print(
        "No search parameters supplied."
    )

    print()
    print(
        "Examples:"
    )

    print(
        "  python -m src.investigation_search "
        "--upper black"
    )

    print(
        "  python -m src.investigation_search "
        "--upper black --lower blue"
    )

    print(
        "  python -m src.investigation_search "
        "--camera camera_02"
    )

    print(
        "  python -m src.investigation_search "
        "--camera camera_02 --track 181"
    )

    print(
        "  python -m src.investigation_search "
        "--reference-camera camera_01 "
        "--reference-track 102"
    )


if __name__ == "__main__":
    main()