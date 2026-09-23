import json
from pathlib import Path


ROOT = Path("data/unified")


def load_json(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main():
    print()
    print("=" * 70)
    print("VISIONFORGE AI - M11.6 UNIFIED DATASET VALIDATION")
    print("=" * 70)

    profile_files = sorted(
        ROOT.glob("profiles/*/canonical_track_profiles.json")
    )

    reid_files = sorted(
        ROOT.glob("reid/*/market1501_track_embeddings.json")
    )

    if not profile_files:
        print("ERROR: No canonical profile files found.")
        return

    print()
    print("CANONICAL PROFILES")
    print("-" * 70)

    total_profiles = 0
    total_with_crops = 0
    total_with_attributes = 0
    total_with_embeddings = 0
    total_ready = 0

    for path in profile_files:
        camera = path.parent.name
        profiles = load_json(path)

        camera_crops = 0
        camera_attrs = 0
        camera_embeddings = 0
        camera_ready = 0

        for profile in profiles:
            total_profiles += 1

            has_crops = profile.get("total_crops", 0) > 0
            has_attributes = profile.get("attributes") is not None
            has_embedding = (
                profile.get("embedding") is not None
                and len(profile.get("embedding", [])) > 0
            )

            if has_crops:
                camera_crops += 1
                total_with_crops += 1

            if has_attributes:
                camera_attrs += 1
                total_with_attributes += 1

            if has_embedding:
                camera_embeddings += 1
                total_with_embeddings += 1

            if has_crops and has_attributes and has_embedding:
                camera_ready += 1
                total_ready += 1

        print(
            f"{camera}: "
            f"{len(profiles)} profiles | "
            f"{camera_crops} crops | "
            f"{camera_attrs} attributes | "
            f"{camera_embeddings} embeddings | "
            f"{camera_ready} investigation-ready"
        )

    print()
    print("RE-ID FILES")
    print("-" * 70)

    total_reid_entries = 0

    for path in reid_files:
        camera = path.parent.name
        data = load_json(path)

        if isinstance(data, dict):
            entries = data.get("tracks", data.get("track_embeddings", data))
            if isinstance(entries, dict):
                count = len(entries)
            elif isinstance(entries, list):
                count = len(entries)
            else:
                count = 0
        elif isinstance(data, list):
            count = len(data)
        else:
            count = 0

        total_reid_entries += count

        print(f"{camera}: {count} Re-ID entries")

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"Total canonical profiles : {total_profiles}")
    print(f"Profiles with crops      : {total_with_crops}")
    print(f"Profiles with attributes : {total_with_attributes}")
    print(f"Profiles with embeddings : {total_with_embeddings}")
    print(f"Investigation-ready      : {total_ready}")
    print(f"Re-ID file entries       : {total_reid_entries}")

    print()
    print("PROFILE READINESS")
    print("-" * 70)

    track_only = total_profiles - total_with_crops
    attribute_ready = total_with_crops - total_with_attributes
    embedding_missing = total_with_crops - total_with_embeddings

    print(f"Track-only profiles      : {track_only}")
    print(f"Missing attributes       : {attribute_ready}")
    print(f"Missing embeddings       : {embedding_missing}")

    print()
    print("M11.6 STATUS")
    print("-" * 70)

    if total_profiles == 410 and total_ready == 235:
        print("PASS: Unified dataset structure matches expected results.")
        print("PASS: 410 tracked profiles identified.")
        print("PASS: 235 profiles are investigation-ready.")
    else:
        print("REVIEW REQUIRED: Counts differ from the expected dataset.")

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()