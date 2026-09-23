import json
from pathlib import Path


ROOT = Path(
    "data/unified/profiles"
)


def main():

    files = sorted(
        ROOT.glob(
            "*/canonical_track_profiles.json"
        )
    )

    if not files:
        print(
            "No canonical profile files found."
        )
        return

    total = 0
    ready_total = 0
    track_only_total = 0

    print()
    print(
        "VISIONFORGE AI - "
        "CANONICAL PROFILE UPGRADE"
    )

    print(
        "=" * 70
    )

    for path in files:

        with open(
            path,
            "r",
            encoding="utf-8"
        ) as f:
            profiles = json.load(f)

        ready_count = 0
        track_only_count = 0

        for profile in profiles:

            has_crops = (
                profile.get(
                    "total_crops",
                    0
                ) > 0
            )

            has_attributes = (
                profile.get(
                    "attributes"
                ) is not None
            )

            embedding = profile.get(
                "embedding"
            )

            has_embedding = (
                isinstance(
                    embedding,
                    list
                )
                and len(
                    embedding
                ) > 0
            )

            investigation_ready = (
                has_crops
                and has_attributes
                and has_embedding
            )

            profile[
                "has_crops"
            ] = has_crops

            profile[
                "has_attributes"
            ] = has_attributes

            profile[
                "has_embedding"
            ] = has_embedding

            profile[
                "investigation_ready"
            ] = investigation_ready

            profile[
                "profile_state"
            ] = (
                "REID_READY"
                if investigation_ready
                else "TRACK_ONLY"
            )

            if investigation_ready:
                ready_count += 1
                ready_total += 1

            else:
                track_only_count += 1
                track_only_total += 1

            total += 1

        with open(
            path,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                profiles,
                f,
                indent=2
            )

        print(
            f"{path.parent.name}: "
            f"{len(profiles)} profiles | "
            f"{ready_count} REID_READY | "
            f"{track_only_count} TRACK_ONLY"
        )

    print()
    print(
        "=" * 70
    )

    print(
        f"Total profiles : "
        f"{total}"
    )

    print(
        f"REID_READY     : "
        f"{ready_total}"
    )

    print(
        f"TRACK_ONLY     : "
        f"{track_only_total}"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()