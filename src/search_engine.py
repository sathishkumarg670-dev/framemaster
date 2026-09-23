import json
from pathlib import Path

TIMELINE = Path(
    "data/evidence/investigation_timeline.json"
)


def load_timeline():
    with open(TIMELINE, "r") as f:
        return json.load(f)


def search_by_camera_01_track(track_id, limit=5):
    timeline = load_timeline()

    results = [
        item
        for item in timeline
        if item["camera_01_track"] == track_id
    ]

    results.sort(
        key=lambda x: x["similarity"],
        reverse=True
    )

    return results[:limit]


def search_by_camera_02_track(track_id, limit=5):
    timeline = load_timeline()

    results = [
        item
        for item in timeline
        if item["camera_02_track"] == track_id
    ]

    results.sort(
        key=lambda x: x["similarity"],
        reverse=True
    )

    return results[:limit]


def search_top_candidates(limit=10):
    timeline = load_timeline()

    timeline.sort(
        key=lambda x: x["similarity"],
        reverse=True
    )

    return timeline[:limit]


def print_result(item):

    print(
        f"Camera 1 Track : {item['camera_01_track']:03d}"
    )

    print(
        f"Camera 2 Track : {item['camera_02_track']:03d}"
    )

    print(
        f"Similarity      : {item['similarity']:.4f}"
    )

    print(
        f"Evidence frames : {len(item['evidence'])}"
    )

    print("-" * 50)


if __name__ == "__main__":

    print()
    print("VISIONFORGE AI SEARCH ENGINE")
    print("=" * 50)

    results = search_top_candidates(10)

    print()
    print("Top candidate records:")
    print()

    for result in results:
        print_result(result)

    print("Search engine ready.")