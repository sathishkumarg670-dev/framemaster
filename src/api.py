from pathlib import Path
from typing import Optional
from uuid import uuid4
import json
import shutil

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as ReportImage, PageBreak
from reportlab.lib import colors
from reportlab.lib.utils import ImageReader

from src.investigation_search import (
    load_all_profiles,
    search_by_attributes,
    search_by_camera,
    search_by_reference_track,
    search_by_track,
    dataset_statistics,
)
from src.job_manager import submit as submit_job, list_jobs as list_job_rows, get_job as get_job_row


APP_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = APP_ROOT / "data"
VIDEO_DIRECTORY = DATA_DIR / "videos"
INVESTIGATION_DIRECTORY = DATA_DIR / "investigation"
CANDIDATES_FILE = INVESTIGATION_DIRECTORY / "appearance_candidates.json"

VIDEO_EXTENSIONS = {
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".webm",
    ".m4v",
}


VIDEO_DIRECTORY.mkdir(parents=True, exist_ok=True)
INVESTIGATION_DIRECTORY.mkdir(parents=True, exist_ok=True)


app = FastAPI(
    title="VISIONFORGE AI",
    description="AI-Assisted Multi-Camera Investigation System",
    version="2.1.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# REQUEST MODELS
# ============================================================

class InvestigationRequest(BaseModel):
    target_type: str = "person"

    # Legacy / friendly API names
    shirt_color: Optional[str] = None
    pants_color: Optional[str] = None

    # Canonical names
    upper_body_color: Optional[str] = None
    lower_body_color: Optional[str] = None

    backpack: Optional[str] = None
    headwear: Optional[str] = None

    object_type: Optional[str] = None
    object_color: Optional[str] = None
    brand: Optional[str] = None
    distinctive_feature: Optional[str] = None

    camera_id: Optional[str] = None

    limit: int = 20


class ReportRequest(BaseModel):
    investigation: InvestigationRequest
    candidates: list[dict] = Field(default_factory=list)
    selected_videos: list[dict] = Field(default_factory=list)
    cross_camera_candidates: list[dict] = Field(default_factory=list)


# ============================================================
# HELPERS
# ============================================================

def norm(value):
    if value is None:
        return ""

    return str(value).strip().lower()


def resolve_person_colors(request: InvestigationRequest):
    """
    Resolve old API names and canonical names.

    Canonical fields take priority when both are supplied.
    """

    upper = request.upper_body_color

    if upper is None:
        upper = request.shirt_color

    lower = request.lower_body_color

    if lower is None:
        lower = request.pants_color

    return upper, lower


def make_video_id(path: Path) -> str:
    """
    Generate a stable unique video ID.

    Existing camera-style videos keep their stem when possible.
    Same-stem files receive a suffix based on extension.
    """

    stem = path.stem.lower().replace(" ", "_")
    suffix = path.suffix.lower().lstrip(".")

    same_stem = [
        p
        for p in VIDEO_DIRECTORY.iterdir()
        if p.is_file()
        and p.suffix.lower() in VIDEO_EXTENSIONS
        and p.stem.lower() == path.stem.lower()
    ]

    if len(same_stem) <= 1:
        return stem

    return f"{stem}_{suffix}"


def discover_video_files():
    files = []

    if not VIDEO_DIRECTORY.exists():
        return files

    for path in sorted(VIDEO_DIRECTORY.iterdir()):
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
            files.append(path)

    return files


def find_video(video_id: str):
    wanted = norm(video_id)

    for path in discover_video_files():
        if make_video_id(path) == wanted:
            return path

    return None


def load_candidates():
    if not CANDIDATES_FILE.exists():
        return []

    try:
        data = json.loads(
            CANDIDATES_FILE.read_text(encoding="utf-8")
        )

        if isinstance(data, list):
            return data

        if isinstance(data, dict):
            for key in (
                "candidates",
                "appearance_candidates",
                "results",
            ):
                value = data.get(key)

                if isinstance(value, list):
                    return value

    except Exception:
        return []

    return []


# ============================================================
# BASIC ROUTES
# ============================================================

@app.get("/")
def root():
    return {
        "name": "VISIONFORGE AI",
        "version": app.version,
        "status": "running",
        "description": "AI-Assisted Multi-Camera Investigation System",
    }


@app.get("/health")
def health():
    profiles = load_all_profiles()
    stats = dataset_statistics(profiles)

    return {
        "status": "healthy",
        "profile_statistics": stats,
    }


# ============================================================
# VIDEO MANAGEMENT
# ============================================================

@app.get("/videos")
def list_videos():
    videos = []

    for path in discover_video_files():
        videos.append(
            {
                "video_id": make_video_id(path),
                "filename": path.name,
                "extension": path.suffix.lower(),
                "size_bytes": path.stat().st_size,
            }
        )

    return {
        "count": len(videos),
        "videos": videos,
    }


@app.get("/videos/{video_id}")
def get_video(video_id: str):
    path = find_video(video_id)

    if path is None:
        raise HTTPException(
            status_code=404,
            detail="Video not found",
        )

    return {
        "video_id": make_video_id(path),
        "filename": path.name,
        "path": str(path),
        "extension": path.suffix.lower(),
        "size_bytes": path.stat().st_size,
    }


@app.post("/videos/upload")
async def upload_video(file: UploadFile = File(...)):
    original_name = Path(file.filename or "uploaded_video").name
    extension = Path(original_name).suffix.lower()

    if extension not in VIDEO_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported video format: {extension}",
        )

    unique_name = (
        f"{Path(original_name).stem}_{uuid4().hex[:10]}{extension}"
    )

    destination = VIDEO_DIRECTORY / unique_name

    with destination.open("wb") as output:
        shutil.copyfileobj(file.file, output)

    return {
        "status": "uploaded",
        "video_id": make_video_id(destination),
        "filename": destination.name,
        "path": str(destination),
        "size_bytes": destination.stat().st_size,
    }


# ============================================================
# DATASET
# ============================================================

@app.get("/dataset/stats")
def dataset_stats():
    profiles = load_all_profiles()
    return dataset_statistics(profiles)


# ============================================================
# CANDIDATES
# ============================================================

@app.get("/candidates")
def candidates(limit: int = 20):
    limit = max(1, min(limit, 500))

    data = load_candidates()

    return {
        "count": min(limit, len(data)),
        "candidates": data[:limit],
    }


# ============================================================
# INVESTIGATOR SEARCH
# ============================================================

@app.post("/investigate")
def investigate(request: InvestigationRequest):

    profiles = load_all_profiles()

    target = norm(request.target_type)

    # --------------------------------------------------------
    # OBJECT SEARCH
    # --------------------------------------------------------

    if target == "object":
        from src.object_pipeline import search_objects_by_filters, OBJECT_INDEX

        if not OBJECT_INDEX.exists():
            return {
                "status": "object_index_not_ready",
                "target_type": "object",
                "query": request.model_dump(),
                "matching_tracks": 0,
                "candidate_count": 0,
                "candidates": [],
                "message": (
                    "No object index is present. Run object indexing "
                    "for the required CCTV videos first."
                ),
            }

        limit = max(1, min(request.limit, 500))
        object_rows = search_objects_by_filters(
            class_name=request.object_type,
            object_color=request.object_color,
            brand=request.brand,
            distinctive_feature=request.distinctive_feature,
            top_k=limit,
        )

        return {
            "status": "search_complete",
            "target_type": "object",
            "query": request.model_dump(),
            "matching_tracks": len(object_rows),
            "candidate_count": len(object_rows),
            "candidates": object_rows,
            "identity_confirmed": False,
        }

    # --------------------------------------------------------
    # PERSON SEARCH
    # --------------------------------------------------------

    upper_color, lower_color = resolve_person_colors(request)

    unsupported = []

    results = search_by_attributes(
        profiles,
        upper_body_color=upper_color,
        lower_body_color=lower_color,
        backpack=request.backpack,
        headwear=request.headwear,
        camera_id=request.camera_id,
        ready_only=True,
    )

    limit = max(1, min(request.limit, 500))

    resolved_query = {
        "target_type": "person",

        # Preserve both naming conventions in API output
        "shirt_color": request.shirt_color,
        "pants_color": request.pants_color,
        "upper_body_color": upper_color,
        "lower_body_color": lower_color,

        "backpack": request.backpack,
        "headwear": request.headwear,
        "object_type": request.object_type,
        "object_color": request.object_color,
        "brand": request.brand,
        "distinctive_feature": request.distinctive_feature,
        "camera_id": request.camera_id,
    }

    return {
        "status": "search_complete",
        "target_type": "person",
        "query": resolved_query,
        "matching_tracks": len(results),
        "candidate_count": len(results),
        "unsupported_attributes": unsupported,
        "candidates": results[:limit],
        "identity_confirmed": False,
    }


# ============================================================
# CAMERA SEARCH
# ============================================================

@app.get("/search/camera/{camera_id}")
def camera_search(
    camera_id: str,
    limit: int = 50,
):
    profiles = load_all_profiles()

    limit = max(1, min(limit, 500))

    return {
        "camera": camera_id,
        "matches": search_by_camera(
            profiles,
            camera_id,
            ready_only=False,
        )[:limit],
    }


@app.get("/search/camera/{camera_id}/track/{track_id}")
def track_search(
    camera_id: str,
    track_id: int,
):
    result = search_by_track(
        load_all_profiles(),
        camera_id,
        track_id,
    )

    if result is None:
        raise HTTPException(
            status_code=404,
            detail="Track not found",
        )

    return result


@app.get("/search/reference/{camera_id}/{track_id}")
def reference_search(
    camera_id: str,
    track_id: int,
    limit: int = 20,
):
    limit = max(1, min(limit, 500))

    results = search_by_reference_track(
        load_all_profiles(),
        camera_id,
        track_id,
        limit=limit,
    )

    return {
        "reference": {
            "camera_id": camera_id,
            "track_id": track_id,
        },
        "count": len(results),
        "candidates": results,
        "identity_confirmed": False,
    }


# ============================================================
# TOPOLOGY
# ============================================================

@app.get("/topology")
def topology():
    topology_file = DATA_DIR / "config" / "camera_topology.json"

    if not topology_file.exists():
        topology_file = APP_ROOT / "config" / "camera_topology.json"

    if not topology_file.exists():
        return {
            "status": "not_configured",
            "cameras": [],
            "transitions": [],
        }

    try:
        return json.loads(
            topology_file.read_text(encoding="utf-8")
        )

    except Exception as exc:
        return {
            "status": "error",
            "message": str(exc),
        }


# ============================================================
# OBJECT STATUS
# ============================================================

@app.get("/objects/status")
def object_status():
    return {
        "status": "prototype",
        "ready": False,
        "message": (
            "Dedicated object detection and object Re-ID "
            "index is not yet available in the unified dataset."
        ),
    }


# ============================================================
# JOB ROUTES
# ============================================================

@app.get("/jobs")
def jobs():
    rows = list_job_rows()
    return {"jobs": rows, "count": len(rows)}


@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    row = get_job_row(job_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return row


# ============================================================
# VIDEO PROCESSING
# ============================================================

@app.post("/videos/{video_id}/process")
def process_video(
    video_id: str,
    max_frames: Optional[int] = None,
    frame_stride: int = 1,
):
    path = find_video(video_id)

    if path is None:
        raise HTTPException(status_code=404, detail="Video not found")

    from src.pipeline.orchestrator import process_video as run_pipeline

    frame_stride = max(1, int(frame_stride))
    if max_frames is not None:
        max_frames = max(1, int(max_frames))

    job_id = submit_job(
        "process_video",
        run_pipeline,
        path,
        device="cpu",
        max_frames=max_frames,
        frame_stride=frame_stride,
    )

    return {
        "status": "accepted",
        "job_id": job_id,
        "video_id": make_video_id(path),
        "filename": path.name,
        "message": "Video processing job started.",
    }


# ============================================================
# TIMELINE
# ============================================================

@app.get("/timeline")
def timeline(limit: int = 100):
    timeline_file = (
        INVESTIGATION_DIRECTORY /
        "investigation_timeline.json"
    )

    if not timeline_file.exists():
        return {
            "count": 0,
            "events": [],
        }

    try:
        data = json.loads(
            timeline_file.read_text(
                encoding="utf-8"
            )
        )

        if isinstance(data, list):
            events = data
        elif isinstance(data, dict):
            events = data.get("events", [])
        else:
            events = []

        limit = max(1, min(limit, 500))

        return {
            "count": min(limit, len(events)),
            "events": events[:limit],
        }

    except Exception as exc:
        return {
            "count": 0,
            "events": [],
            "error": str(exc),
        }


# ============================================================
# PDF REPORT
# ============================================================

@app.post("/report/pdf")
def report_pdf(request: ReportRequest):
    """Generate a human-readable investigation package.

    The report deliberately separates attribute matches from cross-camera
    associations so an attribute score is never presented as identity
    confidence. Evidence images are embedded when their local files exist.
    """
    output = INVESTIGATION_DIRECTORY / "visionforge_investigation_report.pdf"
    INVESTIGATION_DIRECTORY.mkdir(parents=True, exist_ok=True)

    styles = getSampleStyleSheet()
    document = SimpleDocTemplate(
        str(output),
        pagesize=A4,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36,
        title="VISIONFORGE AI Investigation Report",
        author="VISIONFORGE AI",
    )
    story = []

    upper, lower = resolve_person_colors(request.investigation)
    cross = request.cross_camera_candidates or []

    story.append(Paragraph("VISIONFORGE AI Investigation Report", styles["Title"]))
    story.append(Paragraph(
        "AI-assisted multi-camera investigation evidence package",
        styles["Subtitle"],
    ))
    story.append(Spacer(1, 12))

    query_rows = [
        ["Target Type", request.investigation.target_type],
        ["Upper Body", upper or "Any"],
        ["Lower Body", lower or "Any"],
        ["Camera", request.investigation.camera_id or "All"],
        ["Attribute Candidates", str(len(request.candidates))],
        ["Cross-Camera Associations", str(len(cross))],
    ]
    table = Table(query_rows, colWidths=[170, 320])
    table.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("BACKGROUND", (0, 0), (0, -1), colors.lightgrey),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    story.append(table)
    story.append(Spacer(1, 14))

    story.append(Paragraph(
        "Important: attribute matching and Re-ID similarity are investigative "
        "signals, not identity probabilities. Identity confirmation is not "
        "performed automatically; candidates require human review.",
        styles["BodyText"],
    ))
    story.append(Spacer(1, 16))

    # -------------------- SUMMARY --------------------
    story.append(Paragraph("Investigation Summary", styles["Heading2"]))
    ready = sum(1 for c in request.candidates if c.get("investigation_ready"))
    story.append(Paragraph(
        f"{len(request.candidates)} attribute-search candidates were returned; "
        f"{ready} are marked investigation-ready. {len(cross)} cross-camera "
        "association records were supplied by the investigation engine.",
        styles["BodyText"],
    ))
    story.append(Spacer(1, 12))

    # -------------------- CROSS-CAMERA --------------------
    if cross:
        story.append(Paragraph("Cross-Camera Associations", styles["Heading2"]))
        story.append(Paragraph(
            "Appearance similarity is shown as a matching signal. It is not a "
            "claim of confirmed identity.",
            styles["BodyText"],
        ))
        story.append(Spacer(1, 8))

        for index, item in enumerate(cross[:12], start=1):
            source_camera = item.get("source_camera") or item.get("camera_01") or item.get("camera_01_id") or "—"
            target_camera = item.get("target_camera") or item.get("camera_02") or item.get("camera_02_id") or "—"
            source_track = item.get("source_track_id", item.get("camera_01_track", "—"))
            target_track = item.get("target_track_id", item.get("camera_02_track", "—"))
            sim = item.get("appearance_similarity", item.get("similarity"))
            attr = item.get("attribute_score")
            temporal = item.get("temporal_analysis") or {}
            temporal_valid = temporal.get("valid")
            gap = temporal.get("time_gap_sec")

            rows = [
                ["Association", f"{source_camera} / Track {source_track}  →  {target_camera} / Track {target_track}"],
                ["Appearance similarity", f"{float(sim) * 100:.2f}%" if isinstance(sim, (int, float)) else "—"],
                ["Attribute match", f"{float(attr) * 100:.1f}%" if isinstance(attr, (int, float)) and attr >= 0 else "Not available"],
                ["Temporal consistency", ("Consistent" if temporal_valid else "Not confirmed") if temporal_valid is not None else "Not available"],
                ["Time gap", f"{float(gap):.2f} sec" if isinstance(gap, (int, float)) else "—"],
            ]
            t = Table(rows, colWidths=[145, 345])
            t.setStyle(TableStyle([
                ("GRID", (0, 0), (-1, -1), 0.35, colors.grey),
                ("BACKGROUND", (0, 0), (0, -1), colors.whitesmoke),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]))
            story.append(Paragraph(f"Association {index}", styles["Heading3"]))
            story.append(t)

            images = []
            for key, label in (("source_representative_crop", "Source"), ("target_representative_crop", "Target")):
                path_value = item.get(key)
                if not path_value:
                    continue
                path = Path(str(path_value))
                if not path.exists():
                    continue
                try:
                    iw, ih = ImageReader(str(path)).getSize()
                    max_w, max_h = 205, 150
                    scale = min(max_w / max(iw, 1), max_h / max(ih, 1))
                    img = ReportImage(str(path), width=iw * scale, height=ih * scale)
                    images.append([Paragraph(label, styles["BodyText"]), img])
                except Exception:
                    continue
            if images:
                img_table = Table(images, colWidths=[55, 425])
                img_table.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP")]))
                story.append(img_table)
            story.append(Spacer(1, 12))

    # -------------------- ATTRIBUTE CANDIDATES --------------------
    story.append(PageBreak())
    story.append(Paragraph("Attribute Search Candidates", styles["Heading2"]))
    story.append(Paragraph(
        "Attribute score represents the fraction of requested attributes that "
        "matched. For example, a score of 100% for one requested attribute means "
        "1/1 requested attributes matched; it is not a 100% identity probability.",
        styles["BodyText"],
    ))
    story.append(Spacer(1, 10))

    for index, candidate in enumerate(request.candidates[:30], start=1):
        camera = candidate.get("camera_id", "—")
        track = candidate.get("track_id", candidate.get("object_id", "—"))
        score = candidate.get("attribute_score")
        score_text = f"{float(score) * 100:.1f}%" if isinstance(score, (int, float)) else "—"
        first = candidate.get("first_seen_sec")
        last = candidate.get("last_seen_sec")
        duration = candidate.get("duration_sec")
        body = (
            f"<b>Candidate {index}</b><br/>"
            f"Camera: {camera} &nbsp;&nbsp; Track: {track}<br/>"
            f"Upper: {candidate.get('upper_body_color') or 'Any'} &nbsp;&nbsp; "
            f"Lower: {candidate.get('lower_body_color') or 'Any'}<br/>"
            f"Attribute match: {score_text} &nbsp;&nbsp; "
            f"First: {first if first is not None else '—'}s &nbsp;&nbsp; "
            f"Last: {last if last is not None else '—'}s &nbsp;&nbsp; "
            f"Duration: {duration if duration is not None else '—'}s<br/>"
            f"Identity confirmed: {bool(candidate.get('identity_confirmed', False))}"
        )
        story.append(Paragraph(body, styles["BodyText"]))
        path_value = candidate.get("representative_crop")
        if path_value and Path(str(path_value)).exists():
            try:
                iw, ih = ImageReader(str(path_value)).getSize()
                scale = min(180 / max(iw, 1), 135 / max(ih, 1))
                story.append(ReportImage(str(path_value), width=iw * scale, height=ih * scale))
            except Exception:
                pass
        story.append(Spacer(1, 10))

    # -------------------- TIMELINE --------------------
    timeline_file = INVESTIGATION_DIRECTORY / "investigation_timeline.json"
    timeline = []
    if timeline_file.exists():
        try:
            raw = json.loads(timeline_file.read_text(encoding="utf-8"))
            timeline = raw if isinstance(raw, list) else raw.get("events", []) if isinstance(raw, dict) else []
        except Exception:
            timeline = []

    if timeline:
        story.append(PageBreak())
        story.append(Paragraph("Investigation Timeline", styles["Heading2"]))
        for index, event in enumerate(timeline[:30], start=1):
            source = f"{event.get('camera_id', '—')} / Track {event.get('track_id', '—')}"
            if "camera_01_track" in event:
                source = f"Track {event.get('camera_01_track')} → Track {event.get('camera_02_track')}"
            timestamp = event.get("timestamp_seconds", event.get("first_seen_sec", "—"))
            similarity = event.get("similarity", event.get("appearance_similarity"))
            sim_text = f" | Similarity: {float(similarity) * 100:.2f}%" if isinstance(similarity, (int, float)) else ""
            description = event.get("description", event.get("evidence_type", ""))
            story.append(Paragraph(
                f"<b>{index}. {source}</b> — Time: {timestamp}s{sim_text}<br/>{description}",
                styles["BodyText"],
            ))
            story.append(Spacer(1, 6))

    story.append(Spacer(1, 14))
    story.append(Paragraph(
        "Human Review Notice: VISIONFORGE AI provides investigative leads and "
        "evidence organization. It does not automatically establish a person's "
        "identity, guilt, or involvement in an incident.",
        styles["BodyText"],
    ))

    document.build(story)
    return FileResponse(
        path=str(output),
        media_type="application/pdf",
        filename="visionforge_investigation_report.pdf",
    )


# ============================================================
# CROP / EVIDENCE IMAGE SERVING
# ============================================================

@app.get("/crops/image")
def get_crop_image(path: str):
    crop_path = Path(path)
    if not crop_path.exists() or not crop_path.is_file():
        raise HTTPException(status_code=404, detail="Crop image not found")
    
    return FileResponse(path=str(crop_path), media_type="image/jpeg")
