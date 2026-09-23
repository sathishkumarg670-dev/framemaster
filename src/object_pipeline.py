"""
VISIONFORGE AI object investigation index.

This deliberately keeps object matching separate from person Re-ID.
It uses YOLO COCO object detections and a generic torchvision visual
embedding. It is a retrieval aid, not an identity classifier.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Iterable
import cv2
import numpy as np
import torch
from torchvision.models import resnet18, ResNet18_Weights
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
OBJECT_INDEX = ROOT / "data" / "investigation" / "object_index.json"
OBJECT_DIR = ROOT / "data" / "investigation" / "objects"

TARGET_CLASSES = {
    24: "backpack", 25: "umbrella", 26: "handbag", 28: "suitcase",
    27: "tie", 41: "cup", 39: "bottle", 63: "laptop", 73: "book",
}

class ObjectEmbedder:
    def __init__(self, device="cpu"):
        self.device = device
        weights = ResNet18_Weights.DEFAULT
        self.model = resnet18(weights=weights)
        self.model.fc = torch.nn.Identity()
        self.model.eval().to(device)
        self.transform = weights.transforms()

    @torch.inference_mode()
    def encode(self, image_bgr):
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        from PIL import Image
        tensor = self.transform(Image.fromarray(rgb)).unsqueeze(0).to(self.device)
        feat = self.model(tensor).flatten().cpu().numpy().astype(np.float32)
        norm = np.linalg.norm(feat)
        return (feat / norm).tolist() if norm else feat.tolist()

def index_video(video_path: str | Path, model_path="yolo11n.pt", device="cpu", stride=15, max_frames=None):
    video_path = Path(video_path)
    OBJECT_DIR.mkdir(parents=True, exist_ok=True)
    detector = YOLO(model_path)
    embedder = ObjectEmbedder(device)
    cap = cv2.VideoCapture(str(video_path))
    records = []
    frame = 0
    while True:
        ok, image = cap.read()
        if not ok:
            break
        if max_frames and frame >= max_frames:
            break
        if frame % max(1, stride):
            frame += 1
            continue
        result = detector.predict(image, classes=list(TARGET_CLASSES), conf=0.20, device=device, verbose=False)[0]
        if result.boxes is not None:
            for i, box in enumerate(result.boxes):
                cls = int(box.cls.item())
                x1,y1,x2,y2 = map(int, box.xyxy[0].tolist())
                x1=max(0,x1); y1=max(0,y1); x2=min(image.shape[1],x2); y2=min(image.shape[0],y2)
                crop=image[y1:y2,x1:x2]
                if crop.size == 0:
                    continue
                oid=f"{video_path.stem}_{frame}_{i}"
                path=OBJECT_DIR/f"{oid}.jpg"
                cv2.imwrite(str(path), crop)
                records.append({
                    "object_id": oid, "video_id": video_path.stem,
                    "frame_index": frame, "class_name": TARGET_CLASSES[cls],
                    "confidence": float(box.conf.item()), "bbox":[x1,y1,x2,y2],
                    "crop_path": str(path.relative_to(ROOT)),
                    "embedding": embedder.encode(crop),
                })
        frame += 1
    cap.release()
    data={"status":"ready","source_video":video_path.name,"objects":records,"embedding_model":"torchvision-resnet18-imagenet","identity_confirmed":False}
    OBJECT_INDEX.parent.mkdir(parents=True, exist_ok=True)
    OBJECT_INDEX.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return data

def search_objects(query_embedding, top_k=20, class_name=None):
    data=json.loads(OBJECT_INDEX.read_text(encoding="utf-8")) if OBJECT_INDEX.exists() else {"objects":[]}
    q=np.asarray(query_embedding,dtype=np.float32)
    q/=max(np.linalg.norm(q),1e-12)
    out=[]
    for row in data.get("objects",[]):
        if class_name and row.get("class_name") != class_name:
            continue
        e=np.asarray(row.get("embedding",[]),dtype=np.float32)
        if e.size != q.size: continue
        out.append((float(np.dot(q,e)), row))
    out.sort(key=lambda x:x[0], reverse=True)
    return [{"similarity":s,"identity_confirmed":False,**r} for s,r in out[:top_k]]


def search_objects_by_filters(
    class_name: str | None = None,
    object_color: str | None = None,
    brand: str | None = None,
    distinctive_feature: str | None = None,
    top_k: int = 50,
):
    """Return indexed object observations matching supported metadata.

    The current object index contains class, confidence, frame, bbox, crop and
    embedding metadata. Color/brand/distinctive-feature fields are accepted
    for forward compatibility but are only applied when those fields exist
    in an indexed record.
    """
    if not OBJECT_INDEX.exists():
        return []

    data = json.loads(
        OBJECT_INDEX.read_text(encoding="utf-8")
    )

    requested = {
        "class_name": class_name.strip().lower()
        if class_name else None,
        "object_color": object_color.strip().lower()
        if object_color else None,
        "brand": brand.strip().lower()
        if brand else None,
        "distinctive_feature": distinctive_feature.strip().lower()
        if distinctive_feature else None,
    }

    rows = []
    for row in data.get("objects", []):
        if requested["class_name"] and str(row.get("class_name", "")).lower() != requested["class_name"]:
            continue

        compatible = True
        for field in ("object_color", "brand", "distinctive_feature"):
            wanted = requested[field]
            if not wanted:
                continue
            actual = row.get(field)
            if actual is None or str(actual).strip().lower() != wanted:
                compatible = False
                break

        if not compatible:
            continue

        rows.append({
            **row,
            "similarity": None,
            "identity_confirmed": False,
        })

    rows.sort(
        key=lambda item: (
            float(item.get("confidence", 0.0)),
            int(item.get("frame_index", 0)),
        ),
        reverse=True,
    )
    return rows[:max(1, min(int(top_k), 500))]
