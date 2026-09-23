from __future__ import annotations
from pathlib import Path
import cv2

def extract_clip(video_path: str | Path, output_path: str | Path, start_sec: float, end_sec: float, fps: float | None = None):
    video_path, output_path = Path(video_path), Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cap=cv2.VideoCapture(str(video_path))
    src_fps=float(cap.get(cv2.CAP_PROP_FPS) or fps or 25.0)
    width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc=cv2.VideoWriter_fourcc(*"mp4v")
    writer=cv2.VideoWriter(str(output_path), fourcc, src_fps, (width,height))
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0,start_sec)*1000)
    while True:
        pos=cap.get(cv2.CAP_PROP_POS_MSEC)/1000
        if pos > end_sec: break
        ok, frame=cap.read()
        if not ok: break
        writer.write(frame)
    writer.release(); cap.release()
    return output_path
