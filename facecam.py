"""Find a streamer's facecam box in a 16:9 stream recording.

Approach
  1. Sample frames and run OpenCV's face detector on each.
  2. The facecam face stays in roughly the same place; faces in the game move.
     Cluster detections by position and keep a cluster seen in most frames.
  3. A webcam's background barely changes while gameplay and chat do, so
     grow a box outward from the face while its edges stay on still pixels.
  4. If no border is found, fall back to a 16:9 box sized from the face.

Returns (x, y, w, h) in 1920x1080 coordinates, or None.
"""

from __future__ import annotations

import subprocess

import numpy as np

W, H = 1920, 1080
SAMPLE_W = 960          # analyse at half resolution
SCALE = W / SAMPLE_W


def _frames(video: str, n: int = 24) -> np.ndarray:
    dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                                "-of", "csv=p=0", video], capture_output=True, text=True).stdout or 0)
    fps = max(0.2, n / max(dur, 1))
    h = SAMPLE_W * 9 // 16
    raw = subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-i", video, "-vf",
                          f"fps={fps:.3f},scale={SAMPLE_W}:{h},format=gray", "-f", "rawvideo", "-"],
                         capture_output=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, h, SAMPLE_W)


def detect(video: str) -> tuple[int, int, int, int] | None:
    import cv2

    frames = _frames(video)
    if len(frames) < 4:
        return None
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    dets = []
    for i, f in enumerate(frames):
        for (x, y, w, h) in cascade.detectMultiScale(f, scaleFactor=1.1, minNeighbors=6, minSize=(24, 24)):
            if h < f.shape[0] * 0.16:   # a big face means a full-screen camera, not an overlay
                dets.append((i, x + w / 2, y + h / 2, w, h))
    if not dets:
        return None

    # Cluster by centre; the facecam cluster appears in many distinct frames.
    best, best_frames = None, 0
    for _, cx, cy, w, _ in dets:
        members = [d for d in dets if abs(d[1] - cx) < w * 0.6 and abs(d[2] - cy) < w * 0.6]
        nframes = len({d[0] for d in members})
        if nframes > best_frames:
            best, best_frames = members, nframes
    if best_frames < max(3, len(frames) * 0.4):
        return None
    cx = float(np.median([d[1] for d in best]))
    cy = float(np.median([d[2] for d in best]))
    fw = float(np.median([d[3] for d in best]))
    fh = float(np.median([d[4] for d in best]))

    box = _static_box(frames, cx, cy, fw, fh)
    if box is None:
        # 16:9 box about 6 face widths wide (typical facecam framing), face near centre.
        bw = fw * 6
        bh = bw * 9 / 16
        box = (cx - bw / 2, cy - bh * 0.45, bw, bh)
    x, y, w, h = box
    x, y = max(0.0, x), max(0.0, y)
    w, h = min(w, SAMPLE_W - x), min(h, frames.shape[1] - y)
    return tuple(int(round(v * SCALE / 2)) * 2 for v in (x, y, w, h))


def _static_box(frames: np.ndarray, cx: float, cy: float, fw: float, fh: float):
    """Grow a box from the face while its edges stay on still pixels.

    A webcam's background (the streamer's room) barely changes, while gameplay
    and chat around it change constantly.
    """
    std = frames.astype(np.float32).std(axis=0)
    still = std < 9.0
    Hs, Ws = still.shape
    left, right = int(cx - fw), int(cx + fw)
    top, bottom = int(cy - fh), int(cy + fh * 1.2)
    step = max(2, int(fw * 0.1))
    grew = True
    while grew:
        grew = False
        if left - step >= 0 and still[top:bottom, left - step:left].mean() > 0.5:
            left -= step; grew = True
        if right + step <= Ws and still[top:bottom, right:right + step].mean() > 0.5:
            right += step; grew = True
        if top - step >= 0 and still[top - step:top, left:right].mean() > 0.5:
            top -= step; grew = True
        if bottom + step <= Hs and still[bottom:bottom + step, left:right].mean() > 0.5:
            bottom += step; grew = True
    # Snap to the frame edge when within a step of it.
    left = 0 if left < step else left
    top = 0 if top < step else top
    right = Ws if Ws - right < step else right
    bottom = Hs if Hs - bottom < step else bottom
    w, h = right - left, bottom - top
    if not (0.9 < w / h < 2.4) or w < fw * 2.5 or w > Ws * 0.36:
        return None
    return (left, top, w, h)


if __name__ == "__main__":
    import sys
    for path in sys.argv[1:]:
        print(path, detect(path))
