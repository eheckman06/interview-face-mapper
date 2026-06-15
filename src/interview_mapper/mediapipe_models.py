from __future__ import annotations

import urllib.request
from pathlib import Path

FACE_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)


def model_dir() -> Path:
    root = Path.cwd() / "data" / "models"
    root.mkdir(parents=True, exist_ok=True)
    return root


def face_landmarker_model() -> Path:
    target = model_dir() / "face_landmarker.task"
    if target.exists() and target.stat().st_size > 0:
        return target

    with urllib.request.urlopen(FACE_LANDMARKER_URL, timeout=120) as response:
        target.write_bytes(response.read())
    return target
