"""MediaPipe Tasks 모델 파일 자동 다운로드."""
from __future__ import annotations

import urllib.request
from pathlib import Path


MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

MODEL_URLS = {
    "face_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
        "face_landmarker/float16/1/face_landmarker.task"
    ),
    "pose_landmarker_lite.task": (
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
    ),
}


def get_model(name: str) -> Path:
    if name not in MODEL_URLS:
        raise KeyError(name)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / name
    if not path.exists():
        url = MODEL_URLS[name]
        print(f"[모델 다운로드] {name} ← {url}", flush=True)
        urllib.request.urlretrieve(url, path)
    return path
