"""상반신 노출 비율 분석 (MediaPipe Tasks API).

PoseLandmarker로 양 어깨/코가 프레임 안에 보이는지 확인한다.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from .models_loader import get_model


SAMPLE_FPS = 3.0
SHOULDER_VISIBILITY_MIN = 0.5
GOOD_RATIO = 0.85
BAD_RATIO = 0.70

LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
NOSE = 0


@dataclass
class PoseResult:
    frames_analyzed: int
    upper_body_ratio: float
    judgment: str
    notes: list[str]


def check_pose(video_path: Path) -> PoseResult:
    model_path = get_model("pose_landmarker_lite.task")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"비디오 열기 실패: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(fps / SAMPLE_FPS)))

    options = mp_vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_poses=1,
    )
    landmarker = mp_vision.PoseLandmarker.create_from_options(options)

    analyzed = 0
    upper_seen = 0
    notes: list[str] = []

    try:
        frame_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % step != 0:
                frame_idx += 1
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts_ms = int(frame_idx * 1000 / fps)
            result = landmarker.detect_for_video(mp_image, ts_ms)

            frame_idx += 1
            analyzed += 1

            if not result.pose_landmarks:
                continue
            lms = result.pose_landmarks[0]
            ls = lms[LEFT_SHOULDER]
            rs = lms[RIGHT_SHOULDER]
            nose = lms[NOSE]

            def in_frame(l) -> bool:
                return (
                    0.0 <= l.x <= 1.0
                    and 0.0 <= l.y <= 1.0
                    and getattr(l, "visibility", 1.0) >= SHOULDER_VISIBILITY_MIN
                )

            if in_frame(ls) and in_frame(rs) and in_frame(nose):
                upper_seen += 1
    finally:
        cap.release()
        landmarker.close()

    ratio = upper_seen / analyzed if analyzed else 0.0

    if ratio >= GOOD_RATIO:
        judgment = "양호"
    elif ratio >= BAD_RATIO:
        judgment = "주의"
        notes.append(f"상반신 노출 비율 {ratio:.0%} — 프레이밍 점검 권장")
    else:
        judgment = "불량"
        notes.append(f"상반신이 자주 잘림 ({ratio:.0%})")

    return PoseResult(
        frames_analyzed=analyzed,
        upper_body_ratio=ratio,
        judgment=judgment,
        notes=notes,
    )
