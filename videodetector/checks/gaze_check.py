"""시선 및 깜박임 분석 (MediaPipe Tasks API).

FaceLandmarker로 매 프레임 시선 방향과 EAR을 계산한다.
- 시선이 정면에서 크게 벗어난 프레임 비율
- 깜박임 횟수 → 분당 깜박임
표준 기준:
- 정상 깜박임 분당 8~35회
- 정면 시선 이탈 비율 < 15%면 양호
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from .models_loader import get_model


# MediaPipe FaceLandmarker는 478개 랜드마크 (iris 4점 포함).
LEFT_IRIS_CENTER = 468
RIGHT_IRIS_CENTER = 473
LEFT_EYE_CORNERS = (33, 133)
RIGHT_EYE_CORNERS = (362, 263)
LEFT_EYE_VERT = [(159, 145), (158, 153)]
RIGHT_EYE_VERT = [(386, 374), (385, 380)]

SAMPLE_FPS = 5.0
GAZE_OFFSET_THRESHOLD = 0.22
EAR_BLINK_THRESHOLD = 0.20
# 암송/낭독 상황은 일반 대화보다 깜박임이 줄어들기 때문에 하한을 5/분으로 둔다.
NORMAL_BLINK_PER_MIN = (5, 35)
GOOD_ABNORMAL_RATIO = 0.15
BAD_ABNORMAL_RATIO = 0.30


@dataclass
class GazeResult:
    frames_analyzed: int
    face_detected_ratio: float
    abnormal_gaze_ratio: float
    blink_count: int
    blinks_per_min: float
    duration_sec: float
    judgment: str
    notes: list[str]


def _eye_aspect_ratio(lms, vert_pairs, img_w, img_h) -> float:
    dists = []
    for a, b in vert_pairs:
        ax, ay = lms[a].x * img_w, lms[a].y * img_h
        bx, by = lms[b].x * img_w, lms[b].y * img_h
        dists.append(math.hypot(ax - bx, ay - by))
    return float(np.mean(dists))


def _gaze_offset(lms, iris_center_idx, corner_a, corner_b, img_w) -> float:
    cx = lms[iris_center_idx].x * img_w
    ax = lms[corner_a].x * img_w
    bx = lms[corner_b].x * img_w
    eye_min, eye_max = sorted((ax, bx))
    eye_w = eye_max - eye_min
    if eye_w < 1e-3:
        return 0.0
    center = (eye_min + eye_max) / 2
    return abs(cx - center) / (eye_w / 2)


def check_gaze(video_path: Path) -> GazeResult:
    model_path = get_model("face_landmarker.task")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"비디오 열기 실패: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = total_frames / fps if fps else 0.0
    step = max(1, int(round(fps / SAMPLE_FPS)))

    options = mp_vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_faces=1,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    landmarker = mp_vision.FaceLandmarker.create_from_options(options)

    analyzed = 0
    face_seen = 0
    abnormal = 0
    blink_count = 0
    in_blink = False
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

            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts_ms = int(frame_idx * 1000 / fps)
            result = landmarker.detect_for_video(mp_image, ts_ms)

            frame_idx += 1
            analyzed += 1

            if not result.face_landmarks:
                continue
            face_seen += 1
            lms = result.face_landmarks[0]

            left_off = _gaze_offset(lms, LEFT_IRIS_CENTER, *LEFT_EYE_CORNERS, w)
            right_off = _gaze_offset(lms, RIGHT_IRIS_CENTER, *RIGHT_EYE_CORNERS, w)
            if (left_off + right_off) / 2 > GAZE_OFFSET_THRESHOLD:
                abnormal += 1

            left_ear = _eye_aspect_ratio(lms, LEFT_EYE_VERT, w, h)
            right_ear = _eye_aspect_ratio(lms, RIGHT_EYE_VERT, w, h)
            left_w = abs(lms[LEFT_EYE_CORNERS[0]].x - lms[LEFT_EYE_CORNERS[1]].x) * w
            right_w = abs(lms[RIGHT_EYE_CORNERS[0]].x - lms[RIGHT_EYE_CORNERS[1]].x) * w
            ear = 0.0
            if left_w > 1e-3 and right_w > 1e-3:
                ear = (left_ear / left_w + right_ear / right_w) / 2
            if ear < EAR_BLINK_THRESHOLD:
                if not in_blink:
                    blink_count += 1
                    in_blink = True
            else:
                in_blink = False
    finally:
        cap.release()
        landmarker.close()

    face_ratio = face_seen / analyzed if analyzed else 0.0
    abnormal_ratio = abnormal / face_seen if face_seen else 1.0
    blinks_per_min = (blink_count / duration * 60) if duration else 0.0

    if face_seen == 0:
        judgment = "얼굴 미감지"
        notes.append("얼굴이 한 번도 감지되지 않음")
    else:
        bad_signals = []
        if abnormal_ratio > BAD_ABNORMAL_RATIO:
            bad_signals.append(f"시선 이탈 비율 {abnormal_ratio:.1%}")
        if blinks_per_min < NORMAL_BLINK_PER_MIN[0]:
            bad_signals.append(f"깜박임 부족 ({blinks_per_min:.1f}/분)")
        elif blinks_per_min > NORMAL_BLINK_PER_MIN[1]:
            bad_signals.append(f"깜박임 과다 ({blinks_per_min:.1f}/분)")

        if bad_signals:
            judgment = "불량"
            notes.extend(bad_signals)
        elif abnormal_ratio > GOOD_ABNORMAL_RATIO:
            judgment = "주의"
            notes.append(f"시선 이탈 비율 {abnormal_ratio:.1%} (경계)")
        else:
            judgment = "양호"

    if face_ratio < 0.7 and face_seen > 0:
        notes.append(f"얼굴 감지율이 낮음 ({face_ratio:.0%}) — 측면 촬영/조명 의심")

    return GazeResult(
        frames_analyzed=analyzed,
        face_detected_ratio=face_ratio,
        abnormal_gaze_ratio=abnormal_ratio,
        blink_count=blink_count,
        blinks_per_min=blinks_per_min,
        duration_sec=duration,
        judgment=judgment,
        notes=notes,
    )
