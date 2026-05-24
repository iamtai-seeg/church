"""편집 흔적 검출.

ffprobe 메타데이터에서 편집 SW 서명을 찾고, PySceneDetect로 하드컷 수를 센다.
원본 카메라/스마트폰 녹화는 보통 컷이 0개이고 인코더 태그가 카메라 펌웨어다.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


EDITOR_SIGNATURES = [
    "lavf", "lavc", "ffmpeg",
    "adobe", "premiere", "after effects", "media encoder",
    "imovie", "final cut", "compressor",
    "davinci", "resolve",
    "vegas", "shotcut", "kdenlive", "openshot",
    "handbrake", "capcut", "vllo", "kinemaster",
    "filmora", "powerdirector", "movavi",
]


@dataclass
class EditResult:
    cut_count: int
    editor_tags: list[str]
    has_editing_signature: bool
    suspected_edited: bool
    notes: list[str]


def _run_ffprobe(video_path: Path) -> dict:
    cmd = [
        "ffprobe", "-v", "error",
        "-print_format", "json",
        "-show_format", "-show_streams",
        str(video_path),
    ]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


def _find_editor_tags(meta: dict) -> list[str]:
    tags_to_scan: list[str] = []
    fmt = meta.get("format", {}) or {}
    for key in ("tags", "format"):
        v = fmt.get(key)
        if isinstance(v, dict):
            tags_to_scan.extend(str(x) for x in v.values())
    for s in meta.get("streams", []):
        st_tags = s.get("tags") or {}
        tags_to_scan.extend(str(x) for x in st_tags.values())

    hits: list[str] = []
    for blob in tags_to_scan:
        low = blob.lower()
        for sig in EDITOR_SIGNATURES:
            if sig in low and blob not in hits:
                hits.append(blob)
                break
    return hits


def _count_scene_cuts(video_path: Path, threshold: float = 27.0) -> int:
    # PySceneDetect의 ContentDetector는 컷 전환을 카운트한다.
    from scenedetect import ContentDetector, open_video, SceneManager

    video = open_video(str(video_path))
    sm = SceneManager()
    sm.add_detector(ContentDetector(threshold=threshold))
    sm.detect_scenes(video, show_progress=False)
    scenes = sm.get_scene_list()
    # 장면 수 - 1 = 컷 수 (장면이 0이면 컷도 0)
    return max(0, len(scenes) - 1)


def check_edits(video_path: Path) -> EditResult:
    if not shutil.which("ffprobe"):
        raise RuntimeError("ffprobe not found on PATH")

    notes: list[str] = []
    try:
        meta = _run_ffprobe(video_path)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffprobe failed: {e.stderr}") from e

    editor_tags = _find_editor_tags(meta)
    has_sig = bool(editor_tags)
    if has_sig:
        notes.append(f"인코더 태그에 편집/재인코딩 서명 감지: {', '.join(editor_tags[:3])}")

    try:
        cuts = _count_scene_cuts(video_path)
    except Exception as e:
        notes.append(f"컷 감지 실패: {e}")
        cuts = 0

    if cuts > 0:
        notes.append(f"하드컷 {cuts}개 감지")

    # 표준 엄격도: 컷 1개 이상이면 편집 의심, 인코더 서명만 있으면 약한 의심
    suspected = cuts >= 1
    return EditResult(
        cut_count=cuts,
        editor_tags=editor_tags,
        has_editing_signature=has_sig,
        suspected_edited=suspected,
        notes=notes,
    )
