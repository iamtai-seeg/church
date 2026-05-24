"""오륜전도훈련생 복음 암기 영상 검증 도구.

사용법:
    python detect.py <파일 또는 디렉토리> [--out report.csv]

체크 항목:
    1. 편집 흔적 (컷 수 + 인코더 서명)
    2. 시선/눈 깜박임 분석
    3. 상반신 노출 비율
    4. 사람 재검증 필요 여부 종합 판정
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path

from checks.edit_check import check_edits, EditResult  # noqa: F401
from checks.gaze_check import check_gaze, GazeResult  # noqa: F401
from checks.pose_check import check_pose, PoseResult  # noqa: F401


VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".mkv", ".avi", ".webm"}


@dataclass
class FinalVerdict:
    needs_human_review: bool
    reasons: list[str]
    summary: str


def _decide_final(edit: EditResult | None, gaze: GazeResult | None,
                  pose: PoseResult | None, errors: list[str]) -> FinalVerdict:
    reasons: list[str] = []

    if errors:
        reasons.extend(f"분석 오류: {e}" for e in errors)

    if edit and edit.suspected_edited:
        reasons.append(f"편집 의심 (컷 {edit.cut_count}개)")
    elif edit and edit.has_editing_signature:
        reasons.append("재인코딩/편집 SW 서명 감지 (약한 신호)")

    if gaze:
        if gaze.judgment == "불량":
            reasons.append("시선/깜박임 불량")
        elif gaze.judgment == "주의":
            reasons.append("시선 경계 수준")
        elif gaze.judgment == "얼굴 미감지":
            reasons.append("얼굴이 감지되지 않음")

    if pose:
        if pose.judgment == "불량":
            reasons.append("상반신 노출 불량")
        elif pose.judgment == "주의":
            reasons.append("상반신 프레이밍 주의")

    # 표준 엄격도: "불량" 또는 "편집 의심"이 하나라도 있으면 재검증 필요
    needs_review = False
    if edit and edit.suspected_edited:
        needs_review = True
    if gaze and gaze.judgment in {"불량", "얼굴 미감지"}:
        needs_review = True
    if pose and pose.judgment == "불량":
        needs_review = True
    if errors:
        needs_review = True

    if needs_review:
        summary = "사람 재검증 필요"
    elif reasons:
        summary = "통과 (경미한 경계 신호 있음)"
    else:
        summary = "통과"

    return FinalVerdict(needs_human_review=needs_review, reasons=reasons, summary=summary)


def _print_result(path: Path, edit, gaze, pose, verdict: FinalVerdict, elapsed: float) -> None:
    print(f"\n{'=' * 70}")
    print(f"파일: {path.name}")
    print(f"분석 시간: {elapsed:.1f}초")
    print("-" * 70)

    if edit:
        print(f"[1] 편집 검출")
        print(f"    - 하드컷 수: {edit.cut_count}")
        print(f"    - 인코더 서명: {'있음 (' + ', '.join(edit.editor_tags[:2]) + ')' if edit.editor_tags else '없음'}")
        print(f"    - 판정: {'편집 의심' if edit.suspected_edited else '편집 흔적 없음'}")
    else:
        print("[1] 편집 검출: 실패")

    if gaze:
        print(f"[2] 시선/깜박임")
        print(f"    - 분석 프레임: {gaze.frames_analyzed} (얼굴 감지율 {gaze.face_detected_ratio:.0%})")
        print(f"    - 시선 이탈 비율: {gaze.abnormal_gaze_ratio:.1%}")
        print(f"    - 깜박임: {gaze.blink_count}회 ({gaze.blinks_per_min:.1f}/분)")
        print(f"    - 판정: {gaze.judgment}")
        for n in gaze.notes:
            print(f"      · {n}")
    else:
        print("[2] 시선/깜박임: 실패")

    if pose:
        print(f"[3] 상반신 노출")
        print(f"    - 노출 비율: {pose.upper_body_ratio:.1%}")
        print(f"    - 판정: {pose.judgment}")
        for n in pose.notes:
            print(f"      · {n}")
    else:
        print("[3] 상반신 노출: 실패")

    print("-" * 70)
    print(f"[4] 종합: {verdict.summary}")
    if verdict.reasons:
        for r in verdict.reasons:
            print(f"    · {r}")
    print("=" * 70)


def _collect_videos(target: Path) -> list[Path]:
    if target.is_file():
        return [target]
    if target.is_dir():
        files = sorted(p for p in target.rglob("*") if p.suffix.lower() in VIDEO_EXTS)
        return files
    raise FileNotFoundError(target)


def _analyze_one(video: Path):
    edit = gaze = pose = None
    errors: list[str] = []

    try:
        edit = check_edits(video)
    except Exception as e:
        errors.append(f"편집검출: {e}")
        traceback.print_exc(file=sys.stderr)

    try:
        gaze = check_gaze(video)
    except Exception as e:
        errors.append(f"시선분석: {e}")
        traceback.print_exc(file=sys.stderr)

    try:
        pose = check_pose(video)
    except Exception as e:
        errors.append(f"상반신: {e}")
        traceback.print_exc(file=sys.stderr)

    verdict = _decide_final(edit, gaze, pose, errors)
    return edit, gaze, pose, verdict, errors


def _row_for_csv(video: Path, edit, gaze, pose, verdict) -> dict:
    return {
        "file": str(video),
        "cut_count": edit.cut_count if edit else "",
        "editor_signature": ";".join(edit.editor_tags) if edit and edit.editor_tags else "",
        "edit_suspected": edit.suspected_edited if edit else "",
        "duration_sec": f"{gaze.duration_sec:.1f}" if gaze else "",
        "face_detected_ratio": f"{gaze.face_detected_ratio:.3f}" if gaze else "",
        "abnormal_gaze_ratio": f"{gaze.abnormal_gaze_ratio:.3f}" if gaze else "",
        "blinks_per_min": f"{gaze.blinks_per_min:.1f}" if gaze else "",
        "gaze_judgment": gaze.judgment if gaze else "",
        "upper_body_ratio": f"{pose.upper_body_ratio:.3f}" if pose else "",
        "pose_judgment": pose.judgment if pose else "",
        "needs_human_review": verdict.needs_human_review,
        "summary": verdict.summary,
        "reasons": " | ".join(verdict.reasons),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="복음 암기 영상 검증")
    ap.add_argument("target", type=Path, help="비디오 파일 또는 디렉토리")
    ap.add_argument("--out", type=Path, default=None, help="CSV 리포트 저장 경로")
    args = ap.parse_args()

    try:
        videos = _collect_videos(args.target)
    except FileNotFoundError:
        print(f"경로를 찾을 수 없음: {args.target}", file=sys.stderr)
        return 2

    if not videos:
        print(f"비디오 파일이 없음: {args.target}", file=sys.stderr)
        return 2

    out_path = args.out or (args.target if args.target.is_dir() else args.target.parent) / "videodetector_report.csv"

    rows = []
    for i, v in enumerate(videos, 1):
        print(f"\n[{i}/{len(videos)}] 분석 시작: {v.name}", flush=True)
        t0 = time.time()
        edit, gaze, pose, verdict, _ = _analyze_one(v)
        elapsed = time.time() - t0
        _print_result(v, edit, gaze, pose, verdict, elapsed)
        rows.append(_row_for_csv(v, edit, gaze, pose, verdict))

    with out_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nCSV 리포트 저장: {out_path}")
    flagged = [r for r in rows if r["needs_human_review"]]
    print(f"\n총 {len(rows)}개 중 재검증 필요: {len(flagged)}개")
    for r in flagged:
        print(f"  - {Path(r['file']).name}: {r['summary']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
