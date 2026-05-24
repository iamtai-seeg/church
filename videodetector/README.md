# videodetector

오륜전도훈련생이 복음 전문을 암기한 영상을 자동 검증하는 도구.

## 체크 항목

1. **편집 흔적**: 영상 메타데이터의 편집 SW 서명 + 하드컷 수
2. **시선/눈동자**: MediaPipe FaceMesh로 시선 방향과 깜박임 분석. 정면 이탈 비율과 분당 깜박임이 정상 범위인지 판정
3. **상반신 노출**: MediaPipe Pose로 양 어깨와 얼굴이 프레임에 보이는 비율
4. **종합 판정**: 위 세 가지를 종합해 사람 재검증 필요 여부 표시

## 설치

```bash
# 사전 요구사항
brew install ffmpeg
python3.11 -m venv .venv

# 패키지
.venv/bin/pip install -r requirements.txt
```

> MediaPipe 호환성 때문에 Python 3.11을 권장합니다 (3.13은 아직 미지원).

## 사용법

```bash
# 디렉토리 안의 모든 비디오 분석
.venv/bin/python detect.py /path/to/videos

# 단일 파일
.venv/bin/python detect.py /path/to/video.mp4

# CSV 출력 경로 지정 (기본: 입력 디렉토리/videodetector_report.csv)
.venv/bin/python detect.py /path/to/videos --out result.csv
```

지원 포맷: `.mp4 .mov .m4v .mkv .avi .webm`

## 출력

- 영상별 콘솔 요약 (편집/시선/상반신/종합)
- CSV 리포트: 모든 영상의 수치와 최종 판정 (`needs_human_review` 열)

## 판정 기준 (표준)

| 항목 | 양호 | 주의 | 불량 |
|---|---|---|---|
| 하드컷 수 | 0 | - | 1 이상 (편집 의심) |
| 시선 이탈 비율 | < 15% | 15~30% | > 30% |
| 분당 깜박임 | 5~35 | - | 그 외 (암송 상황은 일반 대화보다 적게 깜박임) |
| 상반신 노출 | ≥ 85% | 70~85% | < 70% |

종합 판정에서 **불량** 또는 **편집 의심**이 하나라도 있으면 `needs_human_review = True`.

## 한계

- 편집 검출은 컷 + 메타데이터 기반이라, 정교한 페이드/연속 편집은 탐지가 어려움
- 시선 임계값은 정면 촬영 + 일반 조명을 가정. 측면 촬영이나 안경 반사가 심하면 감지율이 떨어질 수 있음
- 최종 판정은 자동화 1차 필터. 의심 영상은 반드시 사람이 다시 확인할 것
