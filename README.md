# 사진 앨범 테스트 · YuNet + SFace

사진 폴더를 입력하면 **검출 0.75 고정**으로 분석하고 사진 앨범 웹 결과를 만듭니다.

현재 기본: **1280 먼저 검출 → 얼굴0개 또는3개 이상이면 960·1600·2400 추가 검사**. 기존1280 상자·랜드마크 유지, 중복 제거 후 최종 얼굴에서 SFace 특징 추출1회. [선택 설정 기록](configs/selected_conditional.json) · [2026/11/16 비교 결과](tests/results/20261116_detection_policy.md).

- 1명 사진: 같은 인물끼리
- 2명 사진: 같은 두 사람 조합끼리(A+B = B+A), 애매하면 확인 필요
- 3명 이상: 단체 사진 모음 하나
- 얼굴 미탐지와 분석 실패는 구분

[테스트 결과와 차트](tests/results/README.md) · [기기 적용 요청](DEVICE_HANDOFF.md)

## 설치

Python 3.11 또는 3.12를 사용합니다.

```sh
python -m venv .venv
# macOS/Linux: 가상환경 활성화
source .venv/bin/activate
# Windows PowerShell: .\.venv\Scripts\Activate.ps1

python -m pip install -r requirements.txt
python -m face_lab prepare-models
python -m face_lab check
```

모델 준비만 네트워크를 사용합니다.

## 자기 사진 실행

```sh
python -m face_lab albums --input "/내/사진/폴더"
# Windows 예시
python -m face_lab albums --input "C:\Users\사용자\Pictures\테스트사진"
```

특정 데이터셋·기존 캐시는 필요하지 않습니다. 하위 폴더도 읽고 매번 새로 분석합니다. JPG/PNG/BMP/WebP/TIFF 지원, HEIC는 변환 후 사용합니다.

마지막 Result 경로의 index.html을 브라우저로 엽니다. 기본 출력은 results/photo_albums/run_...입니다. --output으로 결과 상위 폴더, --models로 모델 위치를 지정합니다. 결과 폴더는 입력 사진 폴더 밖에 둡니다. 개인 사진·벡터·결과가 포함된 기본 results/는 Git에서 제외됩니다.

## 적용 기준

기본1280 / 조건 충족 사진만 960·1600·2400 추가 / 검출0.75 / 원본 좌표 IoU0.30 중복 제거 / top-k5000. 추가 후보가 기존 상자와 겹치면 기존 상자 유지. 조건 판단은 품질 제외 전 검출 수 기준이며 작은 얼굴·오검출도 포함합니다.

원본에서 SFace 정렬112×112 → 특징128차원 → L2 정규화. 1명·2명 사진에서 각각 average0.40·모호성0.01·최소그룹1 적용. 원본 얼굴40px 미만·흐림 지표50 미만·크게 잘린 얼굴의 자동 배정 제외. 사진별 실제 검출 크기·조건 충족·추가 검출 수의 photos.json 기록, 집계·버전·설정의 summary.json 기록.

이번54장 비교: 1280단독154검출·0개사진2장 → 조건부157검출·1장, 추가검사27장. 조건부28.37초, 기존세크기38.10초의 PC1회 사진 분석 시간. 검출 수와 정확도의 구분, 두 얼굴을 찾고 세 번째를 놓친 사진의 조건 미충족 가능성. 기존 오검출의 자동 제거 없음. Android 디코딩·처리 시간·품질의 별도 검증 필요.

탐지 얼굴 수는 실제 인원수와 다를 수 있습니다. 임시 그룹 번호는 영구 인물 ID가 아니며 앨범 수는 정확도가 아닙니다. 대규모 얼굴 군집은 제곱 메모리를 사용하므로 기기에서 처리 범위를 확인해야 합니다.

## 코드

| 위치 | 역할 |
|---|---|
| face_lab/engine.py | 고정 설정의 검출·벡터 추출 |
| face_lab/photo_albums.py | 입력 폴더 분석 및 기기용 group_photos API |
| face_lab/albums.py | 인물·조합·단체 사진 앨범 배정 |
| face_lab/album_report.py, album_report.html | 결과 웹 화면 |
| face_lab/preprocess.py, detection.py, grouping.py, models.py | 전처리·검출·군집·모델 로딩 |
| tests/ | 현재 기능 회귀 검사 |
| tests/results/ | 짧은 과거 실험 문서·이미지·집계 |
과거 개인 원본 결과(tests/private/)는 정리하여 삭제했습니다. 입력 사진 폴더와 공유용 문서·차트·집계는 유지합니다.

Python은 PC 및 기기 이식 기준 구현입니다. 모바일 UI·DB·권한·백그라운드 처리는 기존 앱과 연결해야 합니다. 과거 실험 실행 코드는 제거했으며 tests.compare_photo_counts 등 이전 명령은 사용하지 않습니다.

검사: `python -m unittest discover -s tests -p 'test_*.py'`, `node tests/test_photo_count_view.js`.
