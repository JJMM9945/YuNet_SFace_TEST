# YuNet + SFace 로컬 얼굴 테스트

사진에서 얼굴 검출 → 특징 벡터 추출 → 비슷한 얼굴의 임시 그룹 생성 프로젝트. Android 온디바이스 사진 검색 적용 전 PC CPU 검증 목적. 실제 이름 지정·모델 재학습 범위 제외.

## 팀원 시작 순서

Python **3.11 또는 3.12**, Windows PowerShell 기준. clone 후 프로젝트 루트에서 실행.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m face_lab prepare-models
.\.venv\Scripts\python.exe -m face_lab check
New-Item -ItemType Directory -Force ljm_dataset
# ljm_dataset 폴더에 테스트 사진 복사
.\.venv\Scripts\python.exe -m face_lab analyze --input .\ljm_dataset --output .\results
```

모델 준비 명령만 인터넷 사용. 분석·재그룹화·검토·평가의 로컬 실행. 사진·모델·결과·수동 기록은 Git 미포함이며 각 PC에서 별도 준비. 원본 이동·수정·삭제 없음. 빈 입력 폴더도 분석 가능.

macOS: `python3.12 -m venv .venv` 후 위 명령의 `.\.venv\Scripts\python.exe`를 `.venv/bin/python`으로 변경. 의존성 버전은 [requirements.txt](requirements.txt)에 고정.

## 현재 테스트 설정

CLI 기본값과 [설정 기록](configs/selected_multiscale.json)의 일치. JSON은 기록용이며 `--config` 입력 기능 없음.

| 항목 | 값 | 의미·선택 이유 |
|---|---|---|
| YuNet 검출 크기 | 960·1600·2400px | 세 크기의 검출 결합을 통한 작은 얼굴 누락 감소 |
| 검출 점수 | 0.75 이상 | 기존 0.8보다 누락 감소를 위한 시작값 |
| 중복 제거 | IoU 0.30 | 원본 좌표에서 겹치는 검출 상자의 중복 제거 |
| SFace | 정렬 → 128차원 특징 → L2 정규화 | 자세·위치 차이 완화 및 비교 가능한 벡터 생성 |
| 그룹화 | average, 유사도 0.40 | 그룹 간 평균 유사도 기준; 기존 complete 대비 과도한 분리 완화 |
| 모호성 폭 | 0.01 | 경쟁 그룹과의 판단 차이가 작은 얼굴의 보류 |
| 최소 그룹 크기 | 1 | 단독 인물의 임시 그룹 허용 |
| 품질 제한 | 얼굴 40px, 흐림 지표 50 | 작은 얼굴·흐린 얼굴의 무리한 그룹 배정 제한 |

사용자 사진 검토에 따른 실험 시작값. 검증된 정확도 기준·모든 사진의 최적값 의미 아님. 대표 얼굴 재배정·기존 검출 보존·수동 보정의 자동 적용 없음.

코사인 유사도 = 정규화 벡터의 내적, **클수록 가까움**. 거리 = `1 - 유사도`. average 군집의 절단 거리 `1 - 0.40`. 개별 얼굴 쌍 전체의 유사도 0.40 보장 아님. 평균 기준에 따른 다른 인물 혼합·같은 인물 분리 가능성. 같은 사진의 여러 얼굴이 한 그룹인 경우 검토 표시. 실행별 그룹 번호와 영구 `person_id`의 구분.

## 결과 확인·재검사

명령 마지막 `Result:` 폴더의 **index.html** 열기. 인물 목록·미확인·검토 필요·사진별 필터, 파일명/ID 검색, 확대, 최대4얼굴 비교 기능. 방향키 이동·Esc 닫기. 서버·CDN 불필요. 비교 선택과 실제 그룹 수정의 구분.

| 파일 | 역할 |
|---|---|
| index.html / annotated / montages | 결과 화면·검출 상자 이미지·인물별 모음 |
| faces.csv / photos.csv | 얼굴별·사진별 결과; 얼굴0개 사진 포함, Excel용 UTF-8 BOM |
| summary.json / summary.md | 검출·그룹·미확인 수, 실패·소요 시간 |
| run_config.json / detection_inputs.json | 실제 옵션·모델 SHA·OpenCV 버전·전처리·크기별 입력 |
| cache.json / features.npz | 사진 해시·벡터·face_id 대응; 묶음 보관 필요 |
| ground_truth_template.csv / photo_review_template.csv | 수동 인물 정답·미검출/오검출 양식 |
| diagnostics.json | 그룹 판단 근거·경쟁 후보; 유사도와 정확도 확률의 구분 |

특징 추출 반복 없는 그룹 임계값 변경 예시. `--source`에 실제 폴더 지정.

```powershell
.\.venv\Scripts\python.exe -m face_lab regroup --input .\ljm_dataset --source .\results\run_실제폴더 --output .\results --similarity-threshold 0.42
.\.venv\Scripts\python.exe -m face_lab review --run .\results\run_실제폴더
.\.venv\Scripts\python.exe -m face_lab evaluate --run .\results\run_실제폴더 --ground-truth .\정답.csv
.\.venv\Scripts\python.exe -m face_lab --help
.\.venv\Scripts\python.exe -m face_lab analyze --help
```

모델·전처리·실행 라이브러리·사진 내용/목록/상대 경로·벡터 체크섬 불일치 시 캐시 재사용 차단. 재분석 필요.

`review` HTML에서 가명 ID 입력 및 CSV 저장 후 평가 가능. 미입력·제외·불확실 얼굴의 평가 제외. 포함된 미확인은 서로 다른 단독 예측으로 처리; 같은 사람의 연결 누락은 FN 반영. 동일 인물 얼굴 **쌍** 기준 Precision·Recall·F1 계산; 임시 번호 비교 아님. 미검출 얼굴의 평가는 해당 CSV만으로 불가. [검증 기준](TESTING.md) 참고.

## 코드 구조·전처리

| 경로 | 담당 |
|---|---|
| face_lab/__main__.py | CLI·기본 설정·입력값 검증 |
| models.py | 공식 모델 다운로드·크기/SHA 검증·CPU 로딩 |
| preprocess.py / detection.py | EXIF·크기/좌표 변환·정렬/특징·여러 크기 검출 |
| grouping.py / pipeline.py | 군집화·품질/모호성 보류·사진 처리·캐시·CSV |
| reports.py / report_ui.py / results_ui.html | 이미지·안전한 HTML 데이터·결과 UI |
| review.py / review_ui.html / evaluation.py | 수동 정답 입력·출처 확인·쌍 평가 |
| tests / configs | 현재 기능 검사·선택 설정 기록 |

Pillow EXIF 방향 보정1회 → uint8 BGR. 종횡비 유지·INTER_AREA 축소·확대/패딩 없음. 반올림된 가로/세로 비율로 상자와 5개 랜드마크를 **방향 보정된 원본 좌표**로 복원. 원본에서 `alignCrop(112×112) → feature → L2` 수행. 세 검출 결과의 원본 좌표 점수순 NMS 통합. 입력 정렬·CPU1스레드·난수시드0 적용. 다른 OS/라이브러리의 부동소수점 차이 가능성.

## 검증·Git 공유

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py"
.\.venv\Scripts\python.exe -m pip check
git status --short
git ls-files
```

공유 대상: 코드·HTML 템플릿·tests·configs·requirements·README·TESTING·.gitignore. 사진·얼굴·벡터·Word·결과·수동 기록·가상환경의 Git 미포함 확인 필요. 새 데이터셋 폴더도 `.gitignore` 등록 또는 기본 `ljm_dataset` 사용 권장. 저장소 생성·commit·push의 사용자 수행.

## 과거 테스트 보관 자료

과거 결과의 [TESTING.md](TESTING.md) 통합. [과거 테스트 기록 Word](results/reports/20261101_test_archive/얼굴AI_과거테스트기록_T01-T09.docx)·결과 그림·수치 근거의 `results/reports/20261101_test_archive/` 로컬 보존, clone 시 미포함. 기존 ‘발표참고용’ 명칭을 **과거 테스트 보관 자료**로 변경. 발표 활용 여부는 사용자 선택.

과거 실험 도구·테스트·중복 문서는 `results/reports/cleanup_backup_20261102/`에 복구용 보관. 현재 실행·Git 공유 대상에서 제외. `step1기록 저장/`의 수동 원본 보존.

모델 출처: [OpenCV API](https://docs.opencv.org/4.x/d0/dd4/tutorial_dnn_face.html), [Zoo YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) MIT, [Zoo SFace](https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface) Apache-2.0. 다운로드 URL·고정 SHA의 `models.py` 기록, 준비 결과·라이선스의 로컬 `models/` 보관.
