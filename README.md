# 기기 적용 기준: YuNet 다중 크기 + 0.75 + SFace

기기에서 모델 실행 테스트는 이미 한 차례 완료했습니다. 이번 적용 요청은 **선택한 모델 설정 반영과 사용자 화면 연결**이며, 구체적인 작업 내용은 [DEVICE_HANDOFF.md](DEVICE_HANDOFF.md)에 정리했습니다.

`face_lab/`의 Python 코드는 기존 앱과 결과를 비교하는 기준 구현입니다. 이 저장소 자체에는 모바일 사용자 화면·영구 인물 DB·백그라운드 작업 관리·증분 연결이 구현되어 있지 않습니다.

## 폴더

- `face_lab/engine.py`: 사진 → 얼굴 정보·정규화 벡터, 초기 임시 그룹 API
- `face_lab/preprocess.py`: EXIF 방향 보정, BGR, 축소, 좌표 복원, 정렬·정규화
- `face_lab/detection.py`: 다중 크기 검출과 원본 좌표 NMS
- `face_lab/grouping.py`: 초기 군집화, 품질 제외, 모호성 보류
- `face_lab/models.py`: YuNet/SFace 고정 모델 SHA와 준비·로딩
- `configs/selected_multiscale.json`: 선택 설정 기록
- `DEVICE_HANDOFF.md`: 모델 설정·사용자 화면·수정 저장·적용 확인 작업 명세
- `tests/`: 회귀 검사, 기존 실험 코드와 설명, 공유 가능한 요약 결과
- `tests/private/`: 기존 원본 결과·검증 자료의 로컬 보관소. Git 제외

## 설치와 확인

Python 3.11 또는 3.12를 사용합니다.

```sh
python -m venv .venv
# macOS/Linux
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m face_lab prepare-models
.venv/bin/python -m face_lab check
.venv/bin/python -m unittest discover -s tests -p 'test_*.py'
```

Windows에서는 `.venv/bin/python`을 `.venv\Scripts\python.exe`로 바꿉니다. 모델 준비만 네트워크를 사용하며 모델 파일은 Git에 올리지 않습니다.

사진 한 장의 좌표·특징 추출:

```sh
.venv/bin/python -m face_lab extract --input /path/to/photo.jpg --photo-id photo-001 --output device_results/photo-001
```

출력은 `faces.json`, `features.npz`이며 출력 폴더가 이미 있으면 덮어쓰지 않습니다. 원본 이미지 복사나 HTML 보고서 생성은 하지 않습니다. 사진·벡터를 공유하려면 별도의 동의된 테스트 자료를 사용하세요.

## 고정 설정

검출 긴 변 960/1600/2400, 검출 점수 0.75, NMS 0.30, top-k 5000. 원본보다 확대하지 않으며 같은 입력 크기는 중복 처리하지 않습니다. 원본에서 112×112로 정렬한 후 SFace 128차원 float32 벡터를 L2 정규화합니다. 초기 그룹은 average 0.40, 모호성 폭 0.01, 최소 크기 1입니다. 얼굴 40px 미만, 흐림 50 미만, 크게 잘린 얼굴은 자동 그룹에서 제외합니다.

`0.75`는 검출 점수이며 인물 유사도가 아닙니다. 단독 얼굴은 임시 그룹을 만들 수 있습니다. 같은 사진에서 같은 그룹으로 배정된 얼굴은 검토 표시되며 자동으로 분리되지 않습니다. 그룹 번호는 영구 인물 ID가 아닙니다.

## Git에 전달

`git status --short`로 변경 목록을 검토하고 README, DEVICE_HANDOFF, configs, face_lab, tests, requirements, .gitignore를 커밋하세요. 원본 결과는 `tests/private/`에 보존되어 있고 공유 요약은 `tests/results/experiment_summaries.json`에 있습니다. 이 정리 작업에서는 커밋·푸시를 수행하지 않았습니다.
