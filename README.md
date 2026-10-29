# YuNet + SFace 사진 인물 그룹화 테스트

개인 사진에서 **YuNet으로 얼굴을 검출하고 SFace로 특징을 추출한 뒤, 비슷한 얼굴을 그룹화**하는 로컬 Python 프로젝트입니다. macOS와 Windows에서 CPU로 테스트합니다. 원본 사진을 변경하거나 사진·얼굴·벡터를 외부 서버로 보내지 않으며, 모델을 재학습하지 않습니다.

**처음 테스트할 때는 `average_same_threshold`와 같은 설정을 사용하세요.** 프로젝트 작성자가 개인 사진 결과를 직접 확인했을 때 이 방식이 실제 인물 구분에 가장 가깝다고 판단해 후속 테스트의 시작 설정으로 선택했습니다. 정답 기반 정확도 평가나 모든 사진에 대한 최적 설정 확정은 아닙니다.

| 문서 | 내용 |
|---|---|
| [README.md](README.md) | 시작 설정, 분석 실행, 결과 확인 |
| [노트북_테스트_준비.md](노트북_테스트_준비.md) | 작성자 Mac의 Git 업로드 준비, Mac/Windows 설치·검사 |
| [GROUPING_EXPERIMENTS.md](GROUPING_EXPERIMENTS.md) | average 적용 원리, 비교 실험, 정답 작성, 개발/평가 분리 |
| [TESTING.md](TESTING.md) | 실제 실행 결과·검사와 과거 기록, 기술 기준 |

## 먼저 사용할 테스트 설정

| 항목 | 값 | 의미 |
|---|---|---|
| 모델 | YuNet + SFace | 검출 및 얼굴 특징 추출 |
| `--linkage` | **average** | 두 그룹 사이 모든 얼굴 쌍의 평균 유사도로 병합 판단 |
| `--similarity-threshold` | **0.40** | 코사인 유사도 기준. 같은 사람일 확률 40%라는 뜻은 아님 |
| `--ambiguity-margin` | **0.01** | 자기 그룹과 경쟁 그룹 점수가 비슷하면 보류 |
| `--min-group-size` | **1** | 얼굴 한 장만 있는 그룹도 허용 |
| 미확인 재배정 | **사용하지 않음** | `--reassign-unknown` 옵션을 붙이지 않음 |
| `--score-threshold` | 0.80 | YuNet 검출 기준 |
| `--max-side` | 1600 | 검출할 이미지의 최대 긴 변 |
| `--min-face-size` / `--min-blur` | 40 / 50 | 원본 얼굴 최소 변(px) / 정렬 얼굴의 흐림 점수 기준 |
| `--nms-threshold` / `--top-k` | 0.30 / 5000 | 검출 중복 억제 / 후보 상한 |

complete는 가장 낮은 교차 유사도를 기준으로 판단하므로 같은 사람도 촬영 조건에 따라 나뉠 수 있습니다. average는 평균을 사용해 일부 낮은 유사도를 보완할 수 있습니다. 그만큼 다른 사람이 섞이지 않았는지도 함께 확인해야 합니다.

**코드 기본값은 변경하지 않았습니다.** 옵션을 생략하면 complete·유사도 0.45·모호성 폭 0.03·최소 그룹 2·검출 0.9로 실행됩니다. 아래 명령의 옵션을 모두 사용하세요. `average_same_threshold`는 비교 결과 폴더 이름이며, 실제 선택 옵션은 `--linkage average`입니다.

## 설치

프로젝트 폴더에서 실행합니다. Python **3.11 또는 3.12 64비트**를 사용하고, 다른 OS의 `.venv`를 복사하지 마세요. `requirements.txt`가 macOS와 Windows에 맞는 NumPy·SciPy 버전을 선택합니다. 전체 clone·설치 절차는 [노트북 준비 안내](노트북_테스트_준비.md)에 있습니다.

macOS 터미널(zsh/bash), Python 3.11이 설치된 경우:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
mkdir -p ljm_dataset
.venv/bin/python -m face_lab prepare-models
.venv/bin/python -m face_lab check
```

Windows **PowerShell**, Python 3.12가 설치된 경우:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
New-Item -ItemType Directory -Force -Path .\ljm_dataset | Out-Null
.\.venv\Scripts\python.exe -m face_lab prepare-models
.\.venv\Scripts\python.exe -m face_lab check
```

Python 3.12를 설치한 Mac은 `python3.11` 대신 `python3.12`로 가상환경을 만듭니다. Windows에서 3.11을 설치했다면 `py -3.11`을 사용합니다. 이미 준비된 가상환경은 다시 만들 필요가 없습니다. 위 명령은 가상환경 활성화를 요구하지 않으므로 PowerShell 실행 정책 변경도 필요 없습니다. 모델 준비와 패키지 설치에는 인터넷이 필요하고, 이후 분석은 로컬에서 수행합니다.

## average 설정으로 분석

테스트할 사진을 `ljm_dataset/`에 복사한 뒤 **사용 중인 OS의 명령 하나**를 실행합니다. Git clone에는 개인 사진·모델·기존 결과가 포함되지 않습니다. 각 테스트 PC에서 새로 분석하세요.

macOS:

```bash
.venv/bin/python -m face_lab analyze --input ljm_dataset --output results/average_runs --linkage average --similarity-threshold 0.40 --ambiguity-margin 0.01 --min-group-size 1 --score-threshold 0.80 --max-side 1600 --min-face-size 40 --min-blur 50 --nms-threshold 0.30 --top-k 5000
```

Windows PowerShell:

```powershell
.\.venv\Scripts\python.exe -m face_lab analyze --input .\ljm_dataset --output .\results\average_runs --linkage average --similarity-threshold 0.40 --ambiguity-margin 0.01 --min-group-size 1 --score-threshold 0.80 --max-side 1600 --min-face-size 40 --min-blur 50 --nms-threshold 0.30 --top-k 5000
```

결과는 매번 새 `results/average_runs/run_.../` 폴더에 저장됩니다. 완료 메시지의 `Result:` 경로를 확인하세요. 아래 `run_실제_폴더`는 **자신의 실행에서 출력된 폴더 이름으로 교체**합니다.

macOS:

```bash
FACE_RUN='results/average_runs/run_실제_폴더'
open "$FACE_RUN/index.html"
```

Windows PowerShell:

```powershell
$FaceRun = '.\results\average_runs\run_실제_폴더'
Invoke-Item "$FaceRun\index.html"
```

첫 화면에서 인물별 얼굴 모음, 미확인 후보·점수·사유, 검출 박스·랜드마크와 정렬 얼굴을 확인하세요. **다른 사람이 같은 그룹에 섞인 경우, 같은 사람이 여러 그룹으로 나뉜 경우를 모두 기록**하세요. 같은 사진의 서로 다른 얼굴이 같은 그룹이면 검토 표시가 나옵니다. 거울·사진 속 사진 등의 예외가 있어 자동 금지하지 않습니다.

## 기존 로컬 벡터로 같은 설정 적용

`--source`는 **동일한 PC·사진에서 생성한 실제 실행 폴더**로 바꾸세요. 다른 OS의 캐시는 런타임 차이로 거부될 수 있으므로 처음에는 위 `analyze`를 사용합니다.

macOS:

```bash
.venv/bin/python -m face_lab regroup --source "$FACE_RUN" --input ljm_dataset --output results/average_runs --linkage average --similarity-threshold 0.40 --ambiguity-margin 0.01 --min-group-size 1
```

Windows PowerShell:

```powershell
.\.venv\Scripts\python.exe -m face_lab regroup --source "$FaceRun" --input .\ljm_dataset --output .\results\average_runs --linkage average --similarity-threshold 0.40 --ambiguity-margin 0.01 --min-group-size 1
```

`regroup`는 source의 검출·품질 필터·벡터를 그대로 재사용합니다. 추출 설정도 위 시작 설정과 같아야 전체 조건이 같습니다. 입력·모델·전처리·라이브러리가 달라져 캐시 검증이 실패하면 새 `analyze`가 필요합니다.

## 실제 실행 기록과 결과 파일

2026-10-29 Mac에서 같은 54장·145개 임베딩으로 비교했습니다. 작성자가 확인하고 선택한 average 결과는 **48그룹, 미확인 5개(모호성 3·박스 경계 1·흐림 1)**입니다. complete 기준선은 59그룹·미확인 18개였습니다. average를 별도 `regroup`으로 적용했을 때도 얼굴 구성원이 같았습니다. **48은 정답 인원 수가 아니며, 다른 사진·OS에서 같은 개수가 나오는 것을 합격 기준으로 삼지 않습니다.** 개인 사진 전체 정답이 없어 Precision·Recall·F1은 미평가입니다.

| 파일 | 확인할 내용 |
|---|---|
| `index.html` | 인물별 모음, 미확인 후보, 검출·정렬 확인 |
| `run_config.json`, `summary.json` | 실제 적용 설정, 버전, 개수·사유·시간 |
| `faces.csv`, `photos.csv` | 얼굴별 상태와 사진별 검출·실패 기록 |
| `diagnostics.json`, `diagnostics.csv` | 개별 유사도와 그룹 점수, 후보, 판정 단계 |
| `ground_truth_template.csv` | face_id별 익명 실제 인물 ID 입력 양식 |
| `photo_review_template.csv` | 미검출·오검출 검토 양식 |
| `annotated/`, `geometry/`, `crops/`, `montages/` | 박스·랜드마크, 확대, 정렬 얼굴, 그룹 모음 |
| `features.npz`, `cache.json` | 로컬 벡터와 캐시 검증 정보 |

CSV는 UTF-8 BOM으로 저장합니다. Excel에서는 face_id와 익명 ID 열을 텍스트로 유지하세요. 이미지에 한글 폰트가 없으면 영문 그룹 ID를 표시합니다.

## 정답 검토와 평가

macOS:

```bash
.venv/bin/python -m face_lab review --run "$FACE_RUN"
# 출력된 review_*.html에서 작성한 CSV를 해당 실행 폴더의 ground_truth.csv로 저장한 뒤 실행
.venv/bin/python -m face_lab evaluate --run "$FACE_RUN" --ground-truth "$FACE_RUN/ground_truth.csv"
```

Windows PowerShell:

```powershell
.\.venv\Scripts\python.exe -m face_lab review --run "$FaceRun"
# 출력된 review_*.html에서 작성한 CSV를 해당 실행 폴더의 ground_truth.csv로 저장한 뒤 실행
.\.venv\Scripts\python.exe -m face_lab evaluate --run "$FaceRun" --ground-truth "$FaceRun\ground_truth.csv"
```

같은 사람은 같은 익명 ID(`A01` 등), 다른 사람은 다른 ID를 입력합니다. 자동 그룹 번호를 정답으로 간주하지 마세요. 미입력은 빈칸, 판단 불가는 `annotation_status=uncertain`, 평가 제외는 `include=0`으로 구분합니다. 정답이 없으면 지표를 계산하지 않습니다. 평가 시 미확인은 각기 다른 단독 예측으로 취급하므로 같은 사람 쌍을 놓치면 Recall에 반영됩니다.

임계값 비교·재배정·개발/평가 분리는 [그룹화 실험 안내](GROUPING_EXPERIMENTS.md)를 따릅니다. `compare`의 `--linkage average`는 임계값 비교·재배정의 기반을 average로 지정하지만, complete 기준선과 재배정 실험도 함께 생성합니다. **average만 적용하는 첫 테스트는 위 `analyze` 또는 `regroup`입니다.** Windows에서 `compare`는 디렉터리 심볼릭 링크 생성 권한이 필요할 수 있습니다.

사진·얼굴·벡터·수동 정답·결과는 로컬에 보관합니다. Git 업로드 전 확인 방법은 [노트북 준비 안내](노트북_테스트_준비.md#작성자-macos에서-git-업로드-준비)를 참고하세요.
