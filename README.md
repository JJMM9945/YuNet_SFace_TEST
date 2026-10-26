# YuNet + SFace 사진 인물 구분 테스트

Android 사진 검색 앱에 사용할 얼굴 AI를 **Windows PC에서 먼저 검증**하는 Python 프로젝트입니다. 얼굴이 보이는 사진을 대상으로 하며, 실제 이름 식별·뒷모습 인식·모델 재학습은 하지 않습니다. 원본을 변경하거나 사진을 외부 서버로 보내지 않고 CPU에서 분석합니다.

문서는 세 개입니다: **이 파일은 사용 안내**, [TESTING.md](TESTING.md)는 결과·검증·기술 기준, [노트북 준비](노트북_테스트_준비.md)는 GitHub 업로드와 다른 PC 설치를 설명합니다.

## 처리 흐름과 용어

사진 방향 보정 → **YuNet으로 얼굴 찾기** → **SFace로 정렬·벡터 추출** → **별도 로직으로 그룹화** → 결과 저장.

| 용어 | 쉬운 뜻 |
|---|---|
| 검출·박스 | 얼굴 위치를 사각형으로 찾기. 누구인지 구분하는 단계와 다름 |
| 랜드마크·정렬 | 눈·코·입 기준점으로 얼굴 위치·기울기를 맞추기 |
| 특징 벡터 | 얼굴을 비교하기 위한 128개 숫자. 확실한 신원표는 아님 |
| 유사도·임계값 | 얼마나 비슷한지 나타낸 점수와 통과 기준. **0.8이 확률 80%는 아님** |
| `photo_id` / `face_id` | 사진 한 장 / 검출 얼굴 한 개를 가리키는 ID |
| 그룹·`인물001` | 같은 사람으로 추정한 임시 묶음. 실제 사람 수나 영구 `person_id`와 다름 |
| 미확인 | 얼굴을 찾았지만 품질·그룹 판단 기준으로 배정을 보류한 상태 |
| 캐시·해시 | 계산 결과 재사용 기록 / 파일 내용 변경을 확인하는 지문 |
| GT·정답 | 실제 같은 사람인지 비교할 기준. 자동 그룹 번호가 정답은 아님 |

## 설치와 실행

Python **3.11/3.12 64비트**를 지원하며 실제 확인 환경은 Windows 11·Python **3.12.2**입니다. 의존성 버전은 `requirements.txt`에 고정했습니다. VS Code 인터프리터는 `.venv\Scripts\python.exe`를 선택하세요.

```powershell
Set-Location C:\YuNet_SFace_test
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
New-Item -ItemType Directory -Force -Path .\ljm_dataset, .\gt_dataset | Out-Null
.\.venv\Scripts\python.exe -m face_lab prepare-models
.\.venv\Scripts\python.exe -m face_lab check
```

사진을 `ljm_dataset/`에 복사한 뒤 실행합니다. 준비된 가상환경·모델은 다시 만들 필요가 없습니다. 모델 준비 후 분석은 네트워크 없이 실행하며, 입력 수를 고정하지 않고 빈 폴더·손상 사진도 처리합니다.

```powershell
# 마지막 PC 자동 테스트와 같은 설정
.\.venv\Scripts\python.exe -m face_lab analyze --input .\ljm_dataset --output .\results --score-threshold 0.8 --max-side 1600 --similarity-threshold 0.40 --ambiguity-margin 0.01 --min-group-size 1
```

표시된 새 결과 폴더의 `index.html`을 엽니다. GT 테스트는 `--input .\gt_dataset`으로 바꿉니다. 명령별 옵션은 `python -m face_lab 명령 --help`로 확인할 수 있습니다.

| 설정 | 마지막 테스트 값 | 바꿀 때의 영향 |
|---|---:|---|
| `--score-threshold` | 0.8 | 낮추면 얼굴 후보와 오검출 가능성이 함께 늘어남 |
| `--max-side` | 1600 | 검출 이미지 최대변. 크다고 항상 유리하지는 않음 |
| `--similarity-threshold` | 0.40 | 낮추면 합치기 쉬워지지만 다른 사람 혼합 위험 증가 |
| `--ambiguity-margin` | 0.01 | 커지면 모호성으로 보류하는 범위가 넓어짐 |
| `--min-group-size` | 1 | 단독 얼굴도 그룹 허용. 정확도 향상과는 다른 표시 정책 |
| `--min-face-size` / `--min-blur` | 40 / 50 | 작은 얼굴·흐린 정렬 얼굴의 품질 기준 |

**옵션 생략 시 기본값**은 검출 0.9, 비교 0.45, margin 0.03, 최소 그룹 크기 2입니다. 위 테스트 값도 최적값으로 확정한 것은 아닙니다.

## 결과와 주요 파일

| 폴더·파일 | 역할 |
|---|---|
| `ljm_dataset/`, `gt_dataset/` | 개인 사진 / 공개 GT 사진 입력 |
| `models/`, `requirements.txt` | 모델·라이선스 / Python 의존성 버전 |
| `face_lab/`, `tools/`, `tests/` | 분석 프로그램 / 비교 도구 / 코드 검사 |
| `results/run_.../index.html` | 실행별 결과의 입구 |
| `annotated/`, `photos/` | 박스를 그린 이미지 / 방향 보정한 원본 내용 사본. 입력 원본은 그대로 유지 |
| `crops/`, `montages/` | 정렬 얼굴 / 그룹별 얼굴 모음 |
| `faces.csv`, `photos.csv` | 얼굴별 / 사진별 장부. 0개 검출·오류·미확인 사유 확인 |
| `features.npz`, `cache.json` | 얼굴 ID와 벡터 / 사진 해시·추출 설정·버전 대응 |
| `run_config.json`, `summary.*` | 실행 설정·모델 정보 / 개수·시간 요약 |
| `ground_truth_template.csv`, `photo_review_template.csv` | 인물 정답 / 미검출·오검출 검토 양식 |
| `review_*.html`, `comparison_*/index.html` | 수동 ID 입력 / 미확인 얼굴 상세 비교 |
| `manual_review.json`, `applied_constraints.json` | 사용자 보정의 출처·적용 관계 |
| `evaluations/`, `constraint_checks.json` | 정답 평가 / 지정한 보정 관계의 적용 확인 |

CSV는 Excel용 UTF-8 BOM입니다. 한글 폰트가 없으면 생성 이미지에는 영문 그룹 ID를 표시합니다. 사진·얼굴·벡터·결과·가상환경은 `.gitignore`로 제외하며, 사용자 지정 출력 경로도 별도 제외해야 합니다.

## 검토·재그룹화·수동 수정

`$run`에는 확인할 실제 **자동 실행 폴더**를 지정합니다.

```powershell
$run = '.\results\run_20261026T141743_391410Z_e7d109'

# 저장 벡터로 비교 기준만 바꾸기
.\.venv\Scripts\python.exe -m face_lab regroup --source "$run" --input .\ljm_dataset --similarity-threshold 0.45 --ambiguity-margin 0.01 --min-group-size 1

# 원본 위치·정렬 얼굴·경쟁 그룹 비교
.\.venv\Scripts\python.exe tools\compare_unknown_faces.py --run "$run"

# 브라우저에서 수동 ID 입력 화면 생성
.\.venv\Scripts\python.exe -m face_lab review --run "$run"
```

검토 화면에서 같은 사람은 같은 ID(A01 등), 다른 사람은 다른 ID를 입력하고 **CSV를 다운로드**해 `$run\ground_truth.csv`에 저장합니다. ID는 영문·숫자로 시작하는 1~64자의 영문·숫자·`_`·`-`를 사용합니다. 불확실한 얼굴은 빈칸, 오검출은 평가 포함 해제로 남깁니다. 자동 그룹을 확인 없이 정답으로 일괄 입력하지 마세요.

```powershell
# 자동 결과를 정답과 비교
.\.venv\Scripts\python.exe -m face_lab evaluate --run "$run" --ground-truth "$run\ground_truth.csv"

# 입력한 정답으로 별도의 수정 결과 생성
.\.venv\Scripts\python.exe -m face_lab apply-review --source "$run" --ground-truth "$run\ground_truth.csv" --input .\ljm_dataset
```

`apply-review`는 명시한 ID만 적용하고 미입력·제외 얼굴은 미확인으로 남깁니다. `correct-groups --help`는 일부 얼굴의 같은/다른 사람 관계만 적용하는 기능입니다. 이 경우 언급하지 않은 자동 그룹은 유지합니다. JSON 입력 기준은 [TESTING.md](TESTING.md)에 있습니다.

**수동 보정은 모델 재학습이 아닙니다.** 새 `analyze`·`regroup`에는 보정이 자동 승계되지 않습니다. 사진·추출 환경이 달라지면 캐시/정답 재사용을 거부하므로 새 결과에서 확인해야 합니다. 수동 수정본을 같은 정답으로 다시 평가하는 기능도 차단합니다.

## 문제가 생겼을 때 읽는 법

| 증상 | 의미·가능한 이유 | 확인할 곳 |
|---|---|---|
| 사진 처리 실패 | 디코딩·파일 읽기 등 오류 | `photos.csv`의 `status`, `error` |
| 처리 성공인데 검출 0개 | 얼굴 없음 또는 작은 얼굴·가림·각도·검출 기준 등의 영향 | `annotated/`, 사진별 검토 양식 |
| 얼굴은 있는데 미확인 | 품질 기준 또는 그룹 비교에서 보류 | `faces.csv`의 `reason`, 비교 화면 |
| 같은 사람이 여러 그룹 | 촬영 조건에 따른 벡터 차이, 모든 쌍을 따지는 그룹 규칙 등 | 정렬 얼굴·정답 기반 분리 오류 |
| 다른 사람이 같은 그룹 | 다른 사람도 높은 점수가 나올 수 있고 기준 완화가 영향을 줌 | 얼굴 모음·혼합 오류 |
| 같은 단체 사진의 얼굴에 같은 라벨 | 혼합·중복 검출·반사 등 검토 필요 | `same_photo_same_group` 표시 |

현재 미확인 18개는 **그룹 비교 모호성 16개·박스 경계 1개·흐림 1개**입니다. 모호성 16개 중 최근접 개별 점수가 0.8 이상인 경우도 4개입니다. 한 장과 비슷해도 그룹 전체 비교에서 보류할 수 있으므로, 해상도만 올리면 해결되는 문제는 아닙니다.

전체 사진 크기와 **사진 안 얼굴 크기**는 다릅니다. 정렬 결과를 112×112로 확대해도 원래 없던 세부 정보는 생기지 않습니다. 반대로 큰 얼굴도 모호할 수 있습니다. 기록된 사유는 보류 규칙이며 각도의 영향 등 근본 원인을 모두 확정해 주지는 않습니다.

다음 단계는 **검출·정렬 확인 → 확실한 정답 입력 → 혼합과 분리 함께 평가 → 조정용·최종 확인용 사진 분리 → Android에서 같은 전처리와 기기 성능 검증**입니다. 실제 수치와 검증 한계는 [TESTING.md](TESTING.md)를 참고하세요.
