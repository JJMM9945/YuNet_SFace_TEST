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

## 얼굴 ID로 확인 내용을 반영한 별도 결과

`correct-groups`는 사용자가 명시한 `face_id`만 이동하거나 오검출로 표시하고, 언급하지 않은 얼굴은 기존 자동 결과를 유지합니다. 원래 자동 결과와 사진은 변경하지 않습니다. 사용자 확인을 반영한 결과는 자동 정확도 향상이 아니며, 같은 정답으로 자동 성능 평가를 할 수 없습니다.

먼저 원래 **자동 분석 결과 폴더**를 지정하고 해당 실행에 묶인 빈 JSON 양식을 만듭니다. 아래 경로는 예시이며 실제 결과 폴더로 바꾸세요. 결과와 수동 기록은 Git에서 제외된 로컬 폴더에 보관합니다.

macOS:

```bash
FACE_SOURCE='results/run_실제폴더'
.venv/bin/python -m face_lab corrections-template --source "$FACE_SOURCE" --output results/face_corrections.json
```

Windows PowerShell:

```powershell
$FaceSource = '.\results\run_실제폴더'
.\.venv\Scripts\python.exe -m face_lab corrections-template --source "$FaceSource" --output .\results\face_corrections.json
```

양식의 세 해시와 schema는 유지하고, 원래 결과 화면·`faces.csv`에서 확인한 얼굴 ID만 아래 형식으로 채웁니다. 아래 ID는 설명용입니다. `assignments`의 값은 원래 결과의 대상 그룹 번호 또는 새 `personNNN` 번호이고, `false_detections`는 오검출 이유입니다. 같은 그룹의 모든 얼굴을 이동하려면 각 얼굴 ID를 명시해야 합니다. 한 얼굴을 두 항목에 동시에 넣을 수 없습니다.

```json
{
  "assignments": {"확인한_face_id": "person001"},
  "false_detections": {"오검출_face_id": "손"}
}
```

위 예시는 양식의 **해당 두 항목만** 보여 주므로 기존 양식 전체를 덮어쓰면 안 됩니다. 양식 파일이 이미 있으면 덮어쓰지 않습니다. 편집 후 별도 결과를 생성합니다.

macOS:

```bash
.venv/bin/python -m face_lab correct-groups --source "$FACE_SOURCE" --corrections results/face_corrections.json --output results/manual_corrections
```

Windows PowerShell:

```powershell
.\.venv\Scripts\python.exe -m face_lab correct-groups --source "$FaceSource" --corrections .\results\face_corrections.json --output .\results\manual_corrections
```

출력된 새 폴더의 `index.html`에는 사용자 확인과 미검토 자동 결과가 구분되고, 오검출은 별도 메뉴에 표시됩니다. `corrections.json`, `correction_changes.csv`, `correction_provenance.json`에서 확인 사항·변경 전후·원본 해시를 확인할 수 있습니다. `confirmed_ground_truth.csv`는 명시적으로 확인한 얼굴만 익명 ID를 채운 부분 정답이며, 오검출은 평가에서 제외합니다. 이 CSV로 **원본 자동 실행**을 평가하면 확인된 얼굴에 한정한 부분 지표입니다. 평가 및 새 자동 분석·재그룹화는 원본 자동 실행을 사용합니다. 다른 사진 추출이나 변경된 원본 결과에 이 확인 사항을 자동 적용하지 않습니다.

`review` HTML에서 가명 ID 입력 및 CSV 저장 후 평가 가능. 미입력·제외·불확실 얼굴의 평가 제외. 포함된 미확인은 서로 다른 단독 예측으로 처리; 같은 사람의 연결 누락은 FN 반영. 동일 인물 얼굴 **쌍** 기준 Precision·Recall·F1 계산; 임시 번호 비교 아님. 미검출 얼굴의 평가는 해당 CSV만으로 불가. [검증 기준](TESTING.md) 참고.

## 코드 구조·전처리

### 수동 인물 ID 없는 자동 대표 비교

`compare-automatic-anchors`는 기존 `average-linkage 0.40` 결과에서 정답을 보지 않고 대표 그룹을 고릅니다. 얼굴3개·서로다른사진3장 이상, 같은사진 중복 검토없음, 내부 유사도 중앙값0.50·하위10%0.30 이상, 내부평균과 가장가까운 다른그룹평균 차이0.05 이상, 품질중앙값0.60 이상을 모두 요구합니다. 대표는 기존 품질·다양성 방식으로 최대3개 고정합니다. 선정 기준은 안전성을 보장하는 판정이 아닌 첫 비교 설정입니다.

미확인 재배정과 단독·2개그룹 연결을 기준선의 별도 사본에서 실행합니다. 모든 그룹이 경쟁하며1순위가 대표로 선정된 그룹이어야 합니다. 서로다른 대표사진2개 모두 유사도0.40 이상, 평균0.45 이상, 경쟁차이0.03을 요구하고 단독그룹은 평균0.50·차이0.05로 더 엄격합니다. 그룹 구성원 전원이 통과해야 연결하며, 새 얼굴은 대표로 추가하지 않습니다. 품질 제외는 유지하고 같은사진 충돌은 자동 연결 대신 검토보류합니다.

```bash
.venv/bin/python -m face_lab compare-automatic-anchors --source "results/여러크기 + 0.8/run_20261103T065110_894547Z_b7d781" --ground-truth "results/얼굴크기별_오류진단/ground_truth_all_validated.csv" --target-source "results/인물ID_연결_테스트2/run_실제폴더" --output "results/자동대표_그룹화_비교"
```

원본 완전 자동 예측 후에만 인물정답을 읽어 평가합니다. 사용자 오검출 제외 대조 조건은 별도 표시하며, 정답으로 혼합 그룹을 미리 제외하지 않습니다. `--target-source`는 선택 사항이며, 새 정답이 없는 target에는 정확도·F1을 생성하지 않습니다. Windows에서는 실행 파일을 `.\.venv\Scripts\python.exe`로 바꿉니다. 결과는 새 `run_.../index.html`, 실제 선정·탈락 근거는 각 조건의 `anchor_selection.json`, 얼굴별 후보와 변경 사유는 각 단계의 `decisions.json`에 저장합니다. 기존 사진·벡터·자동 결과는 수정하지 않습니다.

### 확인된 인물 ID로 여러 자동 그룹 연결

`test-identity-links`는 자동 그룹을 보존하고 사용자가 확인한 `face_id`를 별도의 영구 인물 ID에 연결합니다. 혼합 그룹은 확인된 얼굴별로 다른 인물에 연결하며, 검색 사진은 `photo_id`로 중복 제거합니다. 결과의 `identity_store.json`에는 추출·벡터 해시, 얼굴 연결, 변경 전후 이력이 남습니다. 그룹 번호가 바뀌어도 같은 추출의 얼굴 연결은 유지하며, 다른 모델·전처리·추출에는 그대로 적용하지 않습니다.

```bash
.venv/bin/python -m face_lab test-identity-links --run "results/최초 그룹화 유사도 비교/유사도 0.40" --ground-truth "results/얼굴크기별_오류진단/ground_truth_all_validated.csv" --output "results/인물ID_연결_테스트"
```

Windows PowerShell에서는 실행 파일을 `.\.venv\Scripts\python.exe`로 바꿉니다. 결과는 새 `run_...` 폴더의 `index.html`에서 확인합니다. 전체 정답 연결 결과는 수동 확인의 반영이며 자동 정확도 향상이 아닙니다.

추가 배정 실험은 사진 해시로 seed(0~4)·development(5~6)·evaluation(7~9)를 분리합니다. seed에서 확인된 얼굴만 인물의 각 하위 그룹 대표로 사용합니다. 서로 다른 seed 사진의 대표 두 개에 대한 최고 두 유사도의 평균을 인물 점수로 쓰며, 둘 다0.40 이상·평균0.45 이상·경쟁 차이0.03을 요구합니다. 대표는 고정하고, 품질 제외 얼굴은 자동 복귀시키지 않습니다. evaluation 정답은 예측 후 평가에만 사용합니다. 설정은 고정값으로 개발/평가 세트에 동일 적용하며 자동 최적값 선택은 하지 않습니다. 기존 자동 그룹은 전체 사진에서 비지도 생성됐고 같은 인물·연속 사진이 각 세트에 있을 수 있으므로 신규 사진에 대한 독립 검증은 아닙니다.

`edit-identity-links`로 기존 인물 ID에 대한 얼굴별 연결 변경·해제를 새 파일에 저장할 수 있습니다. 변경 JSON은 `extraction`(저장 파일의 해시 객체 복사), `expected_revision`, `changes`(`face_id`: `identity_id` 또는 해제 시 `null`), `reason`을 포함합니다. `--store`, 원본 실행의 `--cache`, `--changes`, 새 `--output`을 지정합니다. 직전 변경 취소는 `--changes` 대신 `--undo`를 사용합니다. 원본 저장 파일은 덮어쓰지 않습니다. 이 명령은 연결 저장소를 갱신하며, 이미 생성된 HTML은 테스트 당시 스냅샷입니다.

새 사진 묶음에는 `apply-identity-dataset`을 사용합니다. `--reference-run`은 기존 확인의 원본 실행(캐시/벡터/얼굴 CSV), `--identity-store`는 그 추출에 연결된 확인 저장소입니다.

```bash
.venv/bin/python -m face_lab apply-identity-dataset --reference-run "results/최초 그룹화 유사도 비교/유사도 0.40" --identity-store "results/인물ID_연결_테스트/run_실제폴더/identity_store.json" --input ljm2_dataset --output "results/인물ID_연결_테스트2"
```

검출0.80·다중크기960/1600/2400·average0.40 설정으로 새로 분석합니다. 같은 파일 내용의 사진도 얼굴 좌표·랜드마크·벡터가 유일하게 일치할 때만 기존 확인을 이어받습니다. 원본 파일명이 같다는 이유로 연결하지 않습니다. 다른 추출의 특징 키는 분리하고 동일사진은 SHA로 구별하여 자기사진을 대표 지지로 쓰지 않습니다. 새로 예측한 얼굴은 `predicted_links.json`에 미확인 제안으로만 저장하며 `identity_store.json`에는 확인된 연결만 남깁니다. 새 정답이 없으면 Precision/Recall/F1은 계산하지 않습니다. `automatic_index.html`에서 원래 자동 그룹, `index.html`에서 인물 연결과 보류 후보, `정답입력.html`에서 정답 입력을 확인할 수 있습니다. 정답에는 자동 예측을 미리 채우지 않습니다.

모델·전처리·추출·라이브러리 변경은 차단합니다. 호스트 OS/platform만 달라진 경우에 한해, 원본사진 해시를 확인하고 기존 모든 벡터를 재추출해 성분 최대오차1e-5 및 cosine>=1-1e-5를 만족해야 대표 참조를 허용합니다. 검증 근거는 `identity_summary.json`에 기록합니다. 기존 `analyze`/`regroup`의 캐시 검사 정책은 그대로입니다.

### 오검출 제외와 대표 얼굴 비교

`compare-representatives`는 원본 자동 실행, 해당 실행에 묶인 수동 확인 JSON과 부분 정답 CSV를 사용합니다. 확인된 오검출만 제외한 뒤 재그룹화하고, 그 결과에서 고정 대표 얼굴을 이용한 그룹 병합과 미확인 재배정을 **각각 별도 실험**합니다. 인물 정답 ID는 그룹화에 사용하지 않고 부분 평가에만 사용합니다.

대표 평균 유사도 기준은 `--representative-score-threshold 0.40`처럼 지정할 수 있습니다. 기본값은 0.45이며, 이는 최초 average-linkage 그룹화의 유사도 기준과 별개입니다. 대표당 지지 기준 0.40과 최소 2장, 경쟁 후보 차이 0.03은 그대로 유지됩니다. 부분 정답으로 비교한 결과는 독립적인 정확도 평가가 아닙니다.

macOS 예시(실제 파일 경로로 바꾸기):

```bash
.venv/bin/python -m face_lab compare-representatives --source "$FACE_SOURCE" --corrections results/face_corrections.json --ground-truth results/confirmed_ground_truth.csv --input ljm_dataset --output results/representative_comparison
```

Windows PowerShell 예시(실제 파일 경로로 바꾸기):

```powershell
.\.venv\Scripts\python.exe -m face_lab compare-representatives --source "$FaceSource" --corrections .\results\face_corrections.json --ground-truth .\results\confirmed_ground_truth.csv --input .\ljm_dataset --output .\results\representative_comparison
```

대표는 품질과 벡터 다양성으로 그룹당 최대 3장을 고정합니다. 대표 유사도 평균 0.45, 0.40 이상인 대표 최소 2장, 경쟁 후보와 0.03 차이를 요구하는 **실험용 설정**입니다. 그룹 병합은 양쪽 지지 및 서로의 1순위도 확인하고 한 그룹을 연쇄 병합하지 않습니다. 기존에 같은 사진 중복으로 검토 표시된 그룹은 경쟁 후보에는 남기되 자동 병합·재배정 대상으로 쓰지 않습니다. 사용자 확인 오검출을 활용한 결과이며 같은 부분 정답으로 측정하므로 독립 평가가 아닙니다. `comparison.json`, 각 실험의 `representative_diagnostics.json`에서 조건·후보·판정 이유를 확인합니다.

대표 최대 3장과 5장을 비교하려면, 위 명령으로 만든 **기존 3장 비교 폴더**를 `--compare-with`로 지정하고 `--representatives 5`를 추가합니다. 예시의 경로는 실제 파일·폴더로 바꾸세요.

```bash
FACE_REFERENCE='results/representative_comparison/run_실제폴더'
.venv/bin/python -m face_lab compare-representatives --source "$FACE_SOURCE" --corrections results/face_corrections.json --ground-truth results/confirmed_ground_truth.csv --input ljm_dataset --representatives 5 --compare-with "$FACE_REFERENCE" --output results/representative_count_comparison
```

```powershell
$FaceReference = '.\results\representative_comparison\run_실제폴더'
.\.venv\Scripts\python.exe -m face_lab compare-representatives --source "$FaceSource" --corrections .\results\face_corrections.json --ground-truth .\results\confirmed_ground_truth.csv --input .\ljm_dataset --representatives 5 --compare-with "$FaceReference" --output .\results\representative_count_comparison
```

원본 얼굴·임베딩·정답·오검출 확인 해시, 오검출 제외 후 그룹, 대표 선택 순서와 다른 설정이 같은지 검사합니다. 새 결과의 `representative_count_comparison.html`에서 그룹 병합과 미확인 재배정을 각각 비교합니다. 비교 화면의 3장 결과 링크는 기존 비교 폴더를 참조하므로 두 폴더를 함께 보관하세요. 그룹당 얼굴이 적으면 실제 대표 수는 최대값보다 작습니다.

### 단독 그룹·미확인 얼굴 수동 검토

기존 `compare-representatives` 결과로 별도의 로컬 검토 화면을 만듭니다. 새 검출·재그룹화나 기준 변경은 하지 않습니다. `--comparison`에는 기존 비교의 실제 상위 폴더를 지정합니다.

macOS:

```bash
FACE_COMPARISON='results/representative_comparison/run_실제폴더'
.venv/bin/python -m face_lab singleton-review --comparison "$FACE_COMPARISON" --output results/singleton_review --top-candidates 3
```

Windows PowerShell:

```powershell
$FaceComparison = '.\results\representative_comparison\run_실제폴더'
.\.venv\Scripts\python.exe -m face_lab singleton-review --comparison "$FaceComparison" --output .\results\singleton_review --top-candidates 3
```

기존에 작성한 해당 추출의 정답 CSV가 있으면 `--ground-truth 실제_정답.csv`를 추가해 이전에 확인한 익명 ID를 참고용으로 표시할 수 있습니다. 후보 순위·점수에는 영향을 주지 않고 선택도 미리 채우지 않습니다. 다른 추출의 CSV는 거부합니다.

생성된 `index.html`에서 단독 그룹과 미확인 얼굴을 필터링하고, 후보 그룹의 평균 점수·대표별 개별 유사도·사진 위치를 확인합니다. 그룹 번호는 오검출 제외 후 바뀔 수 있으므로 화면에 원본 번호도 표시합니다. 각 대표와의 **두 얼굴 관계만** 같음·다름·판단 보류로 표시합니다. 같은 쌍이 반대로 다시 표시돼도 선택이 동기화되며, 같은 사람 관계와 다른 사람 관계가 논리적으로 충돌하면 화면에 표시됩니다.

선택 후 **선택 결과 JSON 저장**을 누릅니다. 브라우저 임시 기록에만 의존하지 말고 다운로드한 파일을 Git에서 제외된 `results/` 아래에 보관하세요. 같은 검토 화면에서 이전 JSON을 다시 불러올 수 있으며 다른 추출·비교의 파일은 거부합니다. 이 파일은 명시한 얼굴 쌍의 확인 기록이며 저장만으로 그룹을 병합하거나 새 정답을 자동 전파하지 않습니다.

### 주요 코드

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

## AuraFace 모델 비교

SFace와 다른 얼굴 특징을 확인하는 선택 실험은 [TESTING.md의 AuraFace 비교 실행](TESTING.md#auraface-비교-실행-선택-기능)을 참고하세요. 기본 그룹화 설정은 유지하며, 공개 모델 다운로드 후 사진·특징 추출·평가는 모두 로컬에서 수행합니다. 결과는 `results/AuraFace_비교/run_.../index.html`에 새로 저장합니다.
