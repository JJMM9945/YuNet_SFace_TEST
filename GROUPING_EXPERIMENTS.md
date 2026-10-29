# YuNet + SFace 그룹화 진단·비교

## 이번에 실제로 확인한 결과 (2026-10-29, 로컬 CPU)

**다른 PC의 첫 테스트는 [README의 average 분석 명령](README.md#average-설정으로-분석)을 사용한다.** 작성자가 `average_same_threshold`를 직접 확인해 실제 인물 구분에 가장 가깝다고 판단하여 후속 테스트 시작 설정으로 선택했다. average / .40 / .01 / 최소 1 / 재배정 꺼짐을 명시한다. 정답 평가로 최적임을 확정한 것은 아니다.

macOS와 Windows의 설치·경로·결과 열기는 [노트북 준비 안내](노트북_테스트_준비.md)를 따른다. 아래 작성자 결과 경로는 Git에 포함되지 않으므로 새 clone에서는 존재하지 않는다. 다른 OS의 캐시를 그대로 사용하지 않고 각 PC에서 새로 분석한다. 사진·얼굴·벡터 업로드, 모델 교체, 학습은 수행하지 않았다.

기존 `results/run_20261027T083056_770312Z_616dd9`의 설정과 결과는 요청에 적힌 값과 일치한다. 얼굴 145, 그룹 59, 미확인 18(모호성 16, 경계 이탈 1, 흐림 1)이다. 실제 사용 설정은 유사도 .40, 모호성 폭 .01, 최소 그룹 1, 검출 .8, NMS .3, top-k 5000, 긴 변 1600, 최소 얼굴 40px, 최소 흐림 분산 50이다. **현재 CLI 기본값은 .45/.03/2/검출 .9**이므로 과거 설정을 재현할 때 반드시 옵션을 적어야 한다. 그룹 59는 실제 인원 수가 아니다.

작성자 PC의 새 분석: `results/grouping_diagnostics/run_20261029T051200_243502Z_137567`.
비교 결과: `results/grouping_comparisons/run_20261029T051249_949113Z_16ad0e/index.html`.
동일 average 설정을 별도 적용한 결과: `results/average_runs/run_20261029T053631_919107Z_1ee1a5/` (48그룹·미확인 5개, 비교 결과와 구성원 일치).

동일한 145개 임베딩, 품질 통과 143개로 비교했다. 미확인 전체에는 품질 제외 2개가 포함된다.

| 실험 | 그룹 | 단독 그룹 | 미확인 전체 | 품질 통과 보류 | 품질 제외 | 재배정 |
|---|---:|---:|---:|---:|---:|---:|
| complete .40 | 59 | 25 | 18 | 16 | 2 | 0 |
| average .40 | 48 | 23 | 5 | 3 | 2 | 0 |
| complete .35 | 51 | 20 | 23 | 21 | 2 | 0 |
| complete .45 | 66 | 30 | 20 | 18 | 2 | 0 |
| complete .40 + 재배정 | 59 | 25 | 10 | 8 | 2 | 8 |

연결 방식 비교는 .40에서 방식만 바꾸고, 임계값 비교는 complete에서 임계값만 바꾼다. 재배정은 complete .40의 초기 결과에만 추가한다. 재배정 점수 기준 .45, 경쟁 폭 .03, 대표 최대 3, 최소 지지 2, 개별 지지 기준 .40, 다양성 가중치 .30이다. 재분석 전체 시간 9.99초, 각 비교의 그룹화·진단 약 .17–.21초, HTML·이미지 생성 포함 약 1.96–2.03초였다. 상세 실제 측정값은 `comparison.csv`에 있다. 비교 시간에는 원본 추출 시간이 포함되지 않는다.

개인 사진에 채워진 정답이 없어 Precision·Recall·F1 및 확정 혼합·분리 오류는 **미평가**다. 단순한 숫자 감소로 최적 설정을 자동 선정하지 않았다. 이후 작성자의 직접 검토에 따라 average를 후속 테스트 시작 설정으로 선택했다. 모든 실험에서 같은 사진 중복 그룹 2개가 검토 대상으로 남았다. 이는 확정 혼합 오류가 아니다.

## 판정식과 코드로 확인한 원인

개별 얼굴 유사도는 L2 단위 벡터의 내적 `s(i,j)=clip(x_i·x_j,-1,1)`이고 클수록 가깝다. 코사인 거리는 `d=1-s`이며 작을수록 가깝다.

- complete: 두 군집의 최대 교차 거리, 즉 `1-min(교차 유사도)`를 사용한다.
- average(UPGMA): 모든 교차 얼굴 쌍의 평균 거리, 즉 `1-mean(교차 유사도)`를 사용한다. 군집 크기로 가중하는 표준 average이며 single-link가 아니다.
- 두 방법 모두 거리 `<=1-유사도 기준`에서 자른다. 인원 수는 입력하지 않는다.
- 얼굴 ID를 정렬한 뒤 SciPy에 전달한다. 정확한 동률은 이 순서와 기록된 SciPy 버전에 따라 결정한다. 임의 jitter를 넣지 않는다. 다른 라이브러리 버전 사이의 동률 결과까지 보장하지 않는다.

기준선의 모호성 판정은 다음과 같다.

1. 품질 통과 얼굴로 초기 군집을 만든다.
2. 얼굴 i의 자기 군집 점수 `own`은 자기 자신을 제외한 구성원 유사도의 최소값이다. average 옵션에서는 평균값이다.
3. 각 다른 초기 군집 점수도 같은 집계로 구하고, 그중 최고값을 `other`로 둔다.
4. 초기 군집 크기가 최소 크기 이상이고, 경쟁 군집이 실제 존재하며, `other >= threshold-margin` 및 `own-other <= margin`이면 보류한다.
5. 모호한 얼굴을 제거한 뒤 남은 크기가 최소 크기보다 작으면 나머지도 보류한다. 처음부터 작은 군집과 이 후속 탈락은 별도 사유다.

이 판정은 **최종 1·2위 후보 점수 차이만 검사하는 식과 다르다**. 경쟁 군집 점수가 자기 군집보다 높으면 `own-other`는 음수이며 모호성 조건에 걸릴 수 있다. `candidate_margin_insufficient`는 정렬된 후보의 차이 진단이고, 실제 기준선 보류 여부는 `ambiguity_rule_matched`, `own_competitor_gap`, `competitor_floor` 및 단계 기록으로 확인한다.

단독 초기 군집은 측정할 자기 이웃이 없다. 기존의 `own=1` 정책을 `singleton_prior=1`로 명시해 보존하되, 측정 점수 `own_group_score`는 null이다. 경쟁 후보 부재, 2위 부재, 비교 얼굴 부재도 null이며 -1이나 0으로 대체하지 않는다. 단독 군집 허용은 같은 사람이라는 증거가 아니라 최소 그룹 1 정책이다. average의 군집 평균 거리 절단은 모든 개별 얼굴 쌍의 임계값 통과를 보장하지 않는다.

코드 구조상 complete는 포즈·표정·조명으로 낮아진 한 쌍 때문에 같은 사람을 분리할 수 있고, 분리된 초기 군집들이 서로 경쟁해 모호성을 유발할 수 있다. 실제 보류 예 `pf9b53af8a407886d_f001`의 최근접 개별 유사도는 .6783929, 자기 군집 최소 .4495930, 경쟁 군집 최소 .4452927, 차이 .0043003이다. 개인별 최근접이 높아도 그룹 기준은 다르다는 실제 사례다. **해당 두 군집이 실제 같은 사람인지는 아직 추정**이며 정답 확인이 필요하다. 임계값을 .35로 낮추자 보류가 증가한 것도 병합과 경쟁 판정이 함께 바뀌기 때문이다.

초기 군집 ID를 유지하므로 최종 번호에 빈 번호가 있을 수 있다. 군집이 해체돼도 후보에는 초기 ID와 구성원, `surviving_group_id=null`이 남는다. 실험 사이에는 번호 문자열 대신 구성원 집합으로 소속 변화를 비교한다.

## 전처리와 캐시

Pillow는 open만으로 EXIF 회전하지 않는다. `exif_transpose`를 한 번 적용하고 RGB 배열에서 BGR로 변환한다. OpenCV imread로 다시 회전시키는 경로는 없다. 방향 보정한 원본 크기 W,H와 검출 크기 dw,dh에 대해 `sx=W/dw`, `sy=H/dh`를 따로 계산한다. 반올림으로 두 배율이 조금 달라도 각각 x,w 및 랜드마크 x / y,h 및 랜드마크 y에 적용한다. 검출 점수는 변환하지 않는다. 원본은 파일상의 회전 전 배열이 아니라 **방향 보정된 원본 좌표계**를 뜻한다.

`alignCrop`에는 방향 보정 BGR 원본과 그 원본으로 복원한 15값 row를 함께 전달한다. 공식 OpenCV 4.11 경로는 112×112 정렬 후 feature이며, 내부에서 scale=1, mean=0, swapRB=true로 blob을 만든다. 호출 전에 RGB 교환이나 /255 처리를 추가하지 않는다. 출력은 유한한 128차원인지 검사한 뒤 L2 정규화한다.

공식 근거:
- https://github.com/opencv/opencv/blob/4.11.0/modules/objdetect/src/face_recognize.cpp
- https://docs.opencv.org/4.11.0/d0/dd4/tutorial_dnn_face.html

이번 데이터의 EXIF는 1:40장, 0:3장, 8:3장, 6:5장, 태그 없음:3장이었다. **재추출 벡터가 기존 캐시와 비트 단위로 모두 일치(최대 차이 0)**했고 complete 구성원도 동일했다. 단위 벡터 norm 범위는 .99999988–1.0이다. 확인한 두 예의 박스·랜드마크와 정렬 얼굴도 나란히 점검했다. 전처리 수치 변경은 없어 버전을 유지했으며, 캐시 사용 시 기존 모델 해시·전처리/런타임 버전·입력 파일 해시 검사에 더해 128차원·단위 norm·얼굴 매핑도 검사한다. 모델/전처리 버전이나 사진이 달라지면 재분석이 필요하다. 원본 및 기존 결과 727개 파일의 해시가 유지됐다.

`preprocessing_audit.json`, `preprocessing_preview.jpg`는 이번 실행의 검사 근거다. 박스/랜드마크는 `annotated/`, 얼굴 근처 확대는 `geometry/`, 정렬 얼굴은 `crops/`이며 실험 HTML에 함께 표시된다. 확대는 근처 다른 얼굴의 표시가 포함될 수 있으므로 face_id와 전체 사진을 함께 확인한다.

품질 제외는 원본 최소 변 <40px, 정렬 얼굴 Laplacian 분산 <50, 박스가 얼굴 폭/높이의 10% 허용량보다 원본 경계를 벗어남, 잘못된 검출 좌표 또는 특징 추출 실패다. 검출 점수 필터 .8은 YuNet 단계에서 이미 적용된다. 흐림 값은 원본 ROI가 아닌 112×112 정렬 얼굴 값이다.

## 재배정은 별도 단계

재배정은 기본적으로 꺼져 있다. 초기 군집의 모호성 제거·크기 검사까지 끝난 후 **그룹에 배정된 품질 통과 얼굴만** 대표 후보로 삼는다. 대표 선택은 다음 측정치를 쓴다.

`q = cuberoot(clip(det,0,1) × clip(min(width,height)/160,0,1) × clip(blur/200,0,1))`

첫 얼굴은 q가 가장 큰 얼굴이다. 다음 얼굴은 이미 고른 얼굴과의 최소 정규화 코사인 거리 `div=min((1-s)/2)`를 구해 `(1-w)*q+w*div`가 큰 얼굴을 고른다. 기본 w=.3, 최대 3장이고 동률은 face_id로 정한다. 160과 200은 품질 포화 상수이며 검증된 최적값이 아니다. 실제 값과 선택 점수는 `representatives.json`에 남긴다.

미확인 얼굴은 품질을 통과한 경우만 비교한다. 각 그룹의 **모든 고정 대표와의 유사도 평균**을 그룹 점수로 사용한다. 최고 평균 >= .45, 유사도 >= .40인 대표 최소 2장, 2위가 있으면 최고-2위 >= .03을 모두 요구한다. 대표가 한 장인 그룹도 경쟁 점수에는 표시하지만 단독 최대점 하나로 배정하지 않는다. 경쟁 그룹이 한 개뿐이면 경쟁 차이는 null이며 평균·2장 지지 기준은 여전히 적용한다. 후보가 없으면 계속 보류한다.

대표는 비교 시작 전 고정되며 재배정 얼굴을 추가하지 않는다. 재배정은 기존 그룹에 얼굴을 배정할 뿐 초기 그룹들을 병합하지 않는다. 사전 상태·점수는 `before_reassignment`, `stages`, 초기 `candidates`에, 재배정 후보와 판단은 `reassignment`에 남긴다. 품질 제외 얼굴의 유효 임베딩이 있으면 진단용 후보를 보여줄 수 있지만 자동 배정 대상은 아니다. 같은 사진의 그룹 중복은 재배정 후에도 다시 검토 표시한다. 거울·합성 등의 예외를 고려해 자동 금지하지 않는다.

## 정답 작성과 평가

비교 폴더의 `ground_truth_template.csv`를 복사해 `ground_truth.csv`로 저장하거나 기존 `review` 화면을 사용한다. 기존 수동 정답은 건드리지 않았다. 자동 그룹 ID는 정답에 미리 채우지 않는다.

- 같은 사람의 모든 얼굴에 `manual_person_id=A01` 등 익명 ID를 동일하게 입력한다. 실제 이름은 필요 없다.
- 평가할 정답은 `include=1`, `annotation_status`는 빈칸 또는 `labeled`로 둔다.
- 미입력은 ID와 annotation_status를 비워 둔다. 판단 불가는 ID를 비우고 `annotation_status=uncertain`으로 적는다. review HTML에도 판단 불가 체크박스가 있다.
- 평가 제외는 `include=0`이다. 명시적으로 `annotation_status=excluded`를 쓰려면 ID를 비운다.
- `face_id`, `photo_id`, `_context_sha256`는 바꾸지 않는다. 다른 추출의 정답을 자동 전이하지 않는다.

수동 ID와 그룹 번호의 문자열은 비교하지 않는다. 평가 가능한 모든 정답 얼굴에 대해 동일 인물 쌍의 TP/FP/FN으로 Precision·Recall·F1을 계산한다. 정답이 있는 미확인은 각기 다른 singleton 예측으로 취급하므로 같은 사람 쌍의 재현율에는 FN으로 반영된다. 품질 제외도 정답이 있다면 평가 범위에 포함된다. 미입력·판단 불가는 평가 범위 밖이며, 그 비율과 평가 얼굴 수·자동 배정 비율을 함께 보고한다. 0분모는 null이다. 정답이 하나도 없으면 지표를 계산하지 않는다. 혼합 그룹 및 분리 인물별 얼굴 목록은 `evaluation.json` 또는 evaluate가 만든 JSON에 있다. 미검출 얼굴은 얼굴별 CSV만으로 평가할 수 없어 별도 사진 검토 양식을 유지한다. 검토로 사람이 고친 결과는 자동 성능 평가 대상이 아니다.

## 순서대로 실행 (macOS / Windows)

### 1. 각 PC에서 average 분석부터 실행

[README의 OS별 analyze 명령](README.md#average-설정으로-분석)을 실행한다. `--linkage average --similarity-threshold 0.40 --ambiguity-margin 0.01 --min-group-size 1`을 명시하고 재배정 옵션은 붙이지 않는다. 출력된 `results/average_runs/run_...`의 `index.html`을 확인한다. 원본·기존 결과를 덮어쓰지 않는다.

단일 average 적용은 `analyze` 또는 `regroup`이다. 아래 `compare`는 complete 기준선, average 동일 기준, 선택 방식의 임계값 비교, 재배정 적용을 **모두** 만드는 추가 실험이다. `--linkage average`를 주면 임계값 비교와 재배정의 기반도 average가 된다. 재배정 결과를 `average_same_threshold`와 혼동하지 않는다.

### 2. 로컬 벡터로 비교 (선택)

`run_실제_폴더`는 **자신의 PC에서 출력된 실행 폴더로 교체**한다. source의 추출 설정이 원하는 검출·품질 조건과 같은지 `run_config.json`에서 확인한다.

macOS:

```bash
FACE_SOURCE='results/average_runs/run_실제_폴더'
.venv/bin/python -m face_lab compare --source "$FACE_SOURCE" --input ljm_dataset --output results/grouping_comparisons --linkage average --similarity-threshold .40 --ambiguity-margin .01 --min-group-size 1 --thresholds .35 .45
```

Windows PowerShell:

```powershell
$FaceSource = '.\results\average_runs\run_실제_폴더'
.\.venv\Scripts\python.exe -m face_lab compare --source "$FaceSource" --input .\ljm_dataset --output .\results\grouping_comparisons --linkage average --similarity-threshold .40 --ambiguity-margin .01 --min-group-size 1 --thresholds .35 .45
```

**Windows 제약:** 현재 `compare`는 이미지 공유에 디렉터리 심볼릭 링크를 사용하므로 개발자 모드 또는 심볼릭 링크 생성 권한이 필요할 수 있다. 기본 `analyze`·`regroup`에는 이 제약이 없다. 권한 오류가 나면 우선 README의 단일 average 테스트를 진행한다. Windows에서 이 확장 전체를 재검증한 결과는 아직 없다.

### 3. 정답 작성과 탐색 결과 평가

원본 자동 실행 폴더의 `ground_truth_template.csv`를 `ground_truth.csv`로 복사해 익명 ID를 작성하거나 `review`로 입력한다. 작성한 CSV는 Git에서 제외된 해당 실행 폴더에 보관한다.

macOS:

```bash
.venv/bin/python -m face_lab review --run "$FACE_SOURCE"
# 수동으로 작성한 ground_truth.csv를 저장한 뒤 실행
.venv/bin/python -m face_lab evaluate --run "$FACE_SOURCE" --ground-truth "$FACE_SOURCE/ground_truth.csv"
```

Windows PowerShell:

```powershell
.\.venv\Scripts\python.exe -m face_lab review --run "$FaceSource"
# 수동으로 작성한 ground_truth.csv를 저장한 뒤 실행
.\.venv\Scripts\python.exe -m face_lab evaluate --run "$FaceSource" --ground-truth "$FaceSource\ground_truth.csv"
```

이 전체 자료 평가는 탐색 결과다. 설정 조정 후 공식 확인을 하려면 다음의 개발/평가 사진 분리를 사용한다.

### 4. 개발 세트에서 비교·선택

2번 비교 폴더의 `split_template.csv`를 `results/photo_splits.csv`로 복사하고, 모든 사진에 `split=dev` 또는 `eval`을 입력한다. 미사용 사진은 빈칸/unassigned다. 같은 사진의 얼굴은 한 세트에 속하며 연사·중복 사진도 가능하면 같은 세트에 둔다. 두 세트의 그룹 번호는 독립적이다. 임의 분할은 미리 입력하지 않았다.

macOS:

```bash
.venv/bin/python -m face_lab compare --source "$FACE_SOURCE" --input ljm_dataset --output results/dev_comparisons --subset dev --split-manifest results/photo_splits.csv --ground-truth "$FACE_SOURCE/ground_truth.csv" --linkage average --similarity-threshold .40 --ambiguity-margin .01 --min-group-size 1 --thresholds .35 .45
```

Windows PowerShell:

```powershell
.\.venv\Scripts\python.exe -m face_lab compare --source "$FaceSource" --input .\ljm_dataset --output .\results\dev_comparisons --subset dev --split-manifest .\results\photo_splits.csv --ground-truth "$FaceSource\ground_truth.csv" --linkage average --similarity-threshold .40 --ambiguity-margin .01 --min-group-size 1 --thresholds .35 .45
```

개발 결과의 혼합 오류와 재현율을 검토해 사용자가 명시적으로 설정을 선택한다. 아래는 average .40을 선택하는 예시이며 개발 결과 확인 없이 실행해 최적값으로 확정하는 절차가 아니다. `run_실제_개발폴더`를 바로 앞 명령에서 출력된 폴더로 바꾼다.

macOS:

```bash
FACE_DEV='results/dev_comparisons/run_실제_개발폴더'
.venv/bin/python -m face_lab select-config --comparison "$FACE_DEV" --experiment average_same_threshold
```

Windows PowerShell:

```powershell
$FaceDev = '.\results\dev_comparisons\run_실제_개발폴더'
.\.venv\Scripts\python.exe -m face_lab select-config --comparison "$FaceDev" --experiment average_same_threshold
```

### 5. 고정 설정을 평가 세트에 적용

`selected_config_실제파일.json`을 `select-config`가 출력한 실제 파일명으로 바꾼다. 개발 선택·추출·분할 해시가 같은 경우만 적용되며 평가에서는 그 설정 하나만 실행한다. 평가 정답으로 임계값을 탐색하지 않는다. 평가 CLI에 다른 임계값을 적어도 고정 설정의 값이 적용된다.

macOS:

```bash
.venv/bin/python -m face_lab compare --source "$FACE_SOURCE" --input ljm_dataset --output results/eval_comparisons --subset eval --split-manifest results/photo_splits.csv --selected-config "$FACE_DEV/selected_config_실제파일.json" --ground-truth "$FACE_SOURCE/ground_truth.csv"
```

Windows PowerShell:

```powershell
.\.venv\Scripts\python.exe -m face_lab compare --source "$FaceSource" --input .\ljm_dataset --output .\results\eval_comparisons --subset eval --split-manifest .\results\photo_splits.csv --selected-config "$FaceDev\selected_config_실제파일.json" --ground-truth "$FaceSource\ground_truth.csv"
```

평가 결과를 보고 다시 설정을 고르면 평가 누출이므로 새 보류 평가셋이 필요하다. `all` 비교는 탐색용이며 고정 설정 선택 출처로 사용할 수 없다. dev/eval 결과를 나중에 `evaluate`로 평가할 때도 같은 `--subset`과 `--split-manifest`를 지정한다. 전체 사진으로 묶은 그룹을 평가 얼굴만 필터링해 쓰는 방식은 거부한다.

개인 사진의 dev/eval 비교는 정답과 분할이 아직 없어 실행하지 않았다. 이 경로는 로컬 공개 샘플을 쓰는 Mac 통합 테스트로 검증했으며, 그 테스트 F1은 개인 사진 성능이 아니다.

## 결과 파일과 검사

- `comparison.csv/json`: 실제 전체 설정, 공통 추출 설정, 대상 수, 그룹·보류·제외·재배정·시간·정답 평가·오류 목록.
- `membership_changes.csv`: 기준선 대비 그룹 ID 변경과 실제 구성원 변경을 구분. 재배정 표시 포함.
- 각 실험 `diagnostics.json/csv`: face_id/사진, 품질 값, 초기·최종 그룹, 후보 전체·상위 2개·차이·기준, 최근접 개별 얼굴, 각 단계 상태/사유.
- 각 실험 `representatives.json`: 재배정 대표 선택 근거 및 고정된 대표 ID.
- 각 실험 `index.html`: 인물별 모음, 미확인 후보 링크·점수·사유, 박스/랜드마크·정렬 얼굴, 전체 판정 기록.
- `ground_truth_template.csv`, `split_template.csv`: 수동 입력용 빈 양식. 기존 review CSV/UI와 호환.

실행한 검사는 Python 단위 테스트 33개, 모델이 있는 로컬 통합 테스트 2개, JS review UI 테스트다. 단위 테스트는 거리 방향·average/complete 차이·single-link 연쇄 방지·동률/입력 순서·후보 없음/하나·품질 제외·모호성 후 크기 탈락·대표 다양성·다중 지지·고정 대표·정답 없는 평가 거부·미확인 FN·캐시 norm/shape·EXIF/좌표/BGR 계약을 검사한다. 통합 테스트는 네트워크 소켓을 차단하고 모델 로드/분석/캐시 재사용·버전/입력 불일치 거부·수동 검토와 dev 선택→독립 eval 적용을 확인한다.

macOS:

```bash
.venv/bin/python -m unittest discover -s tests -p 'test_*.py'
.venv/bin/python -m unittest discover -s tests -p 'integration_local.py'
node tests/test_review_ui.js
```

Windows PowerShell (통합 테스트는 앞의 심볼릭 링크 권한 제약과 로컬 공개 샘플 준비 확인):

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p 'test_*.py'
.\.venv\Scripts\python.exe -m unittest discover -s tests -p 'integration_local.py'
node tests\test_review_ui.js
```


현재는 모델 교체 필요 여부를 판단할 정답이 없다. 먼저 품질·정렬을 확인하고, 개발 정답으로 현재 설정을 비교한 뒤 고정된 평가 세트에서 혼합 오류를 억제하면서 분리 오류가 줄었는지 판단한다. 확인된 동일 인물 쌍의 SFace 유사도가 계속 낮거나 다른 인물과 겹쳐 원하는 Precision/Recall을 함께 달성하지 못할 때 모델 비교를 고려할 근거가 생긴다. 이번에는 YuNet·SFace를 유지했다.
