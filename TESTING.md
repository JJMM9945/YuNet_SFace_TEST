# 테스트 결과와 기술 기준

최신 비교 기준일: **2026-10-29**. 처음 테스트하는 사람은 [README의 average 설정](README.md#먼저-사용할-테스트-설정)과 [macOS / Windows 준비 안내](노트북_테스트_준비.md)를 따릅니다. 아래 2026-10-27·26 내용은 과거 complete 기준선과 당시 검증 이력입니다.

## 현재 후속 테스트 시작 설정: average (2026-10-29)

프로젝트 작성자가 `average_same_threshold`의 얼굴 모음을 직접 확인해 가장 실제 인물 구분에 가깝다고 판단했고, **후속 테스트 시작 설정으로 average를 선택**했습니다. 모든 데이터에 대한 최적 설정이나 정답 기반 성능 확정은 아닙니다.

- 모델: YuNet + SFace 유지.
- 그룹화: **average-linkage**, 유사도 **0.40**, 모호성 폭 **0.01**, 최소 그룹 **1**.
- 검출·전처리: 검출 .80, NMS .30, top-k 5000, 긴 변 최대 1600px, 최소 얼굴 40px, 최소 흐림 분산 50.
- 미확인 재배정: **꺼짐**. 대표 얼굴 재배정 실험과 구분합니다.
- 코드 기본값은 여전히 complete / .45 / .03 / 최소 2 / 검출 .9이므로 실행 옵션을 명시합니다.

Mac에서 동일 54장·145개 벡터(품질 통과 143개)로 실행한 결과입니다.

| 실행 | 그룹 | 단독 그룹 | 미확인 전체 | 품질통과 보류 | 품질 제외 | 재배정 |
|---|---:|---:|---:|---:|---:|---:|
| 과거 방식 complete .40 재현 | 59 | 25 | 18 | 16 | 2 | 0 |
| **average .40 (`average_same_threshold`)** | **48** | **23** | **5** | **3** | **2** | **0** |
| complete .35 | 51 | 20 | 23 | 21 | 2 | 0 |
| complete .45 | 66 | 30 | 20 | 18 | 2 | 0 |
| complete .40 + 별도 재배정 | 59 | 25 | 10 | 8 | 2 | 8 |

average의 미확인 5개는 모호성 3개·박스 경계 이탈 1개·흐림 1개입니다. 같은 사진 중복 그룹 2개는 검토 대상으로 남았습니다. 이는 확정 혼합 오류가 아닙니다. **48그룹은 실제 인원 수가 아니며, 그룹·미확인 감소만으로 성능 향상을 결론 내리지 않습니다.** 개인 사진 전체 정답이 없어 Precision·Recall·F1 및 확정 혼합·분리 오류는 미평가입니다.

2026-10-29 Mac 재분석 환경은 Python **3.11.18**, OpenCV 4.11.0(패키지 4.11.0.86), NumPy 2.3.1, Pillow 11.3.0, SciPy 1.17.1입니다. 재추출한 145개 벡터는 기존 캐시와 비트 단위로 일치했고 complete 기준선의 구성원도 같았습니다. 원본·기존 결과 727개 파일의 해시가 유지됐습니다. 전체 재추출은 약 9.99초, average 비교의 그룹화·진단은 약 .18초, 결과 생성 포함 약 1.96초였습니다. 후자의 시간은 기존 벡터 재사용이며 새 추출 시간을 포함하지 않습니다.

**검사 이력:** Mac에서 Python 단위 테스트 33개, 로컬 통합 테스트 2개, JS 검토 UI 검사 통과. HTML 7페이지의 로컬 링크 6,934개에서 누락이 없었습니다. 통합 테스트는 소켓을 차단하고 캐시 검증·개발 설정 선택·독립 평가 적용을 검사했습니다. 공개 샘플의 테스트 지표는 개인 사진 성능이 아닙니다. 이번 average 확장 전체를 Windows에서 다시 실행한 기록은 아직 없습니다. Windows 담당자는 새 가상환경에서 `check`와 단위 검사 후 README의 average 분석 명령을 실행해 결과를 기록합니다.

작성자 PC에만 있는 실행 경로(Git 제외, 새 clone에서 열리지 않음):

- 비교: `results/grouping_comparisons/run_20261029T051249_949113Z_16ad0e/average_same_threshold/`
- 동일 설정 별도 적용: `results/average_runs/run_20261029T053631_919107Z_1ee1a5/` — 비교의 average와 구성원 일치 확인.
- 기준 벡터 재추출: `results/grouping_diagnostics/run_20261029T051200_243502Z_137567/`

실행 명령과 정답 작성·평가 세트 분리 절차는 [GROUPING_EXPERIMENTS.md](GROUPING_EXPERIMENTS.md)에 있습니다. 다른 PC에서는 위 경로를 source로 복사해 쓰기보다 먼저 새 `analyze`를 실행합니다.

## 과거 기록: 맥 노트북 설치·complete 실행 (2026-10-27)

macOS **27.0.1·arm64**, 기존 Homebrew **Python 3.11.17**로 프로젝트 안에 `.venv/`를 생성했습니다. 설치 버전은 OpenCV **4.11.0.86**, Pillow **11.3.0**, NumPy **2.3.1**, SciPy **1.17.1**입니다. YuNet·SFace 모델과 라이선스를 내려받고 고정 크기·SHA-256 및 CPU 계산을 확인했습니다. 분석 코드·임계값은 변경하지 않았습니다.

기존 의존성을 그대로 설치했을 때 SciPy 1.15.3의 `_spropack` 바이너리 로딩 실패와 NumPy 2.2.6의 행렬 곱 경고를 재현했습니다. `requirements.txt`에서 macOS에만 위 호환 버전을 고정하고, Windows 등 다른 OS는 기존 버전을 유지합니다. 근거는 [SciPy 오류 보고](https://github.com/scipy/scipy/issues/25635), [SciPy 1.17의 PROPACK C 전환](https://docs.scipy.org/doc/scipy/release/1.17.0-notes.html), [NumPy 2.3.1의 Accelerate 경고 수정](https://numpy.org/doc/stable/release/2.3.1-notes.html)입니다.

| 개인 사진 자동 분석 | 데스크탑 기록 | 이번 맥 |
|---|---:|---:|
| 입력 사진 | 54 | 54 |
| 검출 얼굴 | 145 | 145 |
| 그룹 | 59 | 59 |
| 미확인 얼굴 | 18 | 18 |
| 검출 0개 사진 | 3 | 3 |
| 사진 처리 실패 | — | 0 |
| 전체 처리 시간 | 약 18.3초 | 약 9.3초 |

설정은 검출 0.8·최대변 1600·비교 0.40·margin 0.01·최소 그룹 크기 1입니다. 미확인 사유도 그룹 비교 모호성 16개·박스 경계 1개·흐림 1개로 기록과 같습니다. 위 개수의 일치가 얼굴별 배정 전체의 동일성이나 정확도를 보장하지는 않습니다. 데스크탑 결과·사진 해시는 이 clone에 없어 직접 비교하지 않았습니다. 시간은 각 환경의 단일 실행 기록입니다.

- [맥 최종 자동 결과](results/run_20261027T083056_770312Z_616dd9/index.html)
- [수동 검토·CSV 입력 화면](results/run_20261027T083056_770312Z_616dd9/review_20261027T083125_400772Z.html)
- [미확인 얼굴 비교](results/run_20261027T083056_770312Z_616dd9/comparison_20261027T083125_645968Z/index.html)
- [맥 검증 기록](results/run_20261027T083056_770312Z_616dd9/mac_validation.json)

최종 환경에서 `pip check`, 모델 계산, **단위 테스트 16개**, JavaScript DOM 모의 검사, 공개 샘플 통합 검사(네트워크 차단 포함)가 모두 통과했습니다. 실제 사진 분석은 `-W error::RuntimeWarning`으로도 성공했습니다. 벡터 145×128은 모두 유한·단위 길이이며, 유사도 행렬과 float64 직접 합산의 최대 차이는 약 **4.19×10⁻⁷**입니다. 생성 HTML의 로컬 링크·이미지 1,389개가 모두 존재하고, 분석 전후 개인 사진 54장의 SHA-256이 일치합니다.

로그는 `validation_data/mac_20261027/`에 있습니다. 원본 사진·모델·결과·샘플·가상환경은 Git 제외 상태입니다. NumPy 변경 전 초기 실행도 `results/run_20261027T082946_858799Z_53e57f/`에 남겨두었으나, 당시 complete 결과를 재현하려면 위 최종 실행을 사용합니다. 현재 후속 테스트는 문서 상단의 average 설정으로 새로 실행하세요. 현재 `gt_dataset/`은 없어 맥에서 GT 평가는 실행하지 않았고, 개인 사진 정답 평가·수동 보정도 수행하지 않았습니다.

## 데스크탑에서 보관한 결과 (2026-10-26)

| 데이터·방식 | 사진 | 검출 얼굴 | 그룹 | 미확인 얼굴 | 검출 0개 사진 | 결과 |
|---|---:|---:|---:|---:|---:|---|
| 개인 사진 자동 | 54 | 145 | 59 | 18 | 3 | [자동 결과](results/run_20261026T141743_391410Z_e7d109/index.html) |
| 개인 사진 사용자 보정 | 54 | 145 | 53 | 18 | 3 | [수정본](results/run_20261026T135430_233888Z_d232cc/index.html) |
| GT 자동 | 750 | 750 | 50 | 0 | 0 | [GT 결과](results/run_20261026T140825_292453Z_69c37a/index.html) |

개인 사진 자동 분석은 약 **18.3초**, GT 분석은 약 **58.1초**였습니다. 그룹 수는 실제 사람 수가 아니며 미확인 수는 얼굴 수입니다. 사진·벡터·결과는 Git에서 제외되므로 clone만 한 PC에서는 위 로컬 링크가 열리지 않습니다.

## 과거 complete 기준선에서 확인한 문제

- 처음에는 검출 얼굴 111개·검출 0개 사진 15장이었습니다. 검출 기준 0.9→0.8 조정 후 145개·3장이 됐습니다. 후보 증가가 검출 정확도 향상을 그대로 뜻하지는 않습니다.
- 초기 모호성 규칙의 ‘자기 그룹 최저 vs 다른 그룹 개별 최고’ 비교를 **양쪽 모두 그룹 최저 기준**으로 맞췄습니다. 규칙 버전은 `complete-link-consistent-ambiguity-v2`입니다.
- 당시 벡터와 비교 기준 0.40·margin 0.01에서 최소 그룹 크기 2→1 변경 시 그룹 34→59개, 미확인 43→18개였습니다. 단독 얼굴 표시 정책의 영향입니다.
- 사용자 확인에 따라 24얼굴을 인물060으로 합치고, 기존 인물054의 두 얼굴을 인물061·062로 분리했습니다. 확인하지 않은 얼굴에는 확대 적용하지 않았습니다.
- 동일 인물로 확인한 쌍 중 유사도 약 **-0.044**, 다른 인물로 확인한 쌍은 약 **0.525**인 사례가 있습니다. 임계값 하나로 두 사례를 모두 해결할 수 없으며 전체 개인 사진 정답은 아직 완성하지 않았습니다.

[미확인 비교 화면](results/run_20261026T141743_391410Z_e7d109/comparison_20261026T142412_114466Z/index.html)에서 확인한 내용:

| 사유 | 수 | 관측 |
|---|---:|---|
| 그룹 비교 모호성 | 16 | 경쟁 그룹 점수가 더 높은 경우 12개, 자기 쪽이 높아도 차이가 0.01 이하인 경우 4개 |
| 박스 경계 규칙 | 1 | 얼굴 크기가 약 674px여도 제외됨. 해상도 부족이라고 단정할 수 없음 |
| 흐림 | 1 | 작은 배경 얼굴 최소변 약 41px, 정렬 blur 분산 약 12.3 |

모호성 16개 중 최근접 개별 유사도 0.70 이상은 10개, 0.80 이상은 4개입니다. 개별 최근접 점수와 실제 보류에 쓰는 그룹 점수는 다릅니다. 이 수치만으로 동일 인물 여부를 확정하지 않았습니다.

## 과거 기록: GT 정답 평가

`gt_dataset/gt_db/s01`~`s50`에 인물별 JPEG 15장씩 있습니다. [제작자 README](https://anefian.com/research/GTDB_README.txt)에 따라 폴더를 정답 ID로 사용했으며 원본 배포 아카이브와의 동일성은 별도 검사하지 않았습니다.

사진마다 검출한 얼굴이 1개여서 **750얼굴 모두 평가**, 각 그룹 15얼굴·50그룹으로 정답 폴더와 일치했습니다. TP=5,250, FP=0, FN=0, **Precision·Recall·F1 모두 1.0**입니다. 사람 수 50을 군집화 입력으로 주거나, 수동 보정·GT 결과에 맞춘 임계값 재선택은 하지 않았습니다.

이 평가는 **단일 검출 얼굴이 폴더의 대상 인물이라는 전제**입니다. 얼굴 위치 정답 파일이 없어 검출 IoU·Precision·Recall은 측정하지 않았고, 모든 박스를 사람이 검토하지도 않았습니다. 개인 단체 사진이나 Android에서도 같은 성능이라는 뜻은 아닙니다.

GT 실행 폴더의 `folder_ground_truth.csv`, `benchmark_provenance.json`, `evaluations/`에 근거가 있습니다. 같은 추출에서 평가를 다시 실행할 수 있습니다.

```powershell
$run = '.\results\run_20261026T140825_292453Z_69c37a'
.\.venv\Scripts\python.exe -m face_lab evaluate --run "$run" --ground-truth "$run\folder_ground_truth.csv"
```

평가는 **얼굴 쌍** 기준입니다. TP는 같은 사람을 함께 묶은 쌍, FP는 다른 사람을 섞은 쌍, FN은 같은 사람을 나누거나 미확인으로 놓친 쌍입니다. Precision은 혼합, Recall은 분리를 함께 살펴보는 지표이고 F1은 둘의 균형을 봅니다. 미확인은 각각 단독 예측으로 포함하며, 분모 0은 `null`입니다. 미검출 얼굴은 이 CSV 평가에 들어가지 않습니다.

## 과거 기록: Windows에서 수행한 코드 검사

Python 3.12.2·OpenCV 4.11.0(패키지 4.11.0.86)·NumPy 2.2.6·Pillow 11.3.0·SciPy 1.15.3, Windows 11 CPU에서 확인했습니다.

- **Python 단위 테스트 16개 통과**: 방향·좌표, 그룹화·모호성·진단 수치, 잘못된 모델, 정답 평가, 수동 보정 충돌·출처·HTML 처리.
- 설치 의존성·구문·CLI·빈 입력·모델 해시/로딩/빈 영상 CPU 계산 통과.
- 공개 샘플 통합 검사 통과: 한글 경로·손상 사진·원본 해시 유지·벡터 정규화·재현성·캐시 무효화·수동 수정 흐름. 소켓을 차단해도 로컬 분석 정상 작동.
- JavaScript DOM 모의 검사 통과: 수동 ID 입력·분리·CSV BOM/따옴표/한글·다른 추출 거부·임시 저장 불가 상황.
- 검출·얼굴 모음·비교 개요 시각 확인, 비교 18개 섹션 링크와 모호성 조건 재현 검사.

```powershell
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe tests\integration_local.py
node tests\test_review_ui.js
```

통합 검사는 `validation_data/official_sample/lena.jpg`가 없으면 skip합니다. 이 샘플은 [OpenCV 공개 샘플](https://github.com/opencv/opencv/blob/4.x/samples/data/lena.jpg)이며 개인 사진 성능 자료가 아닙니다. Node.js는 JS 검사에만 필요합니다.

**아직 미검증:** 개인 사진 전체 정답 기반 성능, EXIF 8방향 전체 실사진 회귀, 실제 브라우저/Excel UI 조작, Android 기기. Python 3.11·맥 노트북 검증은 위 2026-10-27 기록을 참고하세요. 데스크탑 중간 검증 결과는 정리 요청에 따라 삭제했으며 검사를 재실행하면 새 결과가 생깁니다.

## 모델·전처리·그룹화 기준

모델은 요청한 두 파일을 고정했습니다. `prepare-models`만 네트워크 다운로드를 수행하고, 크기와 SHA-256이 다르면 LFS 포인터·HTML·손상·다른 모델을 거부합니다. 기존 잘못된 모델을 자동 덮어쓰지 않습니다.

| 모델 | 라이선스 | 크기 | SHA-256 |
|---|---|---:|---|
| `face_detection_yunet_2023mar.onnx` | MIT | 232,589바이트 | `8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4` |
| `face_recognition_sface_2021dec.onnx` | Apache-2.0 | 38,696,353바이트 | `0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79` |

공식 다운로드: [YuNet ONNX](https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx), [SFace ONNX](https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx). 출처·라이선스는 [YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet), [SFace](https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface)와 `models/`에 보관합니다. 배포 시 해당 라이선스·저작권 고지도 함께 제공합니다. API 근거는 [OpenCV 공식 튜토리얼](https://docs.opencv.org/4.13.0/d0/dd4/tutorial_dnn_face.html)입니다.

| 단계·코드 | 재현할 기준 |
|---|---|
| `models.py` | Python으로 모델 바이트를 읽어 OpenCV ONNX buffer API로 로드. 한글 경로 지원 |
| `preprocess.py` 읽기 | Pillow EXIF 방향 보정 **한 번**, RGB→BGR uint8. 별도 ICC 변환 없음 |
| 검출 입력 | 긴 변 최대 1600, 확대·패딩 없음, 종횡비 유지·정수 `round`·`INTER_AREA`. BGR 그대로 `FaceDetectorYN`에 전달 |
| 좌표 복원 | 실제 원본폭/검출폭, 원본높이/검출높이를 축별 적용. 박스·5개 랜드마크는 방향 보정된 원본 픽셀 기준 |
| 특징 추출 | 방향 보정 원본에서 `alignCrop` 112×112 → `feature` 128차원 → L2 정규화. 0/NaN 벡터 거부 |
| 그룹화 | 코사인 `u·v`(클수록 유사), 거리 `1-cosine`(작을수록 유사). 현재 테스트는 average, 과거 기준선은 complete. 둘 다 single-link가 아님 |
| 모호성 | average는 그룹 평균, complete는 그룹 최소 점수를 사용. 경쟁 그룹 점수 중 최고가 ≥ 임계값−margin이고 자기 점수−경쟁 점수 ≤ margin이면 보류 |
| 품질 | 원본 얼굴 최소변 40px, 정렬 blur 분산 50, 큰 박스 경계 이탈. 경험적 기준 |
| 재현·캐시 | 경로·좌표 정렬, seed 0·CPU·스레드 1·OpenCL off. 사진 목록/내용·모델·전처리/라이브러리/런타임 차이 시 캐시 차단 |

그룹 비교는 미확인 제외 전 클러스터 기준입니다. 단독 자기 클러스터의 1.0은 정책상 prior이며 실제 측정 점수는 null로 기록합니다. 다른 얼굴과의 일치 근거가 아닙니다. complete-linkage도 같은 사람 분리·잘못된 높은 점수에 의한 혼합을 막지는 못합니다. 거리 행렬 메모리는 얼굴 수의 제곱으로 증가합니다. 원본 디코딩·정렬도 메모리를 사용하므로 초고해상도 사진은 개별 실패할 수 있습니다.

## 일부 얼굴만 수동 보정할 때

`correct-groups`는 정확한 원본 실행에 묶인 JSON을 사용합니다. 합칠 얼굴만 지정하고, 서로 다르다는 사실만으로 나머지 인물과의 관계까지 정답으로 만들지 않습니다.

```json
{
  "context_sha256": "추출 컨텍스트 해시",
  "source_faces_sha256": "해당 원본 실행 faces.csv의 SHA-256",
  "same_person_sets": [["face_A", "face_B"]],
  "different_person_pairs": [["face_C", "face_D"]]
}
```

```powershell
.\.venv\Scripts\python.exe -m face_lab correct-groups --source "$run" --corrections .\feedback.json --input .\ljm_dataset
```

위 예시는 형식 안내이며 실행 가능한 정답 데이터가 아닙니다. 다른 추출/원본 그룹의 해시·잘못된 얼굴 ID·모순된 관계는 거부합니다. 보정 후 위반 0은 지시 준수 검사이지 자동 성능 지표가 아닙니다.

## Android에서 따로 확인할 것

같은 모델·OpenCV/ABI·CPU 백엔드 로딩, EXIF/미러·RGB(A)→BGR·리사이즈·좌표·정렬·벡터 수치 일치를 먼저 확인합니다. 그 뒤 메모리·지연·발열·배터리, 사진 권한·데이터 보호, 취소/재개·증분 캐시, 영구 `person_id`와 수동 합치기/분리 관리를 검증합니다. PC 임계값을 검증 없이 확정하지 않습니다.
