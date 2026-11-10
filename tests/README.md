# 테스트와 실험 보관

`experiments/`에는 정리 전 CLI·실험 코드·README·TESTING·추가 의존성을 보존했습니다. 기존 수정 사항도 함께 옮겼습니다. `experiments/README.md`와 `TESTING.md`는 당시 문서이므로 경로와 명령은 과거 기준입니다.

기존 CLI는 저장소 루트에서 `python -m tests.experiments.face_lab --help`로 실행합니다. 예를 들어 현재 설정 재현은 다음과 같습니다.

```sh
python -m tests.experiments.face_lab analyze --input ljm_dataset --models models --output tests/private/results/new_selected_075 --score-threshold 0.75 --detection-sides 960 1600 2400 --linkage average --similarity-threshold 0.40 --ambiguity-margin 0.01
```

과거 결과의 절대 경로·해시 결합은 보존되어 있습니다. 이동 후 참조 경로를 자동으로 고치지 않았으므로 과거 연계 실험을 재실행하려면 인자에 새 경로를 명시하고 필요하면 원본 분석부터 재생성하세요. 보고서 스크립트는 당시 데이터 전용 보관본입니다.

## 무엇을 테스트했는가

| 실험 | 목적 | 서비스 반영 |
|---|---|---|
| 여러크기 + 0.75 / 0.8 | 다중 크기와 검출 점수에 따른 누락·오검출 비교 | 960/1600/2400 + 0.75 선택 |
| complete/average, 0.35~0.45 및 최초 그룹 유사도 비교 | 과도한 분리와 혼합 비교 | average 0.40 선택 |
| 품질·얼굴크기·병합/분리 진단 | 작은 얼굴·흐림과 그룹 오류 분석 | 기본 품질 제외 유지 |
| 오검출 수동 보정 | 사용자 확인 반영 | 실험 보관; 자동 성능과 구분 |
| 대표 3/5장, 자동대표, 그룹 평균 비교 | 대표 지지로 재배정·병합 비교 | 추가 검증 필요; 핵심 엔진에 미반영 |
| 인물ID 연결, 새 데이터셋 | 영구 인물과 자동 그룹 분리·확인 전이 | 이식 설계 참고 |
| AuraFace/SFace | 동일 검출에서 특징 모델과 속도 비교 | SFace 선택; AuraFace는 비교 보관 |

각 실행의 집계 결과는 `results/experiment_summaries.json`에 있습니다. 원본 폴더 상대 이름으로 실행을 식별하며 얼굴별 ID·개인 사진·벡터·절대 경로는 제외했습니다. 수동 정답이 개입한 결과와 완전 자동 결과를 혼합하여 정확도로 계산하지 마세요.

선택한 0.75 실행: 54장, 157얼굴, 49그룹, 미확인 7(모호성 5 + 품질 2). 이 수치는 정확도가 아닙니다.

AuraFace 비교의 탐색적 평가: SFace 기준 0.44에서 Precision 0.9841 / Recall 0.5636 / F1 0.7168, AuraFace 기준 0.52에서 1.0 / 0.1273 / 0.2258. 전체 특징 추출은 각각 1.921초 / 35.418초(PC). 평가 얼굴 45개이며 미검출은 평가하지 않았습니다. 과거 확인 사진으로 나눈 탐색적 분할이므로 독립 테스트가 아닙니다. 이 비교의 원본 검출은 0.8 계열이며 선택한 0.75 전체 파이프라인의 정확도 증거로 쓰지 않습니다.

## 검사

저장소 루트에서:

```sh
python -m unittest discover -s tests -p 'test_*.py'
node tests/test_review_ui.js
node tests/test_singleton_review_ui.js
node tests/test_singleton_review_dom.js
```

로컬 원본 자료는 `private/results/`, `private/validation_data/`에 있습니다. Git에는 올라가지 않습니다. 집계를 재생성하려면 `python tests/export_summaries.py`의 출력을 검토하여 `results/experiment_summaries.json`에 반영합니다.
