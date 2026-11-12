# 현재 기능 검증과 과거 결과

- [짧은 실험 결과 + 이미지 모음](results/README.md)
- [현재 사진 앨범 결과](results/photo_count_comparison.md)
- [검증·정리 기록](results/reorganization_validation.md)

현재 기능 실행:

```sh
python -m face_lab albums --input "내 사진 폴더"
```

현재 코드 자동 검사:

```sh
python -m unittest discover -s tests -p 'test_*.py'
node tests/test_photo_count_view.js
```

tests/에는 현재 기능의 회귀 검사만 남겼습니다. 과거 모델 비교·대표 얼굴·수동 보정·진단 실행 코드와 전용 테스트는 삭제했습니다. results/의 결과 문서·차트·집계 JSON은 유지하고, private/의 과거 웹 결과·복사 사진·벡터·검증 자료는 삭제했습니다. 입력 사진 폴더는 유지합니다.
