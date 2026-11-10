"""Build a local, image-backed report from existing experiment artifacts.

Run with: PYTHONPATH=. .venv/bin/python scripts/build_test_report.py
No inference, model download, or source-result mutation is performed.
"""
import csv
import html
import json
import os
import re
import shutil
from collections import Counter
from pathlib import Path

from tests.experiments.face_lab.evaluation import pair_metrics, read_annotations
from face_lab.models import sha256
from tests.experiments.face_lab.pipeline import load_vectors


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'tests/private/results/테스트_종합보고서_안드로이드_20261109'
BASE = ROOT / 'tests/private/results/최초 그룹화 유사도 비교/유사도 0.40'
RAW = ROOT / 'tests/private/results/여러크기 + 0.8/run_20261103T065110_894547Z_b7d781'
TRUTH = ROOT / 'tests/private/results/얼굴크기별_오류진단/ground_truth_all_validated.csv'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def csv_rows(path):
    with path.open(encoding='utf-8-sig', newline='') as file:
        return list(csv.DictReader(file))


def pct(value):
    return '미정의' if value is None else f'{value * 100:.2f}%'


def main():
    if OUT.exists():
        raise ValueError(f'Report already exists: {OUT}')
    cache = read(BASE / 'cache.json')
    faces = csv_rows(BASE / 'faces.csv')
    by_id = {face['face_id']: face for face in faces}
    annotations = read_annotations(TRUTH, cache['faces'], cache)
    vectors = load_vectors(BASE, cache)
    metrics = pair_metrics(faces, annotations)
    assert metrics['evaluated_faces'] == 153
    assert metrics['false_positive_pairs'] == 2 and metrics['false_negative_pairs'] == 451
    assert sum(face['status'] == 'false_detection' for face in faces) == 3
    config = read(BASE / 'run_config.json')
    raw_summary = read(RAW / 'summary.json')
    old = read(ROOT / 'tests/private/results/grouping_comparisons/run_20261029T051249_949113Z_16ad0e/comparison.json')
    size = read(ROOT / 'tests/private/results/얼굴크기별_오류진단/전체정답반영/진단.json')
    rules = read(ROOT / 'tests/private/results/병합_분리_규칙_비교/진단.json')
    original_review = read(ROOT / 'tests/private/results/단독그룹_검토/singleton_review_decisions.json')
    for pair in original_review['decisions']:
        a, b = annotations[pair['face_a']], annotations[pair['face_b']]
        if pair['decision'] != 'uncertain':
            assert (a['manual_person_id'] == b['manual_person_id']) == (pair['decision'] == 'same')

    # Re-evaluate saved predictions on the same final truth; no reclustering.
    threshold_results = []
    sources = [TRUTH, BASE / 'faces.csv', BASE / 'cache.json', RAW / 'summary.json',
               BASE / 'run_config.json', ROOT / 'tests/experiments/face_lab/merge_split_diagnosis.py',
               ROOT / 'face_lab/preprocess.py', ROOT / 'face_lab/grouping.py']
    for threshold in ('0.40', '0.38', '0.36'):
        run = BASE.parent / f'유사도 {threshold}'
        rows = csv_rows(run / 'faces.csv')
        assert {f['face_id'] for f in rows} == set(by_id)
        assert read(run / 'cache.json')['features_sha256'] == cache['features_sha256']
        metric = pair_metrics(rows, annotations)
        summary = read(run / 'summary.json')
        threshold_results.append(dict(threshold=threshold, summary=summary, metrics=metric))
        sources.extend([run / 'faces.csv', run / 'summary.json'])
    OUT.mkdir(parents=True)
    (OUT / 'images').mkdir()
    assets = []
    md, body = [], []

    def heading(title, level=2):
        md.append('#' * level + ' ' + title + '\n')
        body.append(f'<h{level}>{html.escape(title)}</h{level}>')

    def paragraph(text):
        md.append(text + '\n')
        body.append('<p>' + html.escape(text) + '</p>')

    def table(headers, rows):
        md.append('| ' + ' | '.join(headers) + ' |\n| ' + ' | '.join(['---'] * len(headers)) + ' |\n' +
                  '\n'.join('| ' + ' | '.join(str(v) for v in row) + ' |' for row in rows) + '\n')
        body.append('<div class="table"><table><thead><tr>' + ''.join('<th>' + html.escape(str(h)) + '</th>' for h in headers) +
                    '</tr></thead><tbody>' + ''.join('<tr>' + ''.join('<td>' + html.escape(str(v)) + '</td>' for v in row) + '</tr>' for row in rows) + '</tbody></table></div>')

    def link(label, path):
        relative = os.path.relpath(path, OUT)
        md.append(f'[{label}](<{relative}>)\n')
        body.append(f'<p><a href="{html.escape(relative, quote=True)}">{html.escape(label)}</a></p>')

    def copy_asset(path, name):
        target = OUT / 'images' / name
        assert path.is_file(), str(path)
        shutil.copy2(path, target)
        assets.append(dict(source=str(path.relative_to(ROOT)), report_file='images/' + name,
                           sha256=sha256(path)))
        return 'images/' + name

    def picture(path, name, caption):
        relative = copy_asset(path, name)
        md.append(f'![{caption}]({relative})\n\n{caption}\n')
        body.append(f'<figure><a href="{relative}"><img class="wide" src="{relative}" alt="{html.escape(caption, quote=True)}"></a><figcaption>{html.escape(caption)}</figcaption></figure>')

    def gallery(ids, caption):
        md.append(caption + '\n')
        body.append('<figure><div class="gallery">')
        for fid in ids:
            face = by_id[fid]
            relative = copy_asset(BASE / 'crops' / f'{fid}.jpg', fid + '.jpg')
            size_px = min(float(face['width']), float(face['height']))
            label = f"{fid} / 自動 {face['group_id'] or '미확인'} / 수동 {annotations[fid]['manual_person_id']} / {size_px:.1f}px"
            label = label.replace('自動', '자동')
            md.append(f'![{label}]({relative})\n\n{label}\n')
            body.append(f'<div class="face"><img src="{relative}" alt="{html.escape(label, quote=True)}"><small>{html.escape(label)}</small></div>')
        body.append('</div><figcaption>' + html.escape(caption) + '</figcaption></figure>')

    heading('YuNet·SFace 얼굴 그룹화 테스트 종합 보고서', 1)
    paragraph('작성일: 2026-11-09 / 대상: ljm_dataset 54장 / 목적: Android 실기기 시험 전 PC 결과와 적용 판단 근거 정리')
    paragraph('현재 권고는 YuNet·SFace, average-linkage 0.40을 Android의 비교 기준선으로 옮기고, 자동 추가 병합·자동 분리는 보류하는 것이다. 같은 사람 분리는 남아 있으나 기준 완화 시 다른 사람 혼합이 증가했다. 이 설정이 제품 최적값이거나 Android에서 검증됐다는 의미는 아니다.')
    paragraph('이 보고서에는 개인 사진에서 생성된 결과 이미지가 포함되어 있다. 이미지와 보고서는 로컬 파일로 보관하며 외부로 업로드하지 않았다. 결과 폴더 전체를 함께 보관하면 이미지가 유지된다.')

    heading('1. 이번 테스트가 답하려던 질문')
    paragraph('얼굴 탐지는 “사진에서 얼굴 위치를 찾는 것”, 얼굴 인식·그룹화는 “찾은 얼굴 중 누가 같은 사람인지 묶는 것”이다. 같은 사람을 여러 그룹으로 나누는 분리 오류와 다른 사람을 한 그룹으로 묶는 혼합 오류를 구별했다. 목표는 그룹 수를 줄이는 것이 아니라 혼합을 억제하면서 분리를 줄이는 것이다.')
    paragraph('사람이 적은 수의 사진을 보고 직접 묶은 결과는 정답을 입력하는 데 사용했다. 수동으로 바꾼 그룹 자체를 자동 성능 향상이라고 계산하지 않았다. 얼굴 ID가 기준이며 실행별 person 번호와 실제 인물 ID 문자열의 일치로 평가하지 않는다.')
    paragraph('읽는 순서: 설정 선택 근거는4절, 실제 오류 사진은5절, 평가 코드의 한계는7절, Android 구현·실험 절차는8~10절을 우선 확인하면 된다.')

    heading('2. 데이터·설정·평가 범위')
    table(['항목', '실제 확인한 값'], [
        ['사진', '54장, 개인 사진 한 묶음'],
        ['과거 단일 크기 실행', '검출 145개; complete 59그룹/미확인18, average 48그룹/미확인5'],
        ['다중 크기 검출 0.80 원본', '검출 후보 156개, 49그룹, 미확인7(품질 제외2 포함)'],
        ['사용자 오검출 제외 후 기준선', '실제 얼굴153개, 48그룹, 단독24그룹, 미확인5, 품질 제외0'],
        ['정답', '153개 얼굴 전부 익명 인물 ID 입력, 오검출3개 제외'],
        ['정답의 서로 다른 ID', str(len({a['manual_person_id'] for a in annotations.values() if a['include'] == '1'})) + '개: 검출된 얼굴 안에서만 센 인물 ID 수'],
        ['검출 입력 긴 변', '960 / 1600 / 2400px; 원본보다 크게 확대하지 않음'],
        ['YuNet 검출 / NMS / top_k', '0.80 / IoU 0.30 / 5000'],
        ['얼굴 품질 기준', '원본 얼굴 상자 짧은 변 40px, 정렬 얼굴 Laplacian 분산 50'],
        ['그룹화', 'average-linkage, 코사인 유사도0.40, 모호성폭0.01, 최소그룹1'],
        ['PC 실행', config['runtime']['platform'] + '; OpenCV ' + config['runtime']['opencv'] + '; CPU, 1스레드'],
    ])
    paragraph('검출 후보 156개 중 3개를 사람이 얼굴이 아니라고 확인하여 제외했다. 따라서 최종 기준선은 “사람이 확인한 오검출 제외 정보가 있는 평가”다. 새 사진을 전부 자동 처리할 때도 153개만 남는다고 가정할 수 없다. 원본에는 실제 얼굴이 몇 개 있는지 전부 위치를 표시하지 않았으므로 YuNet의 검출 Recall은 아직 계산하지 못했다.')
    paragraph('중요: README/CLI의 analyze 기본 검출 점수는 현재 0.75다. 이 보고서의 평가 기준선은 0.80이므로 Android와 PC에서 0.80을 명시해야 한다. 두 값은 인물 그룹화 유사도0.40과 서로 다른 설정이다.')
    paragraph('사용자 정답 안의 인물 ID34개와 자동 그룹48개는 같은 개념이 아니다. 자동 그룹이 실제 인원이라고 가정하지 않았다. 검출된153개 얼굴에 대한 익명 ID를 비교하며 이름이나 신원을 모델이 알아내는 기능을 시험한 것도 아니다.')

    heading('3. 실제 진행한 실험과 얻은 결과')
    e = old['experiments']
    table(['실험', '동일 조건에서 변경한 항목', '실행 결과', '판단'], [
        ['complete ↔ average', 'linkage 방식만 변경, 유사도0.40', f"과거145얼굴: complete {e['baseline_complete']['summary']['groups']}그룹/미확인18 → average48그룹/미확인5", '사람이 보기에는 개선; 당시 정답이 없어 정확도 우위는 확정 못함'],
        ['검출0.75 ↔ 0.80', '검출 점수만 변경, 다중 크기 동일', '157후보 ↔ 156후보, 두 실행 모두49그룹/미확인7', '놓친 얼굴 정답이 없어 검출률 비교 불가'],
        ['수동 보정·정답 입력', 'face_id별 확인, 오검출 제외', '부분정답67 → 80 → 전체153', '평가 기준 확보; 자동 알고리즘 향상과 구별'],
        ['대표 최대3장 ↔ 5장', '대표 장수만 변경', '두 조건 모두 추가 병합0/미확인재배정0', '대표 수만 늘려서는 해결 안 됨'],
        ['대표 평균0.45/0.40/0.35', '추가 단계의 평균점수만 변경', '0.35만 한 쌍 병합; 재배정은 모두0', '021+035는 다른 사람: 잘못된 병합'],
        ['최초 그룹화0.40/0.38/0.36', '처음 그룹을 만드는 유사도만 변경', '48/46/39그룹, 미확인5/5/3', '낮추면 혼합이 증가; 최신 전체정답 평가를 다음 표에 제시'],
        ['얼굴 크기 진단', '동일 결과를 원본 상자 크기별로 구분', '얼굴별 동료 연결83~88%', '작은 얼굴만 주원인이라는 근거 부족'],
        ['병합·분리 규칙 진단', '상호1순위 평균점수, 같은사진, 내부최소점수', '자동 기준으로 채택할 만큼 안전하지 않음', '정답 개입 한계가 있어 추가 검증 필요'],
    ])
    paragraph('대표 선택은 검출점수·얼굴크기·선명도를 결합한 품질과 임베딩 다양성을 사용했다. 대표 비교는 평균점수, 유사도0.40 이상인 지지대표2장, 경쟁 차이0.03을 함께 요구했다. 미확인5개는 1순위가 단독그룹이라 대표2장 조건을 통과하지 못했다. 3장·5장은 최대치이므로 단독그룹의 실제 대표는 항상1장이다.')
    table(['초기145후보 실험의 원래 이름', '방식/유사도', '그룹', '미확인', '재배정 얼굴'],
          [[name, f"{item['config']['linkage']} / {item['config']['similarity_threshold']:.2f}",
            item['summary']['groups'], item['summary']['unknown_faces'], item['summary'].get('reassigned_faces', 0)]
           for name, item in e.items()])
    paragraph('초기 complete_reassignment에서는 미확인8개가 재배정됐다. 이 결과에는 당시 정답 평가가 없으므로8개가 올바르게 이동했다고 결론내릴 수 없다. 이후 다중크기156후보와 오검출 제외를 적용한 대표3/5장 실험의 재배정0개와는 얼굴집합과 초기그룹이 달라 직접 효과를 비교하지 않는다. 초기 complete0.35에서는 미확인이 오히려 늘었는데, 모호성 보류가 함께 적용되기 때문에 임계값을 낮추면 항상 미확인이 줄어드는 것도 아니다.')
    paragraph('모호성 허용 폭0.010→0.008 등 세분화 실험은 논의만 했고 실행하지 않았다. 이 값은 그룹 병합이 아니라 최초 배정 얼굴을 미확인으로 보류하는 판정에 작용한다. 검출 누락 위치 주석, 다른 인식 모델, Android 실행, 재학습은 수행하지 않았다.')

    heading('4. 전체 153개 정답으로 다시 본 그룹화 기준')
    rows = []
    for result in threshold_results:
        s, m = result['summary'], result['metrics']
        rows.append([result['threshold'], s['groups'], s['unknown_faces'], m['true_positive_pairs'],
                     m['false_positive_pairs'], m['false_negative_pairs'], pct(m['precision']), pct(m['recall']), pct(m['f1'])])
    table(['최초 유사도', '그룹', '미확인', '맞게 연결한쌍 TP', '다른사람 혼합쌍 FP', '같은사람 분리쌍 FN', 'Precision', 'Recall', 'F1'], rows)
    paragraph('이 표는 기존 예측 faces.csv를 최신 전체정답으로 재평가한 수치다. 과거67개 부분정답 평가 수치와 분모가 달라 직접 증감을 성능 향상으로 해석하면 안 된다. 0.36은 F1이 더 높지만 혼합쌍이2→16으로 늘었다. 다른 사람 혼합 억제라는 목적 때문에 F1만으로0.36을 선택하지 않는다.')
    paragraph('Precision은 “자동으로 같은 사람이라고 연결한 얼굴 쌍 중 맞은 비율”, Recall은 “실제로 같은 사람인 얼굴 쌍 중 연결해 낸 비율”이다. 99.72% Precision은 얼굴 탐지 정확도나 사람별 정확도가 아니다. 기준선 TP724, FP2, FN451이며, 같은 사람 정답쌍은1175개다. 미확인5개는 각각 다른 단독 예측으로 계산되어 같은 사람과 연결하지 못한 쌍이 FN에 들어간다.')
    paragraph('기준선의 실제 얼굴 자동 배정률은148/153=96.73%다. 자동 배정됐어도 다른 그룹으로 분리될 수 있으므로 배정률과 Recall은 다르다. 평가 범위는 검출된 얼굴153/153으로100%이나, 사진에서 놓친 얼굴까지100% 평가했다는 뜻은 아니다.')

    heading('5. 결과 이미지로 보는 실패 사례')
    picture(BASE / 'annotated/p8c543e0f77e51512.jpg', '01_detected_and_grouped.jpg',
            '그림1. 실제 검출 상자·랜드마크·자동 그룹 표시. 이 사진의 f002와 f005는 정답상 다른 사람인데 자동 person038로 함께 묶였다. 표시 번호는 이 실행의 임시 그룹 ID다.')
    gallery(['p8c543e0f77e51512_f002', 'p8c543e0f77e51512_f005'],
            '그림2. 혼합 사례 A: 자동 person038, 수동정답 person046/person047. 개별 코사인 유사도 약0.497. 얼굴 짧은 변은 약200/182px로, 매우 작은 얼굴만의 문제가 아니다.')
    gallery(['pe198177494aad66d_f003', 'p09fce8d8ac165d52_f001'],
            '그림3. 혼합 사례 B: 자동 person008, 수동정답 person008/person002. 서로 다른 사진이라 같은 사진 검토 규칙으로 찾지 못한다. 개별 유사도 약0.418.')
    same_ids = ['p35444593207dc461_f001', 'p4b1071b558a25d25_f003', 'p09be138b21a29c45_f001', 'pb124a2d55ee88cb9_f001']
    assert len({annotations[fid]['manual_person_id'] for fid in same_ids}) == 1
    gallery(same_ids, '그림4. 같은 사람 분리 사례: 수동정답은 모두 person001이지만 자동 그룹은 서로 다르다. 모습의 차이와 함께 검토할 수 있도록 실제 SFace 정렬 얼굴을 표시했다. 원인이 각도·조명인지 모델 한계인지는 이 이미지와 점수만으로 확정하지 않는다.')
    gallery(['p2ebe3d32c26f3797_f001', 'pe57665b665caa249_f003', 'p8c543e0f77e51512_f003',
             'pf2d97b55d98e2e99_f003', 'p896ac36cbd7f8ac2_f001', 'pa6c7770be5e797d6_f003'],
            '그림5. 잘못된 추가 병합: 앞3개 자동021, 뒤3개 자동035는 수동정답상 다른 인물이다. 두 그룹의 전체9쌍 평균0.388, 최대0.509. 대표 평균0.35 또는 최초 유사도0.38에서 잘못 합쳐졌다.')
    false_ids = [face['face_id'] for face in faces if face['status'] == 'false_detection']
    picture(BASE / 'montages/false_detection_001.jpg', '06_false_detections.jpg',
            '그림6. 사용자가 간판·비인물·손으로 확인한 오검출3개. 평가에서 수동 제외했으며 YuNet이 자동으로 걸러냈다는 의미는 아니다.')

    heading('6. 저해상도 가설: 확인된 사실과 남은 의문')
    size_rows = []
    for row in size['band_results']:
        q = row['identity_faces']
        size_rows.append([row['size_band'], row['detected_faces'], row['labeled_faces'], q['eligible_faces'],
                          q['faces_with_same_person_mate'], pct(q['success_rate']),
                          f"{row['same_person']['error_pairs']}/{row['same_person']['total_pairs']}"])
    table(['얼굴 상자 짧은변', '실제 검출얼굴', '정답 얼굴', '동일인 동료있는 얼굴', '동료와 같은그룹 얼굴', '한명 이상 연결률', '같은사람 분리쌍/전체쌍'], size_rows)
    paragraph('한명 이상 연결률은 같은 사람이 데이터에 두 번 이상 나온 얼굴 중 같은 자동 그룹에서 그 사람을 한 명이라도 찾은 비율이다. 전부 올바르게 묶였다는 비율이 아니다. 200px 이상 연결률86.25%여도 같은 사람 쌍214/523은 분리되어 있다. 예를 들어 한 사람이5개와5개로 나뉘면 각 얼굴은 동료를 찾지만 두 그룹 사이 연결은 모두 누락된다.')
    paragraph('얼굴 상자 크기는 원본 좌표에서 측정한 픽셀 수이며 실제 광학 해상도·압축·확대 이력·자세·가림을 직접 측정하지 않는다. 작은 구간은6개/11개뿐이고 인물 구성이 달라 인과관계를 결론낼 수 없다. 따라서 “저해상도 때문에 실패했다”와 “크기 영향이 없다” 모두 단정할 수 없다. 현 자료는 큰 얼굴에도 분리·혼합이 존재한다는 점만 뒷받침한다.')

    heading('7. 병합·분리 규칙 결과의 중요한 정정')
    paragraph('코드 재검토 결과 merge_split_diagnosis.py의 group_pair_diagnostics는 수동정답으로 단일 인물만 있는 pure 그룹을 골라 후보 점수와 상호1순위를 계산한다. 실제 앱에서는 어떤 그룹이 pure인지 미리 알 수 없다. 따라서 이전의 개발/확인 병합 후 F1 개선 수치에는 정답 정보가 후보 선택에 개입했고, 자동 시스템의 일반화 성능이나 Android 적용 근거로 사용할 수 없다. 기존 파일은 이력을 위해 보존하고 보고서에서 이 한계를 정정한다.')
    paragraph('같은사진 제외 전 전체 진단0.37의 “올바른3/잘못된4 병합”, 제외 후 “올바른3/잘못된3”도 정답으로 혼합그룹을 미리 제외한 조건부 결과다. 평균점수만으로 안전한 경계를 만들기 어렵다는 관찰에는 도움이 되지만, 실제 자동 파이프라인의 종단 성능으로 제시하면 안 된다.')
    paragraph('사진 해시 분할은 얼굴이 검출된 사진만 이용해 개발32장/확인21장=53장이었다. 원래54장 중 얼굴이 없는1장은 분할 기록에서 빠졌다. 동일·유사 인물과 연속사진이 두 세트에 함께 존재할 수 있고, 이미 전체 자료의 오류를 확인했으므로 완전히 손대지 않은 최종 평가 세트도 아니다. 이전 대화에서 인물 기준 분리를 언급했지만 실제 구현은 사진 기준 분리였다.')
    split = rules['full_data']['split_flags']
    table(['분리 검토 조건', '표시그룹', '실제 혼합그룹 발견', '정상그룹도 표시'], [
        ['같은 사진 여러 얼굴', split['same_photo']['flagged_groups'], split['same_photo']['mixed_groups_caught'], split['same_photo']['pure_groups_flagged']],
        *[[f"내부 최소 유사도 < {r['threshold']:.2f}", r['flagged_groups'], r['mixed_groups_caught'], r['pure_groups_flagged']] for r in split['weak_internal_similarity']],
    ])
    paragraph('같은 사진 검토는 전체48그룹에서 혼합038을 찾았지만008은 찾지 못했다. 이 조건은 검토 표시로 사용하고 거울·사진 속 사진·합성 예외를 고려한다. 내부최소점수로 자동 분리하면 동일 인물 정상그룹을 과도하게 쪼갤 수 있다.')

    heading('8. Android에 옮길 기준선과 전처리 계약')
    table(['단계', '기기에서 맞춰야 할 동작', '근거/검증'], [
        ['모델', '같은 YuNet/SFace ONNX와 SHA-256 사용', '모델 변경과 이식 오류의 영향을 구별'],
        ['이미지 방향', 'EXIF를 정확히1회 적용, RGB/RGBA 디코딩 후 BGR로 명시 변환', 'PC는 Pillow exif_transpose → RGB → BGR'],
        ['검출 리사이즈', '긴변960/1600/2400, 원본보다 확대하지 않음; 실제 정수 크기로 sx/sy 계산', 'W/detectW, H/detectH가 가로/세로 각각 다를 수 있음'],
        ['좌표 복원', '상자x,w 및 랜드마크x는 sx, y,h 및 랜드마크y는 sy 곱함', 'face row15값의 좌표계 점검'],
        ['중복 제거', '원본 좌표에서 global NMS IoU0.30, 검출0.80', 'PC 다중크기와 같은 기준'],
        ['얼굴 정렬', '원본방향 보정 BGR 이미지 + 복원 face row를 alignCrop에 입력', '축소좌표를 원본이미지에 직접 주면 정렬 오류'],
        ['특징', 'FaceRecognizerSF.feature, 128차원 유한값, norm>0, L2정규화', '임의의 추가 /255·평균차감으로 공식 경로를 중복변환하지 않음'],
        ['그룹화', '거리=1-cosine; UPGMA average-linkage; 절단거리0.60', 'single-linkage/최근접 연쇄병합/그룹중심점 평균으로 대체하지 않음'],
        ['모호성', '현재 grouping.py 판정과 동일, 폭0.01, 최소그룹1', '군집화 결과를 그대로 보여주는 것과 달라질 수 있음'],
        ['ID와 캐시', 'face_id와 추출해시 보관, person번호는 임시', '모델·전처리·원본 변경 시 벡터 캐시 재생성'],
    ])
    paragraph('SFace 공식 Java API에는 alignCrop·feature·match가 있다. Android에서 사용하려는 OpenCV 빌드에 해당 API와 DNN 모델 로딩이 포함되는지 먼저 확인해야 한다. Python NumPy/SciPy 코드를 앱에서 그대로 실행하는 방식은 구현되어 있지 않다. average-linkage 구현 시 군집 크기에 따른 모든 교차거리 평균과 동률 처리까지 맞춰야 한다.')
    paragraph('모호성 판정도 Android에서 그대로 재현해야 한다. 얼굴의 자기 군집 점수는 자신을 제외한 동료들과의 평균이며, 단독군집은 별도 prior1.0을 쓴다. 경쟁 군집 평균점수가0.39 이상이고 (자기점수−경쟁점수)가0.01 이하이면 보류한다. 후보가 없으면 경쟁점수나 차이를 임의로 만들지 않는다. 이 규칙은 group_faces의 현재 구현이며 단순히 상위두 점수 차이만 보는 규칙과 다르다.')
    md.append('[OpenCV 4.11 공식 예제](https://docs.opencv.org/4.11.0/d0/dd4/tutorial_dnn_face.html) · [FaceRecognizerSF Java API](https://docs.opencv.org/4.11.0/javadoc/org/opencv/objdetect/FaceRecognizerSF.html)\n')
    body.append('<p><a href="https://docs.opencv.org/4.11.0/d0/dd4/tutorial_dnn_face.html">OpenCV 4.11 공식 예제</a> · <a href="https://docs.opencv.org/4.11.0/javadoc/org/opencv/objdetect/FaceRecognizerSF.html">FaceRecognizerSF Java API</a></p>')

    heading('9. 실기기 시험 순서와 합격 판단')
    table(['순서', '실행 내용', '통과 판단/기록'], [
        ['1. 이식 일치', '같은 모델·사진 SHA, 동일0.80 설정으로10장 골든셋부터 실행', '검출상자·랜드마크·정렬얼굴·벡터를 PC와 나란히 비교; 큰 차이면 임계값 조정보다 전처리부터 수정'],
        ['2. 54장 재현', '동일153정답과 오검출3개 표시를 별도로 사용', '동일입력 얼굴 매칭 후 TP/FP/FN·미확인 비교; 동일얼굴이면 기준 TP724/FP2/FN451'],
        ['3. 탐지 평가', '원본에 모든 실제 얼굴 위치를 수동 표시', '박스매칭 IoU기준을 사전에 고정해 검출 Precision/Recall; 현재 미측정'],
        ['4. 처리시간', 'CPU 기준 cold 시작1회와 warm 반복5회 이상, 단계별시간·전체시간', '기종/SoC/RAM/OS/OpenCV/스레드/백엔드/충전/열상태 기록, 중앙값과p95 보고'],
        ['5. 메모리·열', '최대해상도 사진·다중크기 연속처리, 화면조작과 중단/재시작', '최대PSS·native heap·OOM·ANR·UI반응·열에 따른 시간변화; 기기 측정 전 합격 수치 미정'],
        ['6. 단일크기 대조', '기준선이 재현된 뒤1600 단일크기와3크기를 따로 비교', '검출누락·오검출·속도·메모리를 함께 보고 절충; 아직 안 해 본 기기 실험'],
        ['7. 새 자료 최종검증', '새로운 사진/촬영행사 또는 인물 비중을 구분해 평가', '개발세트로만 설정선택; 평가정답으로 후보필터·대표선택 금지'],
    ])
    paragraph('위 표의10장·warm5회·허용오차 기록은 제안하는 테스트 절차이며 이미 실행한 측정이 아니다. 좌표/임베딩 허용오차의 숫자는 기기 빌드의 반복 측정으로 먼저 정하고 고정한다. face_id가 검출좌표 정렬 순서로 만들어지므로 기기에서 검출 수가 바뀌면 같은 번호만으로 비교하지 말고 원본사진+IoU 박스 매칭으로 얼굴을 대응시켜야 한다.')
    paragraph('골든셋에는 EXIF 회전사진, 작은얼굴, 다인사진, 같은사진 혼합038, 서로다른사진 혼합008, 손·간판 오검출, 얼굴0개사진을 포함한다. 디코딩 이미지 크기·각 입력크기·원본복원 상자/랜드마크·정렬이미지·벡터 norm·PC벡터와의 cosine을 기록한다. 같은 수의 검출만 나왔다고 전처리가 같은 것으로 판단하지 않는다.')
    paragraph('Android 이식 합격은 PC보다 F1이 높아야 한다는 뜻이 아니다. 먼저 동일 연산을 옮겼는지 확인하고, 동일 얼굴·동일 설정에서 새로운 혼합이 발생하면 점수 경계·좌표·정렬 차이를 조사한다. 제품 합격기준(허용 지연, 메모리 한도, 혼합 오류 허용치)은 대상 기종과 UX 목표가 정해진 뒤 명시한다.')
    paragraph('성능 도구 근거: Android 공식 문서는 CPU·메모리·전력 등 자원별 프로파일링을 안내한다. 디바이스에서 실제 측정해야 하며 PC 시간을 Android 시간으로 환산할 근거는 없다.')
    md.append('[Android 성능 검사](https://developer.android.com/topic/performance/inspecting-overview) · [Android Studio 프로파일링](https://developer.android.com/studio/profile)\n')
    body.append('<p><a href="https://developer.android.com/topic/performance/inspecting-overview">Android 성능 검사</a> · <a href="https://developer.android.com/studio/profile">Android Studio 프로파일링</a></p>')

    heading('10. 현재 측정된 비용과 대규모 사진의 위험')
    table(['PC CPU 실제 실행 항목', '기록값', '해석'], [
        ['54장 원본 분석 전체', f"{raw_summary['total_seconds']:.2f}초", '단일실행; 보고서/파일생성 포함'],
        ['YuNet 검출', f"{raw_summary['detection_seconds']:.2f}초", '960·1600·2400 다중크기'],
        ['SFace 특징 단계', f"{raw_summary['feature_seconds']:.2f}초", '정렬·품질계산·crop저장 포함; 순수추론만 아님'],
        ['그룹화/진단 직렬화', f"{raw_summary['grouping_seconds']:.3f}초", 'pipeline.finish 측정구간; 제품폰 지연예측값 아님'],
        ['YuNet 모델 파일', f"{config['extraction_signature']['models']['yunet']['size']:,} bytes", '런타임 RAM 사용량과 다름'],
        ['SFace 모델 파일', f"{config['extraction_signature']['models']['sface']['size']:,} bytes", '런타임 RAM 사용량과 다름'],
    ])
    paragraph('현재 그룹화는 모든 얼굴 간 거리행렬을 만들어 O(N²) 메모리를 사용한다. N=10,000이면 거리 정방행렬 하나만 float32 약400MB, float64 약800MB(10진수 기준)다. condensed 배열·복사·원본사진 메모리 등이 추가된다. 153개에서 실행된 결과가 수만 장 갤러리에서도 그대로 가능하다는 근거는 없다. 먼저 실기기에서 수백/수천 얼굴까지 단계적으로 메모리를 측정해야 한다.')

    heading('11. 다음 구현 방향과 보류할 판단')
    paragraph('1차 Android 테스트는 기존 YuNet·SFace와0.40 기준선 재현에 집중한다. 같은사진 충돌과 상호1순위 그룹은 검토 후보로 제시하고, 사용자가 확인한 face_id만 병합·분리 결과에 반영한다. 단독그룹·미확인은 독립 상태로 유지하며 대표1장 최대점수만으로 자동배정하지 않는다.')
    paragraph('이 진단만으로 다른 모델이 더 정확하다고 결론낼 수 없다. 자동화 수준을 더 높여야 한다면 전처리 일치와 탐지 누락을 확인한 뒤, 동일사진·동일얼굴집합·개발/평가 분리 조건에서 다른 인식모델과 비교한다. 이번 작업에서 모델 교체·학습·Android 앱 제작·기기 배포는 수행하지 않았다.')
    paragraph('현재까지 입증된 것은 “이54장과 입력된 정답에서 보수적0.40 기준선이 완화기준보다 혼합을 덜 만든다”, “작은 얼굴만 문제라는 근거는 부족하다”, “점수평균만으로 안전한 병합·분리를 결정하기 어렵다”이다. 완전한 탐지율, 신규사진 일반화, 실기기속도와 안정성은 아직 검증해야 한다.')
    link('Android 비교 기준 설정(JSON)', OUT / '안드로이드_기준설정.json')

    heading('12. 근거 파일·재현 방법')
    for label, path in [
        ('현재 자동0.40 결과', BASE / 'index.html'),
        ('최종153얼굴 정답', TRUTH),
        ('얼굴크기별 전체정답 진단', ROOT / 'tests/private/results/얼굴크기별_오류진단/전체정답반영/진단.json'),
        ('최초 그룹화 유사도 실험', BASE.parent / 'index.html'),
        ('대표 장수 실험', ROOT / 'tests/private/results/대표_3장_5장_비교/run_20261103T073856_940358Z_a0163f/representative_count_comparison.html'),
        ('병합·분리 진단(정답개입 한계를 함께 읽을 것)', ROOT / 'tests/private/results/병합_분리_규칙_비교/index.html'),
    ]:
        link(label, path)
    paragraph('이 보고서에는 새 모델 실험을 추가하지 않았다. 기존 저장 결과를 최종153얼굴 정답으로 재평가하고, 원본 결과 이미지를 복사하여 첨부했다. 사용자가 수정한 CSV의 원본은 보존하고, 상태 필드만 정규화한 ground_truth_all_validated.csv를 기준으로 사용했다.')
    paragraph('macOS: PYTHONPATH=. .venv/bin/python scripts/build_test_report.py / Windows PowerShell: $env:PYTHONPATH="."; .\\.venv\\Scripts\\python.exe scripts/build_test_report.py. 보고서 폴더가 이미 있으면 덮어쓰지 않으므로 재생성 시 스크립트 OUT을 새 이름으로 지정한다.')
    paragraph('PC 기준선 재현 예시: python -m face_lab analyze --input ljm_dataset --models models --output results/android_reference_new --score-threshold 0.8 --detection-sides 960 1600 2400 --nms-threshold 0.3 --top-k 5000 --min-face-size 40 --min-blur 50 --linkage average --similarity-threshold 0.40 --ambiguity-margin 0.01 --min-group-size 1. 이 명령은 수동 오검출 제외를 자동 적용하지 않으므로 원본156후보 실행과 비교한다.')

    payload = dict(created_date='2026-11-09', evaluation_scope='153 detected faces, manual false exclusions, same-collection evaluation',
                   baseline_metrics=metrics, threshold_evaluations=threshold_results,
                   source_hashes=[dict(path=str(path.relative_to(ROOT)), sha256=sha256(path)) for path in dict.fromkeys(sources)],
                   assets=assets,
                   audit_corrections=['merge_split_diagnosis uses truth-pure groups for candidate ranking; not an autonomous held-out evaluation',
                                      'photo split includes only53 face-containing photos; people and events not disjoint',
                                      'size success means any same-person mate, not full clustering recall'],
                   android_execution='not performed')
    (OUT / '근거_및_수치.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    android_reference = dict(
        status='proposed parity-test baseline; not executed on Android',
        models={key: {name: spec[name] for name in ('name', 'sha256', 'size')}
                for key, spec in config['extraction_signature']['models'].items()},
        backend='OpenCV DNN CPU', threads=1,
        preprocessing=cache['signature']['runtime']['preprocessing'],
        detection=dict(score_threshold=.8, nms_threshold=.3, top_k=5000,
                       detection_sides=[960, 1600, 2400], upscale=False),
        quality=dict(min_original_face_short_side=40, min_aligned_laplacian_variance=50),
        recognition=dict(dimensions=128, l2_normalize=True, input='BGR through FaceRecognizerSF alignCrop/feature'),
        grouping=dict(linkage='average-UPGMA', cosine_similarity_threshold=.4,
                      cosine_distance_cutoff=.6, ambiguity_margin=.01, min_group_size=1,
                      automatic_extra_merge=False, automatic_split=False),
        pc_reference=dict(raw_detected_candidates=156, raw_groups=49, raw_unknown_faces=7,
                          user_excluded_false_detections=3, evaluated_faces=153,
                          groups_after_user_false_exclusions=48, unknown_after_exclusions=5,
                          tp=724, fp=2, fn=451),
        warning='User false-detection exclusions are evaluation annotations, not an automatic detector capability.')
    (OUT / '안드로이드_기준설정.json').write_text(json.dumps(android_reference, ensure_ascii=False, indent=2), encoding='utf-8')
    (OUT / '테스트_종합보고서.md').write_text('\n'.join(md), encoding='utf-8')
    style = '''body{font:16px -apple-system,BlinkMacSystemFont,"Malgun Gothic",sans-serif;line-height:1.8;color:#172b35;margin:0;background:#f4f6f8}main{max-width:1120px;margin:auto;padding:38px;background:white}h1{font-size:30px;line-height:1.4}h2{margin-top:44px;border-bottom:2px solid #dbe5eb;padding-bottom:8px}p{overflow-wrap:anywhere}.table{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:14px}th,td{border:1px solid #cbd6df;padding:9px;text-align:left;vertical-align:top}th{background:#edf4f8}figure{margin:24px 0;padding:15px;background:#f7f9fb;border:1px solid #d8e1e8}figcaption{font-size:14px;margin-top:12px}.wide{display:block;max-width:100%;max-height:760px;margin:auto}.gallery{display:flex;gap:16px;flex-wrap:wrap}.face{max-width:155px}.face img{width:112px;height:112px;object-fit:contain;background:#ddd}.face small{display:block;font-size:11px;line-height:1.5;overflow-wrap:anywhere}a{color:#075aa4}.print{font-size:14px;padding:10px;border:1px solid #bbb;background:#f9fbfc}@media print{body{background:white}main{max-width:none;padding:0;font-size:10pt}h2{break-after:avoid}table,figure{break-inside:avoid}.print{display:none}.wide{max-height:460px}}'''
    page = '<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>YuNet·SFace 테스트 종합 보고서</title><style>' + style + '</style><main><p class="print">이미지는 로컬 첨부 파일입니다. 이미지 클릭으로 원본 결과를 확대할 수 있습니다. 브라우저 인쇄에서 PDF 저장도 가능합니다. <a href="테스트_종합보고서.md">Markdown</a> · <a href="근거_및_수치.json">근거 JSON</a></p>' + '\n'.join(body) + '</main></html>'
    (OUT / 'index.html').write_text(page, encoding='utf-8')
    links = re.findall(r'(?:href|src)="([^"]+)"', page)
    missing = [value for value in links if not value.startswith(('https://', '#')) and not (OUT / html.unescape(value)).exists()]
    assert not missing, missing
    print(json.dumps(dict(report=str(OUT / 'index.html'), images=len(assets),
                          evaluated_faces=metrics['evaluated_faces'], links_checked=len(links)), ensure_ascii=False))


if __name__ == '__main__':
    main()
