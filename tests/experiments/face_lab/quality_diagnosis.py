"""Source-bound face-size diagnostics for an existing automatic grouping run."""
import csv
import html
import json
import os
import re
from collections import Counter
from itertools import combinations
from pathlib import Path

from .evaluation import pair_metrics, read_annotations
from .models import sha256
from .pipeline import write_csv, write_json
from .review import context_hash


DEFAULT_BINS = (60.0, 100.0, 200.0)


def size_band(size, boundaries=DEFAULT_BINS):
    if any(b <= 0 for b in boundaries) or list(boundaries) != sorted(set(boundaries)):
        raise ValueError('Size boundaries must be unique, positive and increasing')
    lower = 0.0
    for upper in boundaries:
        if size < upper:
            return f'{lower:g}–{upper:g}px'
        lower = upper
    return f'{lower:g}px 이상'


def _read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as file:
        return list(csv.DictReader(file))


def _pair_outcome(a, b):
    same_group = bool(a['group_id'] and a['group_id'] == b['group_id'])
    return same_group


def _rates(records):
    total = len(records)
    wrong = sum(record['wrong'] for record in records)
    return dict(total_pairs=total, error_pairs=wrong,
                error_rate=(wrong / total if total else None))


def diagnose_quality(args):
    run = Path(args.run).resolve()
    truth_path = Path(args.ground_truth).resolve()
    output = Path(args.output).resolve()
    if output.exists():
        raise ValueError('Output directory already exists; preserve the previous diagnosis')
    cache = json.loads((run / 'cache.json').read_text(encoding='utf-8'))
    faces = _read_csv(run / 'faces.csv')
    expected = {face['face_id'] for face in cache['faces']}
    if len(faces) != len(expected) or {face['face_id'] for face in faces} != expected:
        raise ValueError('Run face mapping does not match its cache')
    if sha256(run / 'features.npz') != cache['features_sha256']:
        raise ValueError('Feature cache checksum mismatch')
    annotations = read_annotations(truth_path, cache['faces'], cache)
    by_id = {face['face_id']: face for face in faces}
    labeled = [face for face in faces
               if annotations[face['face_id']]['include'] == '1'
               and annotations[face['face_id']]['annotation_status'] == 'labeled'
               and annotations[face['face_id']]['manual_person_id']]
    if not labeled:
        raise ValueError('No labeled faces to diagnose')
    boundaries = tuple(float(value) for value in args.size_boundaries)
    face_rows = []
    for face in faces:
        size = min(float(face['width']), float(face['height']))
        face_rows.append(dict(face_id=face['face_id'], size=size,
                              band=size_band(size, boundaries), status=face['status'],
                              group_id=face['group_id'], quality_reason=face.get('quality_reason', ''),
                              detection_score=float(face['detection_score']),
                              blur_variance=(float(face['blur_variance']) if face['blur_variance'] else None)))
    size_by_id = {row['face_id']: row for row in face_rows}

    same_pairs, different_pairs = [], []
    for first, second in combinations(sorted(labeled, key=lambda face: face['face_id']), 2):
        first_id, second_id = first['face_id'], second['face_id']
        same_person = (annotations[first_id]['manual_person_id'] ==
                       annotations[second_id]['manual_person_id'])
        grouped_together = _pair_outcome(first, second)
        weakest = min(size_by_id[first_id]['size'], size_by_id[second_id]['size'])
        record = dict(face_a=first_id, face_b=second_id, weakest_face_size=weakest,
                      size_band=size_band(weakest, boundaries),
                      grouped_together=grouped_together,
                      wrong=(not grouped_together if same_person else grouped_together))
        (same_pairs if same_person else different_pairs).append(record)

    ordered_bands = [size_band(0, boundaries)]
    ordered_bands += [size_band(value, boundaries) for value in boundaries]
    ordered_bands = list(dict.fromkeys(ordered_bands))
    band_rows = []
    for band in ordered_bands:
        all_faces = [row for row in face_rows if row['band'] == band and row['status'] != 'false_detection']
        labeled_faces = [row for row in all_faces if annotations[row['face_id']]['annotation_status'] == 'labeled']
        same = [record for record in same_pairs if record['size_band'] == band]
        different = [record for record in different_pairs if record['size_band'] == band]
        identity_faces = []
        for face in labeled_faces:
            fid = face['face_id']
            identity = annotations[fid]['manual_person_id']
            mates = [other for other in labeled if other['face_id'] != fid
                     and annotations[other['face_id']]['manual_person_id'] == identity]
            if mates:
                identity_faces.append(dict(face_id=fid, found_same_person=any(
                    _pair_outcome(by_id[fid], by_id[other['face_id']]) for other in mates)))
        band_rows.append(dict(size_band=band, detected_faces=len(all_faces),
                              labeled_faces=len(labeled_faces),
                              unknown_faces=sum(row['status'] == 'unknown' for row in all_faces),
                              identity_faces=dict(eligible_faces=len(identity_faces),
                                                  faces_with_same_person_mate=sum(row['found_same_person'] for row in identity_faces),
                                                  success_rate=(sum(row['found_same_person'] for row in identity_faces) / len(identity_faces)
                                                                if identity_faces else None)),
                              same_person=_rates(same), different_people=_rates(different)))

    reviewed = None
    if args.review_decisions or args.review_manifest:
        if not (args.review_decisions and args.review_manifest):
            raise ValueError('Review decisions and manifest must be provided together')
        decisions_path, manifest_path = Path(args.review_decisions).resolve(), Path(args.review_manifest).resolve()
        decisions = json.loads(decisions_path.read_text(encoding='utf-8'))
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        provenance = manifest['provenance']
        if decisions.get('provenance') != provenance or decisions.get('conflicts'):
            raise ValueError('Review decisions do not match the manifest or contain conflicts')
        if provenance['source_context_sha256'] != context_hash(cache) or provenance['features_sha256'] != cache['features_sha256']:
            raise ValueError('Reviewed pairs use a different extraction')
        allowed = {tuple(sorted((pair['query_face_id'], pair['candidate_face_id'])))
                   for pair in manifest['pairs']}
        seen, reviewed_pairs = set(), []
        for decision in decisions['decisions']:
            pair = tuple(sorted((decision['face_a'], decision['face_b'])))
            if pair not in allowed or pair in seen or decision['decision'] not in {'same', 'different', 'uncertain'}:
                raise ValueError('Invalid or duplicate reviewed pair')
            seen.add(pair)
            if decision['decision'] == 'uncertain':
                continue
            a, b = pair
            weakest = min(size_by_id[a]['size'], size_by_id[b]['size'])
            grouped = _pair_outcome(by_id[a], by_id[b])
            reviewed_pairs.append(dict(face_a=a, face_b=b, decision=decision['decision'],
                                       weakest_face_size=weakest, size_band=size_band(weakest, boundaries),
                                       grouped_together=grouped,
                                       wrong=(not grouped if decision['decision'] == 'same' else grouped)))
        reviewed = []
        for band in ordered_bands:
            rows = [record for record in reviewed_pairs if record['size_band'] == band]
            reviewed.append(dict(size_band=band,
                                 same=_rates([record for record in rows if record['decision'] == 'same']),
                                 different=_rates([record for record in rows if record['decision'] == 'different'])))

    result = dict(run=str(run), run_faces_sha256=sha256(run / 'faces.csv'),
                  features_sha256=cache['features_sha256'], truth_sha256=sha256(truth_path),
                  size_definition='minimum of original-coordinate face box width and height; pair band uses the smaller face in the pair',
                  boundaries=list(boundaries), evaluated_faces=len(labeled),
                  overall_metrics=pair_metrics(faces, annotations),
                  band_results=band_rows, reviewed_pair_results=reviewed,
                  limitations=('Partial, selected labels from the same photo set. Pair rows share faces and are not independent. '
                               'Detected face box size is a proxy for available detail, not measured image resolution.'))
    output.mkdir(parents=True)
    write_json(output / '진단.json', result)
    _render(output, run, result)
    print(f'Quality diagnosis: {output / "index.html"}')
    return result


def make_quality_review(args):
    run = Path(args.run).resolve()
    truth_path = Path(args.ground_truth).resolve()
    target = Path(args.output).resolve()
    if target.exists():
        raise ValueError('Review output already exists; preserve the previous input page')
    cache = json.loads((run / 'cache.json').read_text(encoding='utf-8'))
    faces = _read_csv(run / 'faces.csv')
    expected = {face['face_id'] for face in cache['faces']}
    if len(faces) != len(expected) or {face['face_id'] for face in faces} != expected:
        raise ValueError('Run face mapping does not match its cache')
    annotations = read_annotations(truth_path, cache['faces'], cache)
    min_size = float(getattr(args, 'min_face_size', 0))
    max_value = getattr(args, 'max_face_size', None)
    max_size = float(max_value) if max_value is not None else None
    if min_size < 0 or (max_size is not None and max_size <= min_size):
        raise ValueError('Face size range must satisfy 0 <= minimum < maximum')
    visible = []
    asset_prefix = os.path.relpath(run, target.parent) + '/'
    for face in faces:
        if not re.fullmatch(r'[A-Za-z0-9_-]+', face['face_id']):
            raise ValueError('Invalid generated face ID')
        annotation = annotations[face['face_id']]
        face.update(manual_person_id=annotation['manual_person_id'], include=annotation['include'],
                    annotation_status=annotation['annotation_status'], notes=annotation['notes'],
                    crop_path=asset_prefix + f"crops/{face['face_id']}.jpg" if face.get('crop_path') else '')
        size = min(float(face['width']), float(face['height']))
        in_range = size >= min_size and (max_size is None or size < max_size)
        unlabeled = annotation['annotation_status'] == 'unlabeled' and not annotation['manual_person_id']
        if in_range and face['status'] != 'false_detection' and (not args.only_unlabeled or unlabeled):
            visible.append(face['face_id'])
    if not visible:
        raise ValueError('No faces match the requested size filter')
    range_label = (f'{min_size:g}px 이상' if max_size is None else
                   f'{max_size:g}px 미만' if min_size == 0 else
                   f'{min_size:g}–{max_size:g}px')
    scope = f'face-size-{min_size:g}-to-{max_size if max_size is not None else "inf"}'
    if args.only_unlabeled:
        scope += '-unlabeled'
    export_name = ('ground_truth_100px_plus.csv' if min_size == 100 and max_size is None
                   else 'ground_truth_face_size_review.csv')
    payload = dict(context=context_hash(cache), source=str(run), faces=faces,
                   visible_face_ids=visible, asset_prefix=asset_prefix,
                   review_scope=scope, export_filename=export_name,
                   title=f'{range_label} 얼굴 정답 입력',
                   review_note=(f'{range_label} 얼굴 중 정답 미입력 얼굴 ' if args.only_unlabeled else f'{range_label} 얼굴 ')
                               + (f'{len(visible)}개만 표시합니다. 기존 정답은 미리 채워져 있으며, '
                                'CSV 다운로드에는 전체 얼굴 행이 포함됩니다.'))
    encoded = json.dumps(payload, ensure_ascii=False).replace('&', '\\u0026').replace('<', '\\u003c').replace('>', '\\u003e')
    template = Path(__file__).with_name('review_ui.html').read_text(encoding='utf-8')
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(template.replace('__REVIEW_PAYLOAD__', encoded), encoding='utf-8')
    diagnosis_path = target.parent / '진단.json'
    if diagnosis_path.exists():
        _render(target.parent, run, json.loads(diagnosis_path.read_text(encoding='utf-8')))
    print(f'Quality ground-truth review: {target} ({len(visible)} faces shown; full CSV export)')
    return target


def normalize_quality_truth(args):
    run = Path(args.run).resolve()
    source = Path(args.input).resolve()
    target = Path(args.output).resolve()
    if target.exists():
        raise ValueError('Normalized truth output already exists')
    cache = json.loads((run / 'cache.json').read_text(encoding='utf-8'))
    rows = _read_csv(source)
    expected = {face['face_id'] for face in cache['faces']}
    if not rows or len(rows) != len(expected) or {row['face_id'] for row in rows} != expected:
        raise ValueError('Ground truth face mapping does not match the run')
    fields = list(rows[0])
    if not {'face_id', 'manual_person_id', 'include', 'annotation_status', '_context_sha256'} <= set(fields):
        raise ValueError('Ground truth columns are incomplete')
    changed = 0
    for row in rows:
        if row['_context_sha256'] != context_hash(cache):
            raise ValueError('Ground truth extraction context mismatch')
        manual = row['manual_person_id'].strip()
        include = row['include'].strip()
        status = row['annotation_status'].strip()
        if manual and include == '1' and status in {'', 'unlabeled'}:
            row['annotation_status'] = 'labeled'
            changed += 1
        elif status in {'uncertain', 'excluded', 'unlabeled'} and manual:
            raise ValueError(f'Contradictory manual ID and status: {row["face_id"]}')
        elif status == 'labeled' and (not manual or include != '1'):
            raise ValueError(f'Invalid labeled row: {row["face_id"]}')
    target.parent.mkdir(parents=True, exist_ok=True)
    write_csv(target, rows, fields)
    read_annotations(target, cache['faces'], cache)
    print(f'Normalized ground truth: {target} ({changed} statuses set to labeled)')
    return target


def _fmt(value):
    return '해당 쌍 없음' if value is None else f'{100 * value:.1f}%'


def _render(output, run, result):
    rows = []
    for item in result['band_results']:
        rows.append('<tr>' + ''.join(f'<td>{html.escape(str(value))}</td>' for value in (
            item['size_band'], item['detected_faces'], item['labeled_faces'], item['unknown_faces'],
            item['identity_faces']['eligible_faces'], item['identity_faces']['faces_with_same_person_mate'],
            _fmt(item['identity_faces']['success_rate']),
            item['same_person']['total_pairs'], item['same_person']['error_pairs'],
            _fmt(item['same_person']['error_rate']), item['different_people']['total_pairs'],
            item['different_people']['error_pairs'], _fmt(item['different_people']['error_rate']))) + '</tr>')
    reviewed = ''
    if result['reviewed_pair_results'] is not None:
        review_rows = []
        for item in result['reviewed_pair_results']:
            review_rows.append('<tr>' + ''.join(f'<td>{html.escape(str(value))}</td>' for value in (
                item['size_band'], item['same']['total_pairs'], item['same']['error_pairs'],
                _fmt(item['same']['error_rate']), item['different']['total_pairs'],
                item['different']['error_pairs'], _fmt(item['different']['error_rate']))) + '</tr>')
        reviewed = ('<h2>별도로 직접 확인한 141개 얼굴 쌍</h2><table><tr><th>작은 얼굴 크기</th>'
                    '<th>같은 사람 쌍</th><th>분리</th><th>분리율</th><th>다른 사람 쌍</th>'
                    '<th>혼합</th><th>혼합률</th></tr>' + ''.join(review_rows) + '</table>')
    review_links = []
    if (output / '정답입력.html').exists():
        review_links.append('<a href="정답입력.html">100px 미만 얼굴 정답 입력</a>')
    if (output / '정답입력_100px이상.html').exists():
        review_links.append('<a href="정답입력_100px이상.html">100px 이상 미입력 얼굴 정답 입력</a>')
    review_link = (' · ' + ' · '.join(review_links)) if review_links else ''
    overall = result.get('overall_metrics')
    overall_html = ''
    if overall:
        overall_html = (f'<h2>전체 입력 정답 기준</h2><p>평가 얼굴 {overall["evaluated_faces"]}개 · '
                        f'다른 사람 혼합 쌍 {overall["false_positive_pairs"]}개 · '
                        f'같은 사람 분리 쌍 {overall["false_negative_pairs"]}개 · '
                        f'Precision {_fmt(overall["precision"])} · Recall {_fmt(overall["recall"])} · '
                        f'F1 {_fmt(overall["f1"])}</p>')
    page = f'''<!doctype html><html lang="ko"><meta charset="utf-8"><title>얼굴 크기별 오류 진단</title>
<style>body{{font:15px sans-serif;line-height:1.65;margin:24px;max-width:1300px}}table{{border-collapse:collapse}}td,th{{border:1px solid #ccc;padding:7px}}th{{background:#eee}}</style>
<h1>얼굴 크기별 그룹화 오류 진단</h1>
<p>기존 자동 그룹을 변경하지 않고 average-linkage 0.40 결과를 분석했습니다. 얼굴 크기는 원본 좌표의 검출 상자에서 가로·세로 중 작은 값입니다. 얼굴 쌍은 둘 중 작은 얼굴을 기준으로 구간을 정했습니다.</p>
{overall_html}
<h2>수동 인물 ID가 입력된 67개 얼굴</h2>
<table><tr><th>얼굴 크기</th><th>전체 검출 얼굴</th><th>정답 얼굴</th><th>미확인 얼굴</th><th>동료가 있는 정답 얼굴</th><th>같은 사람 동료를 찾은 얼굴</th><th>얼굴별 성공률</th><th>같은 사람 쌍</th><th>분리</th><th>분리율</th><th>다른 사람 쌍</th><th>혼합</th><th>혼합률</th></tr>{''.join(rows)}</table>
{reviewed}
<h2>해석 주의</h2><p>얼굴별 성공률은 정답상 같은 사람이 한 명 이상 더 있는 얼굴 중, 같은 자동 그룹에서 그 사람을 한 명이라도 찾은 비율입니다. {html.escape(result['limitations'])} 얼굴 상자 크기는 저해상도를 직접 측정한 값이 아니라 얼굴에 남은 세부 정보의 대용 지표입니다. 작은 구간에 정답이 적으면 오류율이 크게 흔들릴 수 있습니다. 별도 확인 141쌍은 원래 서로 다른 그룹의 후보를 검토한 자료이므로 같은 사람 쌍의 분리율이 100%인 것은 선정 방식에 따른 결과입니다.</p>
<p><a href="진단.json">전체 설정과 수치(JSON)</a> · <a href="{html.escape(os.path.relpath(run / 'index.html', output), quote=True)}">분석한 0.40 결과</a>{review_link}</p></html>'''
    (output / 'index.html').write_text(page, encoding='utf-8')
