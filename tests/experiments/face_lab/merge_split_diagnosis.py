"""Development/evaluation diagnosis for conservative group merge and split rules."""
import csv
import hashlib
import html
import itertools
import json
import os
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path

import numpy as np

from .evaluation import pair_metrics, read_annotations
from .grouping import group_faces
from .models import sha256
from .pipeline import load_vectors, write_json


RULE_THRESHOLDS = (0.40, 0.38, 0.37, 0.35)


def _read_faces(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as file:
        return list(csv.DictReader(file))


def _same_photo(first, second):
    return bool({face['photo_id'] for face in first} & {face['photo_id'] for face in second})


def group_pair_diagnostics(faces, vectors, annotations):
    groups = defaultdict(list)
    for face in faces:
        if face.get('group_id'):
            groups[face['group_id']].append(face)
    identities = {gid: {annotations[face['face_id']]['manual_person_id'] for face in members}
                  for gid, members in groups.items()}
    pure = {gid: next(iter(ids)) for gid, ids in identities.items() if len(ids) == 1}
    scores = {}
    rows = []
    for first, second in itertools.combinations(sorted(pure), 2):
        values = np.array([[float(np.clip(vectors[a['face_id']] @ vectors[b['face_id']], -1, 1))
                            for b in groups[second]] for a in groups[first]])
        score = float(values.mean())
        scores[first, second] = score
        rows.append(dict(group_a=first, group_b=second, identity_a=pure[first], identity_b=pure[second],
                         same_identity=pure[first] == pure[second], score=score,
                         minimum=float(values.min()), maximum=float(values.max()),
                         group_a_size=len(groups[first]), group_b_size=len(groups[second]),
                         same_photo=_same_photo(groups[first], groups[second])))
    def score(a, b):
        return scores[tuple(sorted((a, b)))]
    best = {gid: max((other for other in pure if other != gid),
                     key=lambda other: (score(gid, other), other))
            for gid in pure} if len(pure) > 1 else {}
    mutual = []
    row_map = {tuple(sorted((row['group_a'], row['group_b']))): row for row in rows}
    for gid, other in sorted(best.items()):
        if gid < other and best.get(other) == gid:
            mutual.append(row_map[gid, other])
    mixed = []
    for gid, ids in sorted(identities.items()):
        if len(ids) > 1:
            members = groups[gid]
            values = [float(np.clip(vectors[a['face_id']] @ vectors[b['face_id']], -1, 1))
                      for a, b in itertools.combinations(members, 2)]
            mixed.append(dict(group_id=gid, actual_person_ids=sorted(ids), face_ids=[f['face_id'] for f in members],
                              member_count=len(members), same_photo_collision=any(
                                  a['photo_id'] == b['photo_id'] for a, b in itertools.combinations(members, 2)),
                              minimum_internal_similarity=min(values) if values else None))
    return dict(groups=groups, identities=identities, pure=pure, all_pairs=rows,
                mutual_pairs=mutual, mixed_groups=mixed)


def accepted_merges(mutual_pairs, threshold, same_photo_veto=True):
    return [row for row in mutual_pairs
            if row['score'] >= threshold and not (same_photo_veto and row['same_photo'])]


def merge_rule_result(mutual_pairs, threshold, same_photo_veto=True):
    accepted = accepted_merges(mutual_pairs, threshold, same_photo_veto)
    correct = sum(row['same_identity'] for row in accepted)
    return dict(threshold=threshold, same_photo_veto=same_photo_veto,
                accepted=len(accepted), correct_merges=correct,
                wrong_merges=len(accepted) - correct, pairs=accepted)


def choose_development_threshold(mutual_pairs):
    candidates = []
    for step in range(60, 91):
        threshold = step / 200  # 0.300 ... 0.450 in 0.005 increments
        result = merge_rule_result(mutual_pairs, threshold, True)
        candidates.append(result)
    safe = [result for result in candidates if result['wrong_merges'] == 0]
    if not safe:
        return max(candidates, key=lambda result: (-result['wrong_merges'], result['correct_merges'], result['threshold']))
    return max(safe, key=lambda result: (result['correct_merges'], result['threshold']))


def _apply_merges(faces, accepted):
    remap = {}
    for row in accepted:
        first, second = sorted((row['group_a'], row['group_b']))
        remap[second] = first
    for face in faces:
        if face.get('group_id') in remap:
            face['group_id'] = remap[face['group_id']]
    return faces


def _split_flags(diagnostics, vectors, thresholds=(0.35, 0.40, 0.45, 0.50)):
    groups, identities = diagnostics['groups'], diagnostics['identities']
    same_photo = []
    for gid, members in sorted(groups.items()):
        if any(a['photo_id'] == b['photo_id'] for a, b in itertools.combinations(members, 2)):
            same_photo.append(dict(group_id=gid, mixed=len(identities[gid]) > 1,
                                   actual_person_ids=sorted(identities[gid])))
    weak = []
    for threshold in thresholds:
        flags = []
        for gid, members in sorted(groups.items()):
            if len(members) < 2:
                continue
            minimum = min(float(np.clip(vectors[a['face_id']] @ vectors[b['face_id']], -1, 1))
                          for a, b in itertools.combinations(members, 2))
            if minimum < threshold:
                flags.append(dict(group_id=gid, mixed=len(identities[gid]) > 1,
                                  minimum_internal_similarity=minimum,
                                  actual_person_ids=sorted(identities[gid])))
        weak.append(dict(threshold=threshold, flagged_groups=len(flags),
                         mixed_groups_caught=sum(flag['mixed'] for flag in flags),
                         pure_groups_flagged=sum(not flag['mixed'] for flag in flags), flags=flags))
    return dict(same_photo=dict(flagged_groups=len(same_photo),
                                mixed_groups_caught=sum(flag['mixed'] for flag in same_photo),
                                pure_groups_flagged=sum(not flag['mixed'] for flag in same_photo), flags=same_photo),
                weak_internal_similarity=weak)


def _photo_partition(photo_id):
    value = int(hashlib.sha256(photo_id.encode('utf-8')).hexdigest()[:8], 16)
    return 'development' if value % 10 < 7 else 'evaluation'


def _subset_experiment(name, source_faces, false_ids, vectors, annotations):
    photos = {face['photo_id'] for face in source_faces if _photo_partition(face['photo_id']) == name}
    faces = [deepcopy(face) for face in source_faces
             if face['photo_id'] in photos and face['face_id'] not in false_ids]
    group_faces(faces, vectors, threshold=.40, min_group=1, margin=.01, method='average')
    baseline = pair_metrics(faces, annotations)
    diagnostics = group_pair_diagnostics(faces, vectors, annotations)
    return dict(name=name, photo_ids=sorted(photos), faces=faces, baseline=baseline,
                diagnostics=diagnostics)


def diagnose_merge_split(args):
    run, truth_path, output = Path(args.run).resolve(), Path(args.ground_truth).resolve(), Path(args.output).resolve()
    if output.exists():
        raise ValueError('Output directory already exists; preserve the previous comparison')
    cache = json.loads((run / 'cache.json').read_text(encoding='utf-8'))
    faces = _read_faces(run / 'faces.csv')
    if len(faces) != len(cache['faces']) or {f['face_id'] for f in faces} != {f['face_id'] for f in cache['faces']}:
        raise ValueError('Run face mapping does not match its cache')
    if sha256(run / 'features.npz') != cache['features_sha256']:
        raise ValueError('Feature cache checksum mismatch')
    annotations = read_annotations(truth_path, cache['faces'], cache)
    vectors = load_vectors(run, cache)
    false_ids = {face['face_id'] for face in faces if face['status'] == 'false_detection'}
    if any(annotations[fid]['include'] != '0' for fid in false_ids):
        raise ValueError('False detections must be excluded in ground truth')
    full = group_pair_diagnostics([face for face in faces if face['face_id'] not in false_ids], vectors, annotations)
    full_rules = [merge_rule_result(full['mutual_pairs'], threshold, veto)
                  for threshold in RULE_THRESHOLDS for veto in (False, True)]
    split_flags = _split_flags(full, vectors)

    source_faces = deepcopy(cache['faces'])
    development = _subset_experiment('development', source_faces, false_ids, vectors, annotations)
    evaluation = _subset_experiment('evaluation', source_faces, false_ids, vectors, annotations)
    selected = choose_development_threshold(development['diagnostics']['mutual_pairs'])
    evaluation_rule = merge_rule_result(evaluation['diagnostics']['mutual_pairs'], selected['threshold'], True)
    development_after = pair_metrics(_apply_merges(deepcopy(development['faces']), selected['pairs']), annotations)
    evaluation_after = pair_metrics(_apply_merges(deepcopy(evaluation['faces']), evaluation_rule['pairs']), annotations)
    partition = dict(
        method='SHA-256(photo_id) modulo 10: 0-6 development, 7-9 evaluation; no identity labels used for partition',
        development=dict(photos=len(development['photo_ids']), faces=len(development['faces']),
                         baseline=development['baseline'], selected_rule=selected, after_merge=development_after),
        evaluation=dict(photos=len(evaluation['photo_ids']), faces=len(evaluation['faces']),
                        baseline=evaluation['baseline'], applied_rule=evaluation_rule, after_merge=evaluation_after))
    report = dict(run=str(run), run_faces_sha256=sha256(run / 'faces.csv'),
                  features_sha256=cache['features_sha256'], truth_sha256=sha256(truth_path),
                  baseline_settings=dict(linkage='average', similarity_threshold=.40, ambiguity_margin=.01,
                                         min_group_size=1, false_detections_excluded=len(false_ids)),
                  score_definition='arithmetic mean of all cross-group cosine similarities; reciprocal best groups only',
                  full_data=dict(groups=len(full['groups']), pure_groups=len(full['pure']),
                                 mixed_groups=full['mixed_groups'], mutual_pairs=full['mutual_pairs'],
                                 rule_comparisons=full_rules, split_flags=split_flags),
                  partition_experiment=partition,
                  limitations=('Ground truth comes from the same 54-photo collection. The photo split prevents image overlap, '
                               'but people may occur in both partitions. This is an exploratory comparison, not an independent accuracy claim.'))
    output.mkdir(parents=True)
    write_json(output / '진단.json', report)
    _render(output, run, report)
    print(f'Merge/split diagnosis: {output / "index.html"}')
    return report


def _pct(value):
    return '미정의' if value is None else f'{100 * value:.1f}%'


def _metric_row(label, metric):
    return (f'<tr><td>{html.escape(label)}</td><td>{metric["evaluated_faces"]}</td>'
            f'<td>{metric["false_positive_pairs"]}</td><td>{metric["false_negative_pairs"]}</td>'
            f'<td>{_pct(metric["precision"])}</td><td>{_pct(metric["recall"])}</td><td>{_pct(metric["f1"])}</td></tr>')


def _render(output, run, report):
    full = report['full_data']
    rule_rows = []
    for rule in full['rule_comparisons']:
        pairs = ', '.join(f"{p['group_a']}+{p['group_b']} ({p['score']:.3f}, {'같음' if p['same_identity'] else '다름'})"
                          for p in rule['pairs']) or '없음'
        rule_rows.append(f'<tr><td>{rule["threshold"]:.2f}</td><td>{"적용" if rule["same_photo_veto"] else "미적용"}</td>'
                         f'<td>{rule["accepted"]}</td><td>{rule["correct_merges"]}</td><td>{rule["wrong_merges"]}</td>'
                         f'<td>{html.escape(pairs)}</td></tr>')
    mixed_rows = []
    for group in full['mixed_groups']:
        mixed_rows.append(f'<tr><td>{group["group_id"]}</td><td>{html.escape(", ".join(group["actual_person_ids"]))}</td>'
                          f'<td>{group["member_count"]}</td><td>{"예" if group["same_photo_collision"] else "아니요"}</td>'
                          f'<td>{group["minimum_internal_similarity"]:.3f}</td></tr>')
    split = full['split_flags']
    split_rows = [f'<tr><td>같은 사진의 여러 얼굴</td><td>{split["same_photo"]["flagged_groups"]}</td>'
                  f'<td>{split["same_photo"]["mixed_groups_caught"]}</td><td>{split["same_photo"]["pure_groups_flagged"]}</td></tr>']
    for item in split['weak_internal_similarity']:
        split_rows.append(f'<tr><td>내부 최소 유사도 &lt; {item["threshold"]:.2f}</td><td>{item["flagged_groups"]}</td>'
                          f'<td>{item["mixed_groups_caught"]}</td><td>{item["pure_groups_flagged"]}</td></tr>')
    part = report['partition_experiment']; dev, ev = part['development'], part['evaluation']
    selected = dev['selected_rule']; applied = ev['applied_rule']
    selected_pairs = ', '.join(f"{p['group_a']}+{p['group_b']}" for p in selected['pairs']) or '없음'
    applied_pairs = ', '.join(f"{p['group_a']}+{p['group_b']}" for p in applied['pairs']) or '없음'
    page = f'''<!doctype html><html lang="ko"><meta charset="utf-8"><title>그룹 병합·분리 규칙 진단</title>
<style>body{{font:15px sans-serif;line-height:1.65;margin:24px;max-width:1500px}}table{{border-collapse:collapse}}td,th{{border:1px solid #ccc;padding:7px}}th{{background:#eee}}</style>
<h1>그룹 병합·분리 규칙 진단</h1>
<p>average-linkage 0.40 자동 결과를 변경하지 않고 후보 규칙을 비교했습니다. 그룹 병합 점수는 두 그룹의 모든 얼굴 간 코사인 유사도 평균이며, 서로의 1순위인 그룹만 후보입니다.</p>
<h2>전체 정답에서 본 병합 규칙</h2><table><tr><th>평균 기준</th><th>같은 사진 병합 방지</th><th>병합 후보</th><th>올바른 병합</th><th>잘못된 병합</th><th>상세</th></tr>{''.join(rule_rows)}</table>
<p>이 표는 규칙의 구분 가능성을 설명하기 위한 전체 자료 진단입니다. 설정 선택과 독립 평가를 동시에 의미하지 않습니다.</p>
<h2>현재 혼합 그룹</h2><table><tr><th>자동 그룹</th><th>실제 인물 ID</th><th>얼굴</th><th>같은 사진 충돌</th><th>내부 최소 유사도</th></tr>{''.join(mixed_rows)}</table>
<h2>분리 검토 표시 규칙</h2><table><tr><th>표시 조건</th><th>표시 그룹</th><th>혼합 그룹 발견</th><th>정상 그룹도 표시</th></tr>{''.join(split_rows)}</table>
<p>분리 규칙은 자동 분리를 실행하지 않고 검토 대상으로 표시한 결과입니다. 같은 사진 규칙에도 거울·사진 속 사진 같은 예외가 있으므로 사용자 확인이 필요합니다.</p>
<h2>사진이 겹치지 않는 개발·확인 세트</h2><p>{html.escape(part['method'])}</p>
<p>개발 {dev['photos']}장/{dev['faces']}얼굴에서 잘못된 병합 0개를 유지하면서 가장 많은 올바른 병합을 만드는 기준은 <strong>{selected['threshold']:.3f}</strong>였습니다. 개발 병합: {html.escape(selected_pairs)}. 이 값을 확인 {ev['photos']}장/{ev['faces']}얼굴에 그대로 적용한 병합: {html.escape(applied_pairs)}.</p>
<table><tr><th>결과</th><th>평가 얼굴</th><th>혼합 쌍</th><th>분리 쌍</th><th>Precision</th><th>Recall</th><th>F1</th></tr>
{_metric_row('개발 기준선', dev['baseline'])}{_metric_row('개발 병합 후', dev['after_merge'])}{_metric_row('확인 기준선', ev['baseline'])}{_metric_row('확인 병합 후', ev['after_merge'])}</table>
<h2>해석</h2><p>{html.escape(report['limitations'])} 전체 54장 결과에 같은 기준을 바로 적용하면 전체 자료 표에 나온 잘못된 병합이 생길 수 있으므로 자동 기본값으로 채택하지 않습니다.</p>
<p><a href="진단.json">모든 후보와 수치(JSON)</a> · <a href="{html.escape(os.path.relpath(run / 'index.html', output), quote=True)}">기준 자동 결과</a></p></html>'''
    (output / 'index.html').write_text(page, encoding='utf-8')
