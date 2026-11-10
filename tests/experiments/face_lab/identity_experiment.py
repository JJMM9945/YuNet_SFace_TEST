"""Identity linking fidelity and frozen, seed-only recognition experiments."""
import csv
import hashlib
import html
import json
import os
import shutil
import time
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path

import numpy as np

from .evaluation import pair_metrics, read_annotations
from .identities import (apply_identity_changes, empty_store, identity_collections,
                         register_identity, undo_last_change, validate_store)
from .models import sha256
from .pipeline import load_vectors, new_run, write_csv, write_json
from .representative_experiment import quality


SETTINGS = dict(representatives_per_subgroup=3, diversity_weight=.3, min_support=2,
                support_threshold=.40, mean_threshold=.45, margin=.03)


def _read_faces_for_dataset(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as file:
        return list(csv.DictReader(file))


def photo_partition(photo_id):
    bucket = int(hashlib.sha256(photo_id.encode('utf-8')).hexdigest()[:8], 16) % 10
    return 'seed' if bucket < 5 else 'development' if bucket < 7 else 'evaluation'


def _float_face(face):
    result = dict(face)
    for key in ('width', 'height', 'detection_score', 'blur_variance'):
        value = face.get(key)
        result[key] = float(value) if value not in ('', None) else None
    return result


def frozen_identity_representatives(faces, vectors, confirmed_links, settings=SETTINGS):
    """No evaluation annotations are accepted by this prediction function."""
    by_id = {face['face_id']: _float_face(face) for face in faces}
    groups = defaultdict(list)
    for fid, identity in sorted(confirmed_links.items()):
        face = by_id[fid]
        if not face.get('quality_reason') and fid in vectors:
            groups[identity, face['group_id'] or ('unknown:' + fid)].append(face)
    frozen = defaultdict(list)
    for (identity, subgroup), members in sorted(groups.items()):
        remaining = sorted(members, key=lambda face: face['face_id'])
        chosen = []
        used_photos = set()
        while remaining and len(chosen) < settings['representatives_per_subgroup']:
            scored = []
            for face in remaining:
                if face['photo_id'] in used_photos:
                    continue
                fid = face['face_id']
                q = quality(face)
                diversity = min(((1 - float(np.clip(vectors[fid] @ vectors[x['face_id']], -1, 1))) / 2
                                 for x in chosen), default=0.)
                score = q if not chosen else (1-settings['diversity_weight'])*q + settings['diversity_weight']*diversity
                scored.append((score, fid, face, q, diversity))
            if not scored:
                break
            score, fid, face, q, diversity = min(scored, key=lambda row: (-row[0], row[1]))
            chosen.append(face); used_photos.add(face['photo_id']); remaining.remove(face)
            frozen[identity].append(dict(face_id=fid, photo_id=face['photo_id'], subgroup=subgroup,
                                         quality=q, diversity=diversity, selection_score=score))
    return dict(frozen)


def predict_identities(queries, vectors, frozen, settings=SETTINGS):
    decisions = []
    for face in sorted(queries, key=lambda item: item['face_id']):
        fid = face['face_id']
        candidates = []
        if not face.get('quality_reason') and fid in vectors:
            for identity, representatives in sorted(frozen.items()):
                # A photographed face repeated in subgroups cannot create multiple support votes.
                by_photo = {}
                for rep in representatives:
                    if rep['photo_id'] == face['photo_id'] or rep['face_id'] == fid:
                        continue
                    score = float(np.clip(vectors[fid] @ vectors[rep['face_id']], -1, 1))
                    item = dict(face_id=rep['face_id'], photo_id=rep['photo_id'], cosine=score)
                    old = by_photo.get(rep['photo_id'])
                    if old is None or (-score, rep['face_id']) < (-old['cosine'], old['face_id']):
                        by_photo[rep['photo_id']] = item
                selected = sorted(by_photo.values(), key=lambda item: (-item['cosine'], item['face_id']))[:2]
                if selected:
                    candidates.append(dict(identity_id=identity,
                                           score=float(np.mean([item['cosine'] for item in selected])),
                                           support=sum(item['cosine'] >= settings['support_threshold'] for item in selected),
                                           references=selected))
        candidates.sort(key=lambda item: (-item['score'], item['identity_id']))
        first = candidates[0] if candidates else None
        gap = first['score'] - candidates[1]['score'] if len(candidates) > 1 else None
        if face.get('quality_reason') or fid not in vectors:
            reason = 'quality_excluded_or_missing_embedding'
        elif first is None:
            reason = 'no_candidate'
        elif first['support'] < settings['min_support']:
            reason = 'insufficient_distinct_photo_support'
        elif first['score'] < settings['mean_threshold']:
            reason = 'mean_below_threshold'
        elif gap is not None and gap < settings['margin']:
            reason = 'competing_identity_margin_insufficient'
        else:
            reason = 'assigned'
        decisions.append(dict(face_id=fid, predicted_identity=first['identity_id'] if reason == 'assigned' else None,
                              reason=reason, score_gap=gap, candidates=candidates))
    return decisions


def _probe_metrics(probes, decisions, annotations, seed_store):
    alias_to_identity = {value['display_name']: identity for identity, value in seed_store['identities'].items()}
    predicted = {item['face_id']: item for item in decisions}
    correct, wrong, unregistered_assigned = [], [], []
    pair_faces = []
    known_faces = 0
    for face in probes:
        fid = face['face_id']; prediction = predicted[fid]['predicted_identity']
        actual_alias = annotations[fid]['manual_person_id']
        expected_identity = alias_to_identity.get(actual_alias)
        known_faces += expected_identity is not None
        if prediction:
            (correct if prediction == expected_identity else wrong).append(fid)
            if expected_identity is None:
                unregistered_assigned.append(fid)
        pair_faces.append(dict(face_id=fid, group_id=prediction or ''))
    return dict(probe_faces=len(probes), registered_identity_faces=known_faces,
                unregistered_identity_faces=len(probes)-known_faces,
                assigned=len(correct)+len(wrong), correct_assignments=len(correct), wrong_assignments=len(wrong),
                correct_face_ids=correct, wrong_face_ids=wrong,
                unregistered_wrongly_assigned=unregistered_assigned,
                held=len(probes)-len(correct)-len(wrong),
                assignment_precision=len(correct)/(len(correct)+len(wrong)) if correct or wrong else None,
                correct_registered_face_rate=len(correct)/known_faces if known_faces else None,
                pair_metrics=pair_metrics(pair_faces, annotations) if probes else None)


def _baseline_identity_decisions(probes, seed_faces, seed_links):
    votes = defaultdict(set)
    for face in seed_faces:
        if face['group_id']:
            votes[face['group_id']].add(seed_links[face['face_id']])
    return [dict(face_id=face['face_id'], predicted_identity=(next(iter(votes[face['group_id']]))
                 if face['group_id'] and len(votes[face['group_id']]) == 1 and not face.get('quality_reason') else None))
            for face in probes]


def test_identity_links(args):
    started = time.perf_counter()
    source, truth_path = Path(args.run).resolve(), Path(args.ground_truth).resolve()
    cache = json.loads((source/'cache.json').read_text(encoding='utf-8'))
    with (source/'faces.csv').open(encoding='utf-8-sig', newline='') as file:
        faces = list(csv.DictReader(file))
    if len(faces) != len(cache['faces']) or {f['face_id'] for f in faces} != {f['face_id'] for f in cache['faces']}:
        raise ValueError('Source face mapping differs from cache')
    before_sha = sha256(source/'faces.csv')
    vectors = load_vectors(source, cache)
    annotations = read_annotations(truth_path, cache['faces'], cache)
    usable = [face for face in faces if annotations[face['face_id']]['include'] == '1']
    if any(not annotations[face['face_id']]['manual_person_id'] or
           annotations[face['face_id']]['annotation_status'] != 'labeled' for face in usable):
        raise ValueError('This functional test requires explicit truth for every included face')
    excluded = [face['face_id'] for face in faces if annotations[face['face_id']]['include'] == '0']

    store = empty_store(cache, before_sha, excluded)
    mapping = {alias: register_identity(store, alias) for alias in sorted({annotations[f['face_id']]['manual_person_id'] for f in usable})}
    changes = {face['face_id']: mapping[annotations[face['face_id']]['manual_person_id']] for face in usable}
    store = apply_identity_changes(store, cache, changes, 'Import explicit user-confirmed face identities')
    collections = identity_collections(faces, store)
    expected_photos = defaultdict(set)
    for face in usable:
        expected_photos[changes[face['face_id']]].add(face['photo_id'])
    if any(set(row['photo_ids']) != expected_photos[identity] for identity, row in collections.items()):
        raise ValueError('Identity search photo set differs from explicit confirmations')
    restored = json.loads(json.dumps(store))
    validate_store(restored, cache)
    assert identity_collections(faces, restored) == collections
    renumbered = deepcopy(faces)
    for index, face in enumerate(renumbered):
        face['group_id'] = ('renumbered_' + str(index % 7)) if face['group_id'] else ''
    assert {identity: row['face_ids'] for identity, row in identity_collections(renumbered, restored).items()} == {
        identity: row['face_ids'] for identity, row in collections.items()}
    first = usable[0]['face_id']
    detached = apply_identity_changes(store, cache, {first: None}, 'Functional test: detach one face')
    undone = undo_last_change(detached, cache)
    assert undone['links'] == store['links'] and store['links'].get(first) is not None
    checks = dict(confirmed_faces=len(changes), identities=len(collections), excluded_faces=len(excluded),
                  confirmed_unknown_faces=sum(not f['group_id'] for f in usable),
                  identities_spanning_multiple_groups=sum(len(row['automatic_groups']) > 1 for row in collections.values()),
                  automatic_groups_preserved=True, restart_roundtrip=True,
                  group_renumbering_preserves_face_identity_links=True, detach_and_undo=True,
                  search_photo_sets_match_confirmations=True,
                  notice='These are human-confirmation fidelity checks, not automatic accuracy improvements.')

    partitions = {photo['photo_id']: photo_partition(photo['photo_id']) for photo in cache['photos']}
    seed_faces = [face for face in usable if partitions[face['photo_id']] == 'seed']
    seed_store = empty_store(cache, before_sha, excluded)
    seed_aliases = {alias: register_identity(seed_store, alias)
                    for alias in sorted({annotations[f['face_id']]['manual_person_id'] for f in seed_faces})}
    seed_links = {face['face_id']: seed_aliases[annotations[face['face_id']]['manual_person_id']] for face in seed_faces}
    seed_store = apply_identity_changes(seed_store, cache, seed_links, 'Explicit confirmed seed photos only')
    # Predict with seed links only. Full annotations are not accepted by either function below.
    frozen = frozen_identity_representatives(seed_faces, vectors, seed_links)
    results, decisions_by_split = {}, {}
    for name in ('development', 'evaluation'):
        probes = [face for face in usable if partitions[face['photo_id']] == name]
        predicted = predict_identities(probes, vectors, frozen)
        baseline = _baseline_identity_decisions(probes, seed_faces, seed_links)
        # Evaluate only after predictions for this split have been computed.
        results[name] = dict(baseline_group_to_confirmed_identity=_probe_metrics(probes, baseline, annotations, seed_store),
                             frozen_multigroup_identity=_probe_metrics(probes, predicted, annotations, seed_store))
        decisions_by_split[name] = predicted
    assert sha256(source/'faces.csv') == before_sha

    output = new_run(args.output, Path(cache['input_root']))
    (output/'crops').mkdir(exist_ok=True)
    for face in faces:
        if face.get('crop_path'):
            shutil.copy2(source/'crops'/f"{face['face_id']}.jpg", output/'crops'/f"{face['face_id']}.jpg")
    write_json(output/'identity_store.json', store)
    write_json(output/'seed_identity_store.json', seed_store)
    write_json(output/'representatives.json', frozen)
    write_json(output/'photo_split.json', dict(method='SHA256(photo_id) first8 hex modulo10: seed0-4, development5-6, evaluation7-9',
                                               partitions=partitions))
    write_json(output/'functional_checks.json', checks)
    write_json(output/'identity_collections.json', collections)
    write_json(output/'probe_decisions.json', decisions_by_split)
    write_csv(output/'face_identity_links.csv', [dict(face_id=f['face_id'], photo_id=f['photo_id'],
                                                    automatic_group=f['group_id'], identity_id=store['links'].get(f['face_id'], ''),
                                                    explicit_confirmed_alias=annotations[f['face_id']]['manual_person_id'],
                                                    included=annotations[f['face_id']]['include'],
                                                    partition=partitions[f['photo_id']]) for f in faces],
              ['face_id','photo_id','automatic_group','identity_id','explicit_confirmed_alias','included','partition'])
    report = dict(source=str(source), source_faces_sha256=before_sha, features_sha256=cache['features_sha256'],
                  context_sha256=store['extraction']['context_sha256'], truth_sha256=sha256(truth_path),
                  functional_checks=checks, settings=SETTINGS,
                  seed_faces=len(seed_faces), seed_identities=len(seed_store['identities']),
                  photo_counts=dict(Counter(partitions.values())), results=results,
                  baseline=pair_metrics(faces, annotations), seconds=time.perf_counter()-started,
                  score_definition='mean of best two distinct seed-photo cosine scores across all frozen subgroups of one identity; each >=.40, mean>=.45, competition gap>=.03',
                  limitations='Same previously inspected photo collection; exploratory photo split, not independent validation. Automatic source groups were made from all photos without identity labels. Same people/events may cross partitions. Fixed settings were reused without tuning on evaluation. Human false exclusions are supplied.')
    write_json(output/'summary.json', report)
    _render(output, source, faces, collections, seed_store, decisions_by_split, report)
    print(f'Identity link test: {output / "index.html"}')
    return output


def edit_identity_links(args):
    cache = json.loads(Path(args.cache).read_text(encoding='utf-8'))
    store = json.loads(Path(args.store).read_text(encoding='utf-8'))
    validate_store(store, cache)
    if bool(args.undo) == bool(args.changes):
        raise ValueError('Choose exactly one of --changes and --undo')
    if args.undo:
        result = undo_last_change(store, cache)
    else:
        change = json.loads(Path(args.changes).read_text(encoding='utf-8'))
        if change.get('extraction') != store['extraction']:
            raise ValueError('Change file extraction mismatch')
        if change.get('expected_revision') != store['revision']:
            raise ValueError('Change file is based on a stale store revision')
        result = apply_identity_changes(store, cache, change['changes'], change['reason'])
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('x', encoding='utf-8') as file:
        json.dump(result, file, ensure_ascii=False, indent=2)
    print(f'Identity store revision {result["revision"]}: {target}')
    return target


def _render(output, source, faces, collections, seed_store, decisions_by_split, report):
    by_id = {face['face_id']: face for face in faces}
    def image(fid):
        return f'<img loading="lazy" src="crops/{html.escape(fid, quote=True)}.jpg" alt="{html.escape(fid, quote=True)}">'
    sections = []
    for identity, row in sorted(collections.items(), key=lambda item: item[1]['display_name']):
        cards = []
        for fid in row['face_ids']:
            face = by_id[fid]
            photo_link = os.path.relpath(source/'annotated'/f"{face['photo_id']}.jpg", output)
            cards.append(f'<div class="face">{image(fid)}<small>{html.escape(fid)}<br>自動 {html.escape(face["group_id"] or "미확인")}</small><a href="{html.escape(photo_link, quote=True)}">원본 위치</a></div>'.replace('自動', '자동'))
        sections.append(f'<section id="{identity}"><h3>{html.escape(row["display_name"])}</h3><small>{identity}</small>'
                        f'<p>얼굴 {len(row["face_ids"])}개 / 중복 제거 사진 {len(row["photo_ids"])}장 / 자동 그룹 {html.escape(", ".join(row["automatic_groups"]))} / 자동 미확인 {len(row["unknown_faces"])}개</p><div class="gallery">{"".join(cards)}</div></section>')
    rows = []
    pair_rows = []
    method_labels = dict(baseline_group_to_confirmed_identity='기존 자동 그룹 연결',
                         frozen_multigroup_identity='인물별 여러 그룹 대표')
    for split, result in report['results'].items():
        for method, metric in result.items():
            rows.append('<tr>' + ''.join('<td>'+html.escape(str(value))+'</td>' for value in [split, method_labels[method],
                metric['probe_faces'], metric['registered_identity_faces'], metric['assigned'],
                metric['correct_assignments'], metric['wrong_assignments'], metric['held'],
                metric['unregistered_identity_faces'], len(metric['unregistered_wrongly_assigned'])]) + '</tr>')
            pair_metric = metric['pair_metrics']
            if pair_metric:
                values = [split, method_labels[method], pair_metric['false_positive_pairs'],
                          pair_metric['false_negative_pairs']]
                values += ['미정의' if pair_metric[key] is None else f'{pair_metric[key]*100:.1f}%'
                           for key in ('precision', 'recall', 'f1')]
                pair_rows.append('<tr>' + ''.join('<td>'+html.escape(str(value))+'</td>' for value in values) + '</tr>')
    details = []
    reasons = {'assigned':'自動 배정', 'insufficient_distinct_photo_support':'다른 사진 지지2개 미달',
               'mean_below_threshold':'평균0.45 미달', 'competing_identity_margin_insufficient':'경쟁 차이0.03 미달',
               'no_candidate':'후보없음', 'quality_excluded_or_missing_embedding':'품질제외/특징없음'}
    for split, decisions in decisions_by_split.items():
        cards = []
        for decision in decisions:
            fid=decision['face_id']; top=decision['candidates'][:2]
            candidates = ', '.join(seed_store['identities'][c['identity_id']]['display_name']+f" 평균{c['score']:.3f}/지지{c['support']}" for c in top) or '없음'
            cards.append(f'<div class="probe">{image(fid)}<small>{html.escape(fid)}</small><p>{html.escape(reasons.get(decision["reason"],decision["reason"]))}</p><small>{html.escape(candidates)}</small></div>'.replace('自動', '자동'))
        details.append(f'<details><summary>{split} 얼굴별 예측과 후보</summary><div class="gallery">{"".join(cards)}</div></details>')
    checks=report['functional_checks']
    page=f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>인물 ID 연결 테스트</title>
<style>body{{font:15px sans-serif;line-height:1.7;margin:24px}}table{{border-collapse:collapse}}td,th{{border:1px solid #ccc;padding:7px}}th{{background:#eef3f6}}.gallery{{display:flex;flex-wrap:wrap;gap:12px}}.face{{width:150px}}.probe{{width:230px;border:1px solid #ccc;padding:10px}}img{{width:112px;height:112px;object-fit:contain}}small{{display:block;overflow-wrap:anywhere}}section{{border-top:1px solid #bbb;padding:12px 0}}details{{margin:20px 0}}</style>
<h1>확인된 인물 ID 아래에 여러 얼굴 그룹 연결</h1><p>최초 average-linkage 0.40과 원본 자동 그룹은 유지합니다. 아래 인물 모음은 수동으로 확인한 얼굴 연결을 적용한 것이며 자동 정확도 향상이 아닙니다.</p>
<h2>1. 수동 확인 반영 기능 검증</h2><p>확인 얼굴 {checks['confirmed_faces']}개 → 인물 {checks['identities']}개. 여러 자동 그룹을 연결한 인물 {checks['identities_spanning_multiple_groups']}개. 자동 미확인에서 연결한 얼굴 {checks['confirmed_unknown_faces']}개. 제외 {checks['excluded_faces']}개.</p>
<p>저장·다시 읽기, 그룹 번호 변경 후 얼굴 연결 유지, 단일 얼굴 해제·되돌리기, 검색 사진 중복 제거, 원본 그룹 보존을 검사했습니다. 모두 통과했습니다. 전체 정답을 입력한 결과이므로 자동 인식 지표100%로 보고하지 않습니다.</p>
<h2>2. 확인 사진 일부만 사용한 추가 자동 배정</h2><p>사진 해시로 seed/development/evaluation을 나눴습니다. 사진수 {html.escape(str(report['photo_counts']))}. seed에서만 {report['seed_faces']}얼굴/{report['seed_identities']}인물의 확인정보를 사용했습니다. 대표는 인물의 각 자동 하위그룹에서 최대3개씩 품질·다양성으로 선택하고 같은사진 중복을 억제했습니다.</p>
<p>인물 점수는 서로 다른 seed사진에서 가장 가까운 대표2개의 코사인 유사도 평균입니다. 대표2개 모두0.40 이상, 평균0.45 이상, 경쟁인물 차이0.03을 요구합니다. 최고점 하나만으로 배정하지 않으며 새로 배정된 얼굴은 대표로 추가하지 않습니다. 후보1개일 때 경쟁점수 차이는 null로 두되 지지·평균조건은 적용합니다.</p>
<p>baseline_group_to_confirmed_identity: 같은 자동그룹에 seed 인물ID가 하나만 있으면 연결한 비교 기준. frozen_multigroup_identity: 확인된 인물의 여러 하위그룹 대표를 이용한 예측. 평가할 얼굴의 정답은 두 예측이 끝난 뒤 평가에만 사용합니다.</p>
<table><tr><th>세트</th><th>방법</th><th>얼굴</th><th>seed등록 인물 얼굴</th><th>배정</th><th>정배정</th><th>오배정</th><th>보류</th><th>미등록인물 얼굴</th><th>미등록 오배정</th></tr>{''.join(rows)}</table>
<h3>정답을 숨긴 얼굴끼리의 쌍 평가</h3><table><tr><th>세트</th><th>방법</th><th>혼합 쌍</th><th>분리 쌍</th><th>Precision</th><th>Recall</th><th>F1</th></tr>{''.join(pair_rows)}</table>
<p>얼굴 쌍 평가는 보류된 얼굴을 각각 다른 단독 예측으로 계산합니다. 다른 인물ID로 한 얼굴만 잘못 연결되면 같은세트의 쌍 지표에는 혼합으로 잡히지 않을 수 있어 위의 인물ID 오배정 수도 함께 봐야 합니다. 후보가 없는 얼굴·미등록 인물도 제외하지 않고 표에 포함했습니다.</p>
<p>고정 설정을 사용했고 evaluation 정답으로 설정을 조정하지 않았습니다. 이미 여러 번 본54장 자료이며 같은인물·연속사진이 양쪽에 존재할 수 있어 독립평가가 아닙니다. 기존 자동그룹은 전체사진에서 비지도 생성한 것이므로 원본그룹 기준 비교는 전이적(transductive) 비교입니다.</p>
{''.join(details)}<h2>3. 확인된 인물별 사진 모음</h2>{''.join(sections)}
<h2>저장 파일</h2><p><a href="identity_store.json">영구 인물 연결</a> · <a href="face_identity_links.csv">얼굴별 연결</a> · <a href="summary.json">전체 평가·설정</a> · <a href="functional_checks.json">기능 검증</a> · <a href="photo_split.json">사진 분할</a> · <a href="probe_decisions.json">모든 후보와 점수</a></p></html>'''
    (output/'index.html').write_text(page,encoding='utf-8')
