"""Apply confirmed identity references to a new local photo extraction.

Exact duplicate photos can transfer explicit confirmations; model predictions are
kept separately and never written as confirmed links.
"""
import html
import json
import os
import shutil
import time
from collections import Counter
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .identities import (apply_identity_changes, empty_store, identity_collections,
                         validate_store)
from .identity_experiment import (SETTINGS, frozen_identity_representatives,
                                  predict_identities, _float_face)
from .models import load, sha256
from .pipeline import analyze, extraction_signature, inventory, load_vectors, write_csv, write_json
from .preprocess import decode_oriented, extract
from .review import context_hash


def exact_photo_face_matches(reference_faces, target_faces, reference_cache, target_cache,
                             reference_vectors, target_vectors):
    """Return only unique exact-content/geometry/vector correspondences, never filenames."""
    reference_hashes = {photo['photo_id']: photo['sha256'] for photo in reference_cache['photos']}
    target_hashes = {photo['photo_id']: photo['sha256'] for photo in target_cache['photos']}
    result = {}
    for target in target_faces:
        candidates = []
        for reference in reference_faces:
            if reference_hashes[reference['photo_id']] != target_hashes[target['photo_id']]:
                continue
            first = [float(reference[key]) for key in ('x','y','width','height')]
            second = [float(target[key]) for key in ('x','y','width','height')]
            if not np.allclose(first, second, atol=1e-4, rtol=0):
                continue
            la, lb = reference['landmarks'], target['landmarks']
            la = json.loads(la) if isinstance(la, str) else la
            lb = json.loads(lb) if isinstance(lb, str) else lb
            if not np.allclose(la, lb, atol=1e-4, rtol=0):
                continue
            a, b = reference['face_id'], target['face_id']
            if a not in reference_vectors or b not in target_vectors:
                continue
            if not np.allclose(reference_vectors[a], target_vectors[b], atol=1e-5, rtol=0):
                continue
            candidates.append(a)
        if len(candidates) == 1:
            result[target['face_id']] = candidates[0]
    return result


def content_photo_faces(faces, cache):
    hashes = {photo['photo_id']: photo['sha256'] for photo in cache['photos']}
    result = deepcopy(faces)
    for face in result:
        face['photo_id'] = 'sha256:' + hashes[face['photo_id']]
    return result


def verify_reference_runtime(reference_cache, current_signature, vectors, models):
    old_signature = reference_cache['signature']
    if old_signature == current_signature:
        return dict(status='exact_runtime_signature', faces_checked=0)
    old, new = deepcopy(old_signature), deepcopy(current_signature)
    old_platform = old['runtime'].pop('platform', None)
    new_platform = new['runtime'].pop('platform', None)
    if old != new:
        raise ValueError('Reference model/preprocessing/runtime/extraction differs from this .80 baseline')
    # Only a host OS/platform change is eligible. Recompute every cached vector
    # from the original, checksum-verified photo before using it as a reference.
    _, recognizer = load(models, .8, .3, 5000)
    by_photo = {}
    for face in reference_cache['faces']:
        if face['face_id'] in vectors:
            by_photo.setdefault(face['photo_id'], []).append(face)
    checked, maximum_error, minimum_cosine = 0, 0., 1.
    root = Path(reference_cache['input_root'])
    for photo in reference_cache['photos']:
        if photo['photo_id'] not in by_photo:
            continue
        path = root/photo['relative_path']
        if sha256(path) != photo['sha256']:
            raise ValueError('Original reference photo changed; runtime parity cannot be verified')
        image = decode_oriented(path.read_bytes())
        for face in by_photo[photo['photo_id']]:
            landmarks = face['landmarks']
            row = np.array([face['x'],face['y'],face['width'],face['height'],*landmarks,face['detection_score']], np.float32)
            _, fresh = extract(recognizer, image, row)
            cached = vectors[face['face_id']]
            error = float(np.max(np.abs(cached-fresh)))
            cosine = float(np.clip(cached@fresh,-1,1))
            if error > 1e-5 or cosine < 1-1e-5:
                raise ValueError('Reference vector replay differs after OS change; cannot reuse identity features')
            maximum_error = max(maximum_error,error); minimum_cosine = min(minimum_cosine,cosine); checked += 1
    if checked != len(vectors):
        raise ValueError('Not every reference vector was verified')
    return dict(status='host_platform_change_verified_by_full_vector_replay', previous_platform=old_platform,
                current_platform=new_platform, faces_checked=checked, maximum_absolute_vector_error=maximum_error,
                minimum_cosine=minimum_cosine, absolute_error_limit=1e-5)


def apply_identity_dataset(args):
    from .identity_experiment import _read_faces_for_dataset
    start = time.perf_counter()
    reference = Path(args.reference_run).resolve()
    store_path = Path(args.identity_store).resolve()
    reference_cache = json.loads((reference/'cache.json').read_text(encoding='utf-8'))
    source_store = json.loads(store_path.read_text(encoding='utf-8'))
    validate_store(source_store, reference_cache)
    reference_faces = _read_faces_for_dataset(reference/'faces.csv')
    reference_sha = sha256(reference/'faces.csv')
    store_sha = sha256(store_path)
    reference_vectors = load_vectors(reference, reference_cache)
    if set(source_store['links']) - {f['face_id'] for f in reference_faces}:
        raise ValueError('Reference identity links do not match the source faces')
    options = dict(command='analyze', input=str(Path(args.input).resolve()), output=str(Path(args.output).resolve()),
                   models=args.models, score_threshold=.8, nms_threshold=.3, top_k=5000, max_side=1600,
                   detection_sides=[960,1600,2400], min_face_size=40., min_blur=50.,
                   linkage='average', similarity_threshold=.40, ambiguity_margin=.01,
                   min_group_size=1, reassign_unknown=False)
    runtime_check = verify_reference_runtime(reference_cache, extraction_signature(options), reference_vectors, args.models)
    input_files = inventory(Path(args.input).resolve())
    if not input_files:
        raise ValueError('Input dataset has no supported photos')
    output = analyze(SimpleNamespace(**options))
    target_cache = json.loads((output/'cache.json').read_text(encoding='utf-8'))
    target_faces = _read_faces_for_dataset(output/'faces.csv')
    target_vectors = load_vectors(output, target_cache)
    matches = exact_photo_face_matches(reference_faces, target_faces, reference_cache, target_cache,
                                        reference_vectors, target_vectors)
    excluded = [fid for fid, old in matches.items() if old in source_store['excluded_face_ids']]
    transferred = {fid: source_store['links'][old] for fid, old in matches.items() if old in source_store['links']}
    target_store = empty_store(target_cache, sha256(output/'faces.csv'), excluded)
    target_store['identities'] = deepcopy(source_store['identities'])
    if transferred:
        target_store = apply_identity_changes(target_store, target_cache, transferred,
                                              'Reuse explicit confirmations: exact photo SHA, geometry, landmarks and embedding',
                                              actor='verified_prior_confirmation')
    target_store['reference_store_sha256'] = store_sha
    source_for_reps = content_photo_faces([f for f in reference_faces if f['face_id'] in source_store['links']], reference_cache)
    frozen = frozen_identity_representatives(source_for_reps, reference_vectors, source_store['links'])
    pending = [face for face in target_faces if face['face_id'] not in transferred and face['face_id'] not in excluded]
    query_faces = content_photo_faces(pending, target_cache)
    vectors = dict(reference_vectors)
    # photo_id is a content hash for votes. Face vector keys are namespaced to avoid
    # collisions between independent extractions whose filenames happen to agree.
    remapped_frozen = deepcopy(frozen)
    for identity, reps in remapped_frozen.items():
        for rep in reps:
            old = rep['face_id']; rep['face_id'] = 'reference:' + old
            vectors[rep['face_id']] = reference_vectors[old]
    vectors.update(target_vectors)
    decisions = predict_identities(query_faces, vectors, remapped_frozen)
    predictions = {decision['face_id']: decision['predicted_identity'] for decision in decisions if decision['predicted_identity']}
    view_store = deepcopy(target_store)
    view_store['links'].update(predictions)
    collections = identity_collections(target_faces, view_store)
    origins = {fid:'prior_user_confirmed_exact_photo' for fid in transferred}
    origins.update({fid:'automatic_prediction_unconfirmed' for fid in predictions})
    current_photo_hashes = {item['sha256'] for item in reference_cache['photos']}
    duplicate_photos = sum(item['sha256'] in current_photo_hashes for item in target_cache['photos'])
    new_photo_by_id = {item['photo_id']:item['sha256'] not in current_photo_hashes for item in target_cache['photos']}
    baseline = json.loads((output/'summary.json').read_text(encoding='utf-8'))
    report = dict(reference_run=str(reference), reference_store=str(store_path), reference_store_sha256=store_sha,
                  reference_faces_sha256=reference_sha, target_faces_sha256=sha256(output/'faces.csv'),
                  reference_features_sha256=reference_cache['features_sha256'], target_features_sha256=target_cache['features_sha256'],
                  settings=SETTINGS, extraction_options=options, baseline=baseline,
                  reference_runtime_validation=runtime_check,
                  input_photos=len(target_cache['photos']), duplicate_content_photos=duplicate_photos,
                  new_content_photos=len(target_cache['photos'])-duplicate_photos,
                  detected_faces=len(target_faces), known_identities=len(source_store['identities']),
                  confirmed_transferred_faces=len(transferred), exact_transferred_false_detections=len(excluded),
                  predicted_faces=len(predictions), linked_identity_views=len(collections),
                  held_faces=len(pending)-len(predictions),
                  held_reasons=dict(Counter(d['reason'] for d in decisions if not d['predicted_identity'])),
                  new_photo_predicted_faces=sum(new_photo_by_id[f['photo_id']] for f in target_faces if f['face_id'] in predictions),
                  quality_excluded_in_pending=sum(bool(f.get('quality_reason')) for f in pending),
                  precision=None, recall=None, f1=None,
                  evaluation='No new target ground truth; no automatic accuracy/F1 calculated. Duplicate-content confirmations are not independent evaluation.',
                  seconds=time.perf_counter()-start)
    assert sha256(reference/'faces.csv') == reference_sha and sha256(store_path) == store_sha
    write_json(output/'identity_store.json', target_store)
    write_json(output/'predicted_links.json', dict(confirmed=False, links=predictions, policy='Proposals only; not added to confirmed identity_store'))
    write_json(output/'reference_representatives.json', remapped_frozen)
    write_json(output/'identity_collections.json', collections)
    write_json(output/'identity_decisions.json', decisions)
    write_json(output/'exact_photo_face_matches.json', matches)
    write_json(output/'identity_summary.json', report)
    write_csv(output/'face_identity_links.csv', [dict(face_id=f['face_id'], photo_id=f['photo_id'],
              automatic_group=f['group_id'], identity_id=view_store['links'].get(f['face_id'], ''),
              origin=origins.get(f['face_id'], 'known_false_detection' if f['face_id'] in excluded else 'held'),
              known_identity_alias=(source_store['identities'][view_store['links'][f['face_id']]]['display_name']
                                    if f['face_id'] in view_store['links'] else '')) for f in target_faces],
              ['face_id','photo_id','automatic_group','identity_id','origin','known_identity_alias'])
    fields=['face_id','photo_id','relative_path','manual_person_id','include','annotation_status','review_type','notes','_context_sha256']
    write_csv(output/'identity_ground_truth_template.csv', [dict(face_id=f['face_id'],photo_id=f['photo_id'],
              relative_path=f['relative_path'], manual_person_id=(source_store['identities'][transferred[f['face_id']]]['display_name']
                     if f['face_id'] in transferred else ''), include='0' if f['face_id'] in excluded else '1',
              annotation_status='excluded' if f['face_id'] in excluded else 'labeled' if f['face_id'] in transferred else 'unlabeled',
              review_type='prior_confirmation' if f['face_id'] in transferred else '', notes='',
              _context_sha256=context_hash(target_cache)) for f in target_faces], fields)
    shutil.copy2(output/'index.html', output/'automatic_index.html')
    _render_new_dataset(output, reference, target_faces, collections, origins, source_store, decisions, excluded, report)
    _make_review(output, target_faces, target_cache, transferred, excluded, source_store)
    print(f'Identity dataset result: {output / "index.html"}')
    return output


def _make_review(output, faces, cache, transferred, excluded, source_store):
    review_faces = deepcopy(faces)
    for face in review_faces:
        fid = face['face_id']
        face['manual_person_id'] = source_store['identities'][transferred[fid]]['display_name'] if fid in transferred else ''
        face['include'] = '0' if fid in excluded else '1'
        face['annotation_status'] = 'excluded' if fid in excluded else 'labeled' if fid in transferred else 'unlabeled'
    payload = dict(context=context_hash(cache), source=str(output), faces=review_faces,
                   title='ljm2_dataset 얼굴 정답 확인', review_scope='new-identity-dataset',
                   export_filename='ground_truth_ljm2.csv',
                   review_note='기존 사진과 정확히 일치한 확인만 미리 채웠습니다. 자동 예측 인물ID는 정답에 미리 입력하지 않았습니다.')
    encoded=json.dumps(payload,ensure_ascii=False).replace('&','\\u0026').replace('<','\\u003c').replace('>','\\u003e')
    template=Path(__file__).with_name('review_ui.html').read_text(encoding='utf-8')
    (output/'정답입력.html').write_text(template.replace('__REVIEW_PAYLOAD__',encoded),encoding='utf-8')


def _render_new_dataset(output, reference, faces, collections, origins, source_store, decisions, excluded, report):
    by_id={f['face_id']:f for f in faces}
    def thumb(fid):
        face=by_id[fid]
        return f'<img loading="lazy" src="crops/{html.escape(fid,quote=True)}.jpg" alt="{html.escape(fid,quote=True)}">' if face.get('crop_path') else '<p>정렬얼굴 없음</p>'
    sections=[]
    for identity,row in sorted(collections.items(),key=lambda x:x[1]['display_name']):
        cards=[]
        for fid in row['face_ids']:
            face=by_id[fid]; label='기존 사용자 확인 재사용' if origins[fid]=='prior_user_confirmed_exact_photo' else '자동 예측 · 미확인'
            cards.append(f'<div class="card">{thumb(fid)}<small>{html.escape(fid)}</small><p>{label}</p><small>자동 그룹 {html.escape(face["group_id"] or "미확인")}</small><a href="annotated/{face["photo_id"]}.jpg">원본 위치</a></div>')
        sections.append(f'<section><h2>{html.escape(row["display_name"])} — 얼굴{len(row["face_ids"])}개 / 사진{len(row["photo_ids"])}장</h2><small>{identity}</small><p>현재 데이터의 자동 그룹: {html.escape(", ".join(row["automatic_groups"]))}</p><div class="gallery">{"".join(cards)}</div></section>')
    held=[]
    reason_names={'insufficient_distinct_photo_support':'서로 다른 대표사진2개 지지 부족','mean_below_threshold':'평균0.45 미달',
                  'competing_identity_margin_insufficient':'경쟁인물 차이0.03 미달','no_candidate':'후보없음',
                  'quality_excluded_or_missing_embedding':'품질제외 또는 특징없음'}
    for d in decisions:
        if d['predicted_identity']:
            continue
        fid=d['face_id'];face=by_id[fid]
        candidates=', '.join(source_store['identities'][c['identity_id']]['display_name']+f" 평균{c['score']:.3f}/지지{c['support']}" for c in d['candidates'][:2]) or '없음'
        held.append(f'<div class="card">{thumb(fid)}<small>{html.escape(fid)}</small><p>{html.escape(reason_names.get(d["reason"],d["reason"]))}</p><small>후보: {html.escape(candidates)}</small><small>자동 그룹 {html.escape(face["group_id"] or "미확인")}</small><a href="annotated/{face["photo_id"]}.jpg">원본 위치</a></div>')
    page=f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>ljm2_dataset 인물ID 연결</title>
<style>body{{font:15px sans-serif;line-height:1.7;margin:24px}}.gallery{{display:flex;flex-wrap:wrap;gap:12px}}.card{{width:220px;padding:10px;border:1px solid #ccc}}img{{width:112px;height:112px;object-fit:contain}}small{{display:block;overflow-wrap:anywhere}}section{{border-top:1px solid #bbb;margin-top:24px}}a{{color:#125c9b}}</style>
<h1>ljm2_dataset 인물 ID 연결 테스트</h1><p>사진 {report['input_photos']}장 / 새 내용 {report['new_content_photos']}장 / 기존과 같은 내용 {report['duplicate_content_photos']}장. 검출 {report['detected_faces']}얼굴, 최초 average-linkage 0.40 자동 그룹 {report['baseline']['groups']}개.</p>
<p>기존34인물 중 연결된 인물ID {report['linked_identity_views']}개. 정확히 같은사진에서 확인을 이어받은 얼굴 {report['confirmed_transferred_faces']}개, 자동 예측 {report['predicted_faces']}개, 인물ID 배정 보류 {report['held_faces']}개(품질제외 {report['quality_excluded_in_pending']}개 포함), 기존 확인 오검출 재사용 {report['exact_transferred_false_detections']}개.</p>
<p>예측은 기존 확인 얼굴의 고정 대표로만 계산했습니다. 서로 다른 대표사진2개 모두 유사도0.40 이상, 평균0.45 이상, 경쟁인물 차이0.03을 요구합니다. 새로 예측된 얼굴은 대표로 사용하지 않았습니다. 보류는 새 인물 확정이 아니며 기존인물 연결 근거 부족일 수 있습니다. target 자동 그룹 번호와 기존 인물 별칭은 같은 번호여도 같은 사람을 의미하지 않습니다.</p>
<p>자동 예측은 사용자 미확인 상태입니다. 새 데이터 정답이 없어 정확도·F1을 만들지 않았습니다. 기존사진6장 재사용 부분도 독립평가가 아닙니다.</p>
<p><a href="automatic_index.html">원래 자동 그룹 결과</a> · <a href="정답입력.html">새 데이터 정답 입력</a> · <a href="identity_summary.json">설정과 요약</a> · <a href="face_identity_links.csv">얼굴별 연결</a> · <a href="identity_decisions.json">모든 후보점수</a> · <a href="identity_store.json">확인된 연결만 저장</a> · <a href="predicted_links.json">미확인 자동예측 별도 저장</a></p>
<h2>인물 ID별 모음 — 확인/자동예측 구분</h2>{''.join(sections)}<h2>인물 ID 연결 보류</h2><div class="gallery">{''.join(held)}</div></html>'''
    (output/'index.html').write_text(page,encoding='utf-8')
