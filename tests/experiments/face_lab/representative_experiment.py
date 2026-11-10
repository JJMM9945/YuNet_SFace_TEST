"""Local exploratory comparisons with frozen, quality/diversity-selected representatives.

Only the explicit false-detection IDs influence grouping. Identity annotations are
read for evaluation after all predictions have been computed.
"""
import csv
import html
import json
import math
import os
import shutil
import time
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from .evaluation import pair_metrics, read_annotations
from .grouping import group_faces, mark_same_photo
from .models import MODELS, sha256, verify
from .pipeline import extraction_signature, finish, inventory, load_vectors, new_run, write_csv, write_json
from .review import context_hash

SETTINGS = dict(representatives=3, diversity_weight=.3, min_support=2, support_threshold=.40,
                score_threshold=.45, margin=.03)


def quality(face):
    """Measured detector confidence, aligned blur and original-pixel face size."""
    components = [face.get('detection_score', 0) or 0,
                  min(face.get('width', 0), face.get('height', 0))/160,
                  (face.get('blur_variance', 0) or 0)/200]
    return float(np.prod(np.clip(components, 0, 1)) ** (1/3))


def representatives(members, vectors, count=3, weight=.3):
    remaining = sorted(members, key=lambda f: f['face_id'])
    selected, audit = [], []
    while remaining and len(selected) < count:
        scored = []
        for f in remaining:
            q = quality(f)
            diversity = min((float(1-np.clip(vectors[f['face_id']] @ vectors[s['face_id']], -1, 1))/2
                             for s in selected), default=0.)
            value = q if not selected else (1-weight)*q + weight*diversity
            scored.append((value, f['face_id'], f, q, diversity))
        value, _, chosen, q, diversity = min(scored, key=lambda x: (-x[0], x[1]))
        selected.append(chosen); remaining.remove(chosen)
        audit.append(dict(face_id=chosen['face_id'], quality=q, diversity=diversity, selection_score=value))
    return selected, audit


def freeze_groups(faces, vectors, settings=SETTINGS):
    groups = defaultdict(list)
    for f in faces:
        if f['group_id'] and not f.get('quality_reason'):
            groups[f['group_id']].append(f)
    frozen, audit, flagged = {}, {}, set()
    for gid, members in sorted(groups.items()):
        frozen[gid], audit[gid] = representatives(members, vectors, settings['representatives'], settings['diversity_weight'])
        if any('same_photo_same_group' in f.get('review', '') for f in members):
            flagged.add(gid)
    return frozen, audit, flagged


def unknown_reassignment(faces, vectors, settings=SETTINGS):
    frozen, audit, flagged = freeze_groups(faces, vectors, settings)
    decisions = []
    for f in sorted(faces, key=lambda v:v['face_id']):
        if f['status'] != 'unknown' or f.get('quality_reason'):
            continue
        candidates = []
        for gid, refs in frozen.items():
            scores = [float(np.clip(vectors[f['face_id']] @ vectors[r['face_id']], -1, 1)) for r in refs]
            candidates.append(dict(group_id=gid, score=float(np.mean(scores)),
                                   support=sum(s >= settings['support_threshold'] for s in scores),
                                   representatives=[dict(face_id=r['face_id'], cosine=s) for r,s in zip(refs,scores)],
                                   review_flagged=gid in flagged))
        candidates.sort(key=lambda c:(-c['score'], c['group_id']))
        first = candidates[0] if candidates else None
        gap = first['score']-candidates[1]['score'] if len(candidates)>1 else None
        if not first: reason='no_candidate'
        elif first['score'] < settings['score_threshold']: reason='score_below_threshold'
        elif first['support'] < settings['min_support']: reason='insufficient_support'
        elif first['review_flagged']: reason='candidate_group_already_flagged'
        elif gap is not None and gap < settings['margin']: reason='competing_group_margin_insufficient'
        else: reason='assigned_to_frozen_representatives'
        before = dict(group_id=f['group_id'],status=f['status'],reason=f['reason'])
        if reason=='assigned_to_frozen_representatives':
            f.update(group_id=first['group_id'],label='인물'+first['group_id'][6:],status='grouped',reason='',decision_state='assigned')
        decision = dict(face_id=f['face_id'],before=before,after_group=f['group_id'],reason=reason,
                        score_gap=gap,candidates=candidates,settings=settings)
        f['diagnostic']['representative_reassignment'] = decision
        f['diagnostic']['stages'].append(dict(stage='frozen_representatives',state='assigned' if f['group_id'] else 'held',reason=reason))
        f['diagnostic'].update(final_group_id=f['group_id'], state='assigned' if f['group_id'] else 'held', reason=f['reason'])
        decisions.append(decision)
    mark_same_photo(faces)
    return dict(representatives=audit, flagged_groups=sorted(flagged), decisions=decisions,
                score_definition='mean cosine against all frozen representatives; minimum two representatives above support threshold',
                missing_second='null gap: no competition check; score and support checks still apply', settings=settings)


def merge_groups(faces, vectors, settings=SETTINGS):
    frozen, audit, flagged = freeze_groups(faces, vectors, settings)
    ranks = {}
    for gid, reps in frozen.items():
        candidates = []
        for other, other_reps in frozen.items():
            if gid == other: continue
            matrix = np.array([[float(np.clip(vectors[a['face_id']] @ vectors[b['face_id']],-1,1))
                                for b in other_reps] for a in reps])
            candidates.append(dict(group_id=other,score=float(matrix.mean()),
                                   own_support=int(np.sum(np.max(matrix,axis=1)>=settings['support_threshold'])),
                                   other_support=int(np.sum(np.max(matrix,axis=0)>=settings['support_threshold'])),
                                   similarities=matrix.tolist()))
        ranks[gid] = sorted(candidates,key=lambda c:(-c['score'],c['group_id']))
    decisions, accepted = [], []
    for gid in sorted(frozen):
        candidates = ranks[gid]
        first = candidates[0] if candidates else None
        other = first['group_id'] if first else None
        gap = first['score']-candidates[1]['score'] if len(candidates)>1 else None
        if not first: reason='no_candidate'
        elif first['score'] < settings['score_threshold']: reason='score_below_threshold'
        elif first['own_support'] < settings['min_support'] or first['other_support'] < settings['min_support']: reason='insufficient_support'
        elif gid in flagged or other in flagged: reason='candidate_group_already_flagged'
        elif gap is not None and gap < settings['margin']: reason='competing_group_margin_insufficient'
        elif not ranks[other] or ranks[other][0]['group_id'] != gid: reason='not_mutual_best'
        elif len(ranks[other])>1 and ranks[other][0]['score']-ranks[other][1]['score'] < settings['margin']: reason='other_group_margin_insufficient'
        else:
            reason='mutual_frozen_merge'
            if gid < other: accepted.append((gid,other))
        decisions.append(dict(group_id=gid,reason=reason,score_gap=gap,candidates=candidates))
    # Reciprocal best pairs are disjoint. No group can merge repeatedly or create a chain.
    if len({g for pair in accepted for g in pair}) != 2*len(accepted):
        raise ValueError('Overlapping representative merges')
    remap={other:gid for gid,other in accepted}
    involved={group:pair for pair in accepted for group in pair}
    for f in faces:
        if f['group_id'] in involved:
            before=f['group_id'];f['group_id']=remap.get(before,before);f['label']='인물'+f['group_id'][6:]
            f['diagnostic']['representative_merge']=dict(before_group=before,after_group=f['group_id'],
                                                         merged_groups=involved[before],reason='mutual_frozen_merge')
            f['diagnostic']['stages'].append(dict(stage='frozen_group_merge',state='assigned',reason='mutual_frozen_merge',group_id=f['group_id']))
            f['diagnostic']['final_group_id']=f['group_id']
    mark_same_photo(faces)
    return dict(representatives=audit,flagged_groups=sorted(flagged),decisions=decisions,accepted_pairs=accepted,
                score_definition='mean of all frozen cross-representative cosine scores; >=2 supporting representatives per side; reciprocal best with margins',
                settings=settings)


def read_faces(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def memberships(faces):
    grouped=defaultdict(list)
    for f in faces:
        if f['group_id']: grouped[f['group_id']].append(f['face_id'])
    return {f['face_id']: sorted(grouped[f['group_id']]) if f['group_id'] else None for f in faces}


def validate_count_reference(reference, settings, provenance):
    if any(reference.get(key) != value for key,value in provenance.items()):
        raise ValueError('Reference comparison uses different source, features, truth or false detections')
    previous = reference['settings']
    if (set(previous) != set(settings) or
        any(previous[k] != settings[k] for k in settings if k != 'representatives') or
        previous['representatives'] == settings['representatives']):
        raise ValueError('Reference must differ only in representative count')
    return previous


def compare_representatives(args):
    source, input_root = Path(args.source).resolve(), Path(args.input).resolve()
    feedback_path, truth_path = Path(args.corrections).resolve(), Path(args.ground_truth).resolve()
    settings = dict(SETTINGS, representatives=getattr(args, 'representatives', 3),
                    score_threshold=getattr(args, 'representative_score_threshold', .45))
    if not math.isfinite(settings['score_threshold']) or not -1 <= settings['score_threshold'] <= 1:
        raise ValueError('Representative mean cosine threshold must be between -1 and 1')
    if settings['representatives'] < settings['min_support']:
        raise ValueError('Representative count must be at least the minimum support')
    cache=json.loads((source/'cache.json').read_text(encoding='utf-8'))
    feedback=json.loads(feedback_path.read_text(encoding='utf-8'))
    summary=json.loads((source/'summary.json').read_text(encoding='utf-8'))
    if summary.get('mode') not in {'analyze','regroup'}:
        raise ValueError('Source must be an original automatic run')
    if cache.get('schema')!=1 or cache['signature']!=extraction_signature(cache['config']):
        raise ValueError('Cache model/preprocessing/runtime mismatch')
    if inventory(input_root)!=cache['input_files']:
        raise ValueError('Input photos differ from source extraction')
    for spec in MODELS.values(): verify(Path(args.models)/spec['name'],spec)
    if feedback.get('schema')!=1 or feedback.get('context_sha256')!=context_hash(cache) or feedback.get('source_faces_sha256')!=sha256(source/'faces.csv') or feedback.get('features_sha256')!=cache['features_sha256']:
        raise ValueError('False-detection feedback does not match the source extraction and grouping')
    vectors=load_vectors(source,cache)
    original=read_faces(source/'faces.csv')
    if len(original)!=len(cache['faces']) or {f['face_id'] for f in original}!={f['face_id'] for f in cache['faces']}:
        raise ValueError('Source face mapping mismatch')
    false=feedback.get('false_detections',{})
    if (not isinstance(false,dict) or not false or set(false)-{f['face_id'] for f in original}
            or any(not isinstance(reason, str) or not reason.strip() for reason in false.values())):
        raise ValueError('Explicit valid false-detection IDs required')
    for name in ('photos','crops'):
        if not (source/name).is_dir(): raise ValueError(f'Missing source {name}')
    diagnostics=json.loads((source/'diagnostics.json').read_text(encoding='utf-8'))
    if set(diagnostics)!={f['face_id'] for f in original}: raise ValueError('Source diagnostic mapping mismatch')
    # Ground truth is validated now but identity labels are never passed to any grouping method.
    annotations=read_annotations(truth_path,cache['faces'],cache)
    for fid in false:
        if annotations.get(fid,{}).get('include')!='0': raise ValueError('Known false detections must be excluded in the evaluation CSV')
    opts=json.loads((source/'run_config.json').read_text(encoding='utf-8'))['options']
    if summary['actual_grouping_config']['linkage']!='average': raise ValueError('This comparison expects the current average-linkage baseline')
    reference_path = Path(args.compare_with).resolve() if getattr(args, 'compare_with', None) else None
    reference = None
    if reference_path:
        reference = json.loads((reference_path/'comparison.json').read_text(encoding='utf-8'))
        reference_settings = validate_count_reference(reference, settings, dict(
            source_faces_sha256=sha256(source/'faces.csv'), features_sha256=cache['features_sha256'],
            truth_sha256=sha256(truth_path), feedback_sha256=sha256(feedback_path)))
    group_opts=dict(method=opts['linkage'],threshold=opts['similarity_threshold'],min_group=opts['min_group_size'],margin=opts['ambiguity_margin'])
    faces=deepcopy(cache['faces'])
    eligible=[f for f in faces if f['face_id'] not in false]
    t=time.perf_counter();group_faces(eligible,vectors,**group_opts)
    for f in faces:
        fid=f['face_id']
        if fid in false:
            d=deepcopy(diagnostics[fid]);d['stages'].append(dict(stage='user_false_detection_exclusion',state='excluded',reason=false[fid]))
            d.update(state='false_detection',reason='false_detection',final_group_id='')
            f.update(group_id='',label='오검출',status='false_detection',reason='false_detection',review='',
                     label_origin='user_confirmed_false_detection',decision_state='false_detection',correction_note=false[fid],diagnostic=d)
    if reference_path:
        prior = {f['face_id']: f for f in read_faces(reference_path/'false_excluded'/'faces.csv')}
        if (set(prior) != {f['face_id'] for f in faces} or
            any((f['group_id'],f['status'],f['reason']) !=
                (prior[f['face_id']]['group_id'],prior[f['face_id']]['status'],prior[f['face_id']]['reason']) for f in faces)):
            raise ValueError('Reference false-excluded grouping differs; cannot isolate representative count')
        for name in ('false_excluded', 'representative_merge', 'representative_reassignment'):
            previous_faces = read_faces(reference_path/name/'faces.csv')
            if (len(previous_faces) != len(faces) or
                {f['face_id'] for f in previous_faces} != set(prior) or
                pair_metrics(previous_faces, annotations) != reference['results'][name]['evaluation']):
                raise ValueError('Reference evaluation or face mapping mismatch')
    exclusion_compute=time.perf_counter()-t
    # Separate copies make each change independently comparable to exclusion-only.
    merged=deepcopy(faces); reassigned=deepcopy(faces)
    t=time.perf_counter();merge_audit=merge_groups(merged,vectors,settings);merge_compute=time.perf_counter()-t
    t=time.perf_counter();reassign_audit=unknown_reassignment(reassigned,vectors,settings);reassign_compute=time.perf_counter()-t
    if reference_path:
        for name, audit in [('representative_merge',merge_audit),('representative_reassignment',reassign_audit)]:
            prior_audit = json.loads((reference_path/name/'representative_diagnostics.json').read_text(encoding='utf-8'))
            if prior_audit['settings'] != reference['settings'] or set(prior_audit['representatives']) != set(audit['representatives']):
                raise ValueError('Reference representative diagnostics mismatch')
            small, large = ((prior_audit, audit) if reference_settings['representatives'] < settings['representatives'] else (audit, prior_audit))
            for gid, reps in small['representatives'].items():
                if [r['face_id'] for r in reps] != [r['face_id'] for r in large['representatives'][gid][:len(reps)]]:
                    raise ValueError('Reference representative order differs; cannot isolate count')
    parent=new_run(args.output,input_root)
    experiments=[('false_excluded',faces,exclusion_compute,None),('representative_merge',merged,exclusion_compute+merge_compute,merge_audit),
                 ('representative_reassignment',reassigned,exclusion_compute+reassign_compute,reassign_audit)]
    results={}; summary_rows=[]; all_changes=[]
    baseline_metric=pair_metrics(original,annotations)
    baseline_groups=memberships(original)
    source_ref=os.path.relpath(source/'index.html',parent)
    def append_summary(name,summary_data,metric,compute,link):
        summary_rows.append(dict(experiment=name,detected_faces=summary_data['detected_faces'],
            grouping_eligible_faces=summary_data['grouping_eligible_faces'],groups=summary_data['groups'],
            singleton_groups=summary_data['singleton_groups'],unknown_faces=summary_data['unknown_faces'],
            false_detections=summary_data.get('false_detections',0),evaluated_faces=metric['evaluated_faces'],
            precision=metric['precision'],recall=metric['recall'],f1=metric['f1'],
            true_positive_pairs=metric['true_positive_pairs'],false_positive_pairs=metric['false_positive_pairs'],
            false_negative_pairs=metric['false_negative_pairs'],mixed_groups=len(metric['mixed_groups']),
            split_people=len(metric['split_people']),same_photo_review_groups=len(summary_data['same_photo_review_groups']),
            compute_seconds=compute,link=link))
    append_summary('baseline',summary,baseline_metric,summary['grouping_seconds'],source_ref)
    results['baseline']=dict(summary=summary,evaluation=baseline_metric,link=source_ref)
    for name,predicted,compute,audit in experiments:
        child=parent/name;child.mkdir()
        for folder in ('photos','crops','annotated','montages'): (child/folder).mkdir()
        for folder in ('photos','crops'): shutil.copytree(source/folder,child/folder,dirs_exist_ok=True)
        shutil.copy2(source/'features.npz',child/'features.npz');write_json(child/'cache.json',cache)
        childopts=opts.copy();childopts.update(command='compare-representatives',input=str(input_root),output=str(parent),experiment=name,
                                              representative_settings=settings)
        if audit is not None: write_json(child/'representative_diagnostics.json',audit)
        start=time.perf_counter()
        finish(child,deepcopy(cache['photos']),predicted,vectors,SimpleNamespace(**childopts),start,0.,0.,cache,'experimental-regroup')
        child_summary=json.loads((child/'summary.json').read_text())
        metric=pair_metrics(predicted,annotations);write_json(child/'partial_evaluation.json',metric)
        results[name]=dict(summary=child_summary,evaluation=metric,compute_seconds=compute,
                           changed_by_method=(audit['accepted_pairs'] if name=='representative_merge' else
                             [d['face_id'] for d in audit['decisions'] if d['reason']=='assigned_to_frozen_representatives'] if audit else []),
                           link=name+'/index.html')
        append_summary(name,child_summary,metric,compute,name+'/index.html')
        current={f['face_id']:f for f in predicted};source_by_id={f['face_id']:f for f in original};current_groups=memberships(predicted)
        for fid,f in current.items():
            old=source_by_id[fid]
            all_changes.append(dict(experiment=name,face_id=fid,photo_id=f['photo_id'],relative_path=f['relative_path'],
                                    before_group=old['group_id'],after_group=f['group_id'],before_status=old['status'],after_status=f['status'],
                                    membership_changed=baseline_groups[fid]!=current_groups[fid],same_photo_review=f['review']))
    write_csv(parent/'comparison.csv',summary_rows,list(summary_rows[0]))
    write_csv(parent/'face_changes.csv',all_changes,list(all_changes[0]))
    report=dict(source=str(source),source_faces_sha256=sha256(source/'faces.csv'),features_sha256=cache['features_sha256'],
                truth_sha256=sha256(truth_path),feedback_sha256=sha256(feedback_path),settings=settings,
                quality_formula='cuberoot(clip(detection,0,1)*clip(min_side/160,0,1)*clip(blur/200,0,1))',
                representative_selection='highest quality, then 0.7*quality+0.3*minimum((1-cosine)/2)',
                leakage='Same labeled data provided known false detections and later partial evaluation; exploratory, not independent accuracy. Identity labels not used in grouping or choosing settings.',
                limitations='Only manually selected labeled faces are evaluated; missed detections and unlabeled faces are outside pair evaluation. No automatic best selection.',
                results=results)
    write_json(parent/'comparison.json',report)
    if reference_path:
        render_count_comparison(parent, reference_path, reference, report)
    render_comparison(parent,summary_rows,results,all_changes)
    print(f'Comparison: {parent}/index.html\nKnown false detections from user feedback; partial evaluation is exploratory.')
    return parent


def render_comparison(parent,rows,results,changes):
    settings=json.loads((parent/'comparison.json').read_text(encoding='utf-8'))['settings']
    columns=[('experiment','실험'),('grouping_eligible_faces','대상'),('groups','그룹'),('singleton_groups','단독'),
             ('unknown_faces','미확인'),('false_detections','제외한 오검출'),('evaluated_faces','평가 얼굴'),
             ('precision','Precision'),('recall','Recall'),('f1','F1'),('false_positive_pairs','혼합 쌍 FP'),
             ('false_negative_pairs','분리 쌍 FN'),('compute_seconds','그룹화 초')]
    page=['<!doctype html><html lang="ko"><meta charset="utf-8"><title>부분 정답 탐색 비교</title>',
          '<style>body{font:15px sans-serif;margin:24px}td,th{padding:7px;border:1px solid #ddd}table{border-collapse:collapse}pre{white-space:pre-wrap}</style>',
          '<h1>오검출 제외와 대표 얼굴 비교</h1>',
          f'<p>오검출은 사용자가 직접 확인한 정보입니다. 같은 데이터의 부분 정답 {rows[0]["evaluated_faces"]}개로 비교한 탐색 결과이며 독립적인 정확도 검증이 아닙니다. 정답 ID는 그룹화에 사용하지 않았습니다.</p>',
          '<p>baseline: 원본 · false_excluded: 확인한 오검출만 제외 후 기존 average · representative_merge: 제외 후 고정 대표 그룹 병합 · representative_reassignment: 제외 후 고정 대표로 미확인 배정. 두 대표 실험은 서로 별개입니다. 오검출 제외로 미확인이 줄어든 것은 자동 재배정과 다릅니다.</p>',
          f'<p>대표는 검출 점수·얼굴 크기·흐림 및 벡터 다양성으로 최대 {settings["representatives"]}장 고정합니다. 각 단계 점수와 근거는 실험별 representative_diagnostics.json에 남깁니다. 후보 부족·경쟁 모호성은 보류합니다.</p>',
          '<p><a href="comparison.csv">요약 CSV</a> · <a href="comparison.json">전체 설정과 평가</a> · <a href="face_changes.csv">얼굴별 변화</a></p>',
          '<table><tr>'+''.join('<th>'+label+'</th>' for _,label in columns)+'</tr>']
    if (parent/'representative_count_comparison.html').exists():
        page.insert(-1, '<p><strong><a href="representative_count_comparison.html">대표 얼굴 수 비교 보기</a></strong></p>')
    for row in rows:
        cells=[]
        for k,_ in columns:
            v=row[k];shown='미정의' if v is None else f'{v:.4f}' if isinstance(v,float) else str(v)
            value=html.escape(shown)
            if k=='experiment':value=f'<a href="{html.escape(row["link"],quote=True)}">{value}</a>'
            cells.append('<td>'+value+'</td>')
        page.append('<tr>'+''.join(cells)+'</tr>')
    page.append('</table><h2>대표 얼굴 판정</h2>')
    reasons={'score_below_threshold':'평균 점수 기준 미달','insufficient_support':'대표 지지 2장 미달',
             'candidate_group_already_flagged':'후보 그룹에 기존 검토 표시',
             'competing_group_margin_insufficient':'경쟁 그룹과 점수 차이 부족',
             'other_group_margin_insufficient':'상대 그룹의 경쟁 점수 차이 부족',
             'not_mutual_best':'서로의 1순위가 아님','no_candidate':'후보 없음',
             'mutual_frozen_merge':'그룹 병합','assigned_to_frozen_representatives':'미확인 재배정'}
    for name,title in [('representative_merge','그룹 병합'),('representative_reassignment','미확인 재배정')]:
        audit=json.loads((parent/name/'representative_diagnostics.json').read_text(encoding='utf-8'))
        counts=Counter(reasons.get(d['reason'],d['reason']) for d in audit['decisions'])
        page.append('<h3>'+title+'</h3><p>'+html.escape(', '.join(f'{k}: {v}' for k,v in sorted(counts.items())))+'</p>')
        page.append(f'<p><a href="{name}/representative_diagnostics.json">대표 선택·모든 후보 상세</a></p>')
        if name=='representative_reassignment':
            page.append('<table><tr><th>얼굴</th><th>판정</th><th>상위 후보: 평균점수 / 지지 수</th></tr>')
            for d in audit['decisions']:
                fid=d['face_id']; candidates=', '.join(f"{c['group_id']}: {c['score']:.4f} / {c['support']}장" for c in d['candidates'][:2])
                page.append(f'<tr><td><a href="{name}/index.html#{fid}">{fid}</a></td><td>{html.escape(reasons.get(d["reason"],d["reason"]))}</td><td>{html.escape(candidates)}</td></tr>')
            page.append('</table>')
    page.append('<h2>변경된 얼굴</h2><p>그룹 번호만 바뀐 경우와 구성원 변화는 구별합니다. 자세한 것은 CSV를 확인하세요.</p><table><tr><th>실험</th><th>얼굴</th><th>그룹 전 → 후</th><th>상태 전 → 후</th></tr>')
    for c in changes:
        if not c['membership_changed'] and c['before_status']==c['after_status']:continue
        name,fid=c['experiment'],c['face_id']
        page.append(f'<tr><td>{html.escape(name)}</td><td><a href="{name}/index.html#{fid}">{fid}</a></td><td>{html.escape(c["before_group"] or "미확인")} → {html.escape(c["after_group"] or "미확인")}</td><td>{html.escape(c["before_status"])} → {html.escape(c["after_status"])}</td></tr>')
    page.append('</table><h2>혼합·분리 사례</h2>')
    for name,result in results.items():
        m=result['evaluation']
        page.append('<h3>'+html.escape(name)+'</h3><pre>'+html.escape(json.dumps(dict(mixed_groups=m['mixed_groups'],split_people=m['split_people']),ensure_ascii=False,indent=2))+'</pre>')
    page.append('</html>');(parent/'index.html').write_text('\n'.join(page),encoding='utf-8')


def render_count_comparison(parent, reference_path, reference, current):
    """Compare only representative count on identical initial groups and evaluations."""
    old_count=reference['settings']['representatives'];new_count=current['settings']['representatives']
    rows=[];face_changes=[];candidate_changes=[];selection_summary={}
    methods=[('representative_merge','대표 기반 그룹 병합'),('representative_reassignment','대표 기반 미확인 재배정')]
    for method,title in methods:
        old_faces=read_faces(reference_path/method/'faces.csv');new_faces=read_faces(parent/method/'faces.csv')
        old_members,new_members=memberships(old_faces),memberships(new_faces)
        old_map={f['face_id']:f for f in old_faces};new_map={f['face_id']:f for f in new_faces}
        if set(old_map)!=set(new_map):raise ValueError('Count comparison face mapping differs')
        old_audit=json.loads((reference_path/method/'representative_diagnostics.json').read_text(encoding='utf-8'))
        new_audit=json.loads((parent/method/'representative_diagnostics.json').read_text(encoding='utf-8'))
        expanded=[gid for gid in old_audit['representatives'] if len(old_audit['representatives'][gid]) != len(new_audit['representatives'][gid])]
        selection_summary[method]=dict(groups=len(old_audit['representatives']),groups_with_changed_representative_count=expanded,
            old_selected=sum(len(v) for v in old_audit['representatives'].values()),
            new_selected=sum(len(v) for v in new_audit['representatives'].values()))
        for count,report,audit,location in ((old_count,reference,old_audit,reference_path/method),
                                            (new_count,current,new_audit,parent/method)):
            result=report['results'][method];m=result['evaluation'];s=result['summary']
            rows.append(dict(method=method,method_label=title,max_representatives=count,
                             groups=s['groups'],singleton_groups=s['singleton_groups'],unknown_faces=s['unknown_faces'],
                             false_detections=s['false_detections'],evaluated_faces=m['evaluated_faces'],
                             precision=m['precision'],recall=m['recall'],f1=m['f1'],
                             true_positive_pairs=m['true_positive_pairs'],false_positive_pairs=m['false_positive_pairs'],
                             false_negative_pairs=m['false_negative_pairs'],
                             changed_by_method=len(result['changed_by_method']),
                             selected_representatives=sum(len(v) for v in audit['representatives'].values()),
                             result_link=os.path.relpath(location/'index.html',parent),
                             diagnostics_link=os.path.relpath(location/'representative_diagnostics.json',parent)))
        for fid in sorted(old_map):
            before,after=old_map[fid],new_map[fid]
            face_changes.append(dict(method=method,face_id=fid,photo_id=before['photo_id'],relative_path=before['relative_path'],
                    old_group=before['group_id'],new_group=after['group_id'],old_status=before['status'],new_status=after['status'],
                    membership_changed=old_members[fid]!=new_members[fid]))
        key='group_id' if method=='representative_merge' else 'face_id'
        old_decisions={d[key]:d for d in old_audit['decisions']};new_decisions={d[key]:d for d in new_audit['decisions']}
        if set(old_decisions)!=set(new_decisions):raise ValueError('Count comparison candidate mapping differs')
        for ident in sorted(old_decisions):
            def top(d):
                return d['candidates'][0] if d['candidates'] else None
            a,b=old_decisions[ident],new_decisions[ident];x,y=top(a),top(b)
            candidate_changes.append(dict(method=method,item_id=ident,
                 old_reason=a['reason'],new_reason=b['reason'],
                 old_top_group=x['group_id'] if x else '',new_top_group=y['group_id'] if y else '',
                 old_top_score=x['score'] if x else None,new_top_score=y['score'] if y else None,
                 old_top_support=(x.get('support',x.get('own_support')) if x else None),
                 new_top_support=(y.get('support',y.get('own_support')) if y else None),
                 old_gap=a['score_gap'],new_gap=b['score_gap']))
    write_csv(parent/'representative_counts.csv',rows,list(rows[0]))
    write_csv(parent/'representative_face_changes.csv',face_changes,list(face_changes[0]))
    write_csv(parent/'representative_candidate_changes.csv',candidate_changes,list(candidate_changes[0]))
    comparison=dict(reference=str(reference_path),reference_sha256=sha256(reference_path/'comparison.json'),
                    old_settings=reference['settings'],new_settings=current['settings'],
                    same_false_excluded_grouping=True,representative_prefix_verified=True,
                    selection_summary=selection_summary,rows=rows,
                    leakage='Known false detections were manually identified; same partially annotated dataset used for this exploratory comparison.',
                    note='Only representative maximum changed; no accuracy winner is selected. Representatives are quality/diversity selected and frozen.')
    write_json(parent/'representative_counts.json',comparison)
    page=['<!doctype html><html lang="ko"><meta charset="utf-8"><title>대표 얼굴 수 비교</title>',
          '<style>body{font:15px sans-serif;margin:24px}td,th{padding:7px;border:1px solid #ddd}table{border-collapse:collapse}pre{white-space:pre-wrap}</style>',
          f'<h1>대표 최대 {old_count}장과 {new_count}장 비교</h1>',
          '<p>같은 임베딩·오검출 제외·초기 그룹에서 최대 대표 수만 바꿨습니다. 실제 얼굴이 적은 그룹은 가진 얼굴 수만큼 선택합니다. 대표는 품질·다양성으로 선택 후 고정합니다.</p>',
          '<p>검출 점수와 얼굴 벡터는 동일하며, 평균 점수 0.45·지지 최소 2장(개별 0.40 이상)·경쟁 폭 0.03을 유지합니다. 5장에서는 대표 전체에 대한 평균을 사용하므로 점수 자체가 달라질 수 있습니다.</p>',
          '<p>사용자가 확인한 오검출과 같은 사진의 부분 정답을 사용한 탐색 비교입니다. 확인하지 않은 얼굴과 미검출 얼굴은 정확도 평가 밖이며 자동으로 우수 설정을 선택하지 않습니다.</p>',
          '<p><a href="representative_counts.csv">요약 CSV</a> · <a href="representative_counts.json">설정·결과 JSON</a> · <a href="representative_face_changes.csv">얼굴 소속 변화</a> · <a href="representative_candidate_changes.csv">후보 점수 변화</a> · <a href="index.html">이번 실행 전체</a></p>',
          '<table><tr><th>실험</th><th>대표 최대</th><th>실제 선택</th><th>그룹</th><th>미확인</th><th>자동 변경</th><th>평가 얼굴</th><th>Precision</th><th>Recall</th><th>F1</th><th>FP 쌍</th><th>FN 쌍</th><th>결과</th></tr>']
    for r in rows:
        vals=[r['method_label'],r['max_representatives'],r['selected_representatives'],r['groups'],r['unknown_faces'],
              r['changed_by_method'],r['evaluated_faces']]
        metrics=[r['precision'],r['recall'],r['f1']]
        cells=''.join('<td>'+html.escape(str(v))+'</td>' for v in vals)
        cells+=''.join('<td>'+('미정의' if v is None else f'{v:.4f}')+'</td>' for v in metrics)
        cells+=f'<td>{r["false_positive_pairs"]}</td><td>{r["false_negative_pairs"]}</td>'
        cells+=f'<td><a href="{html.escape(r["result_link"],quote=True)}">결과</a> · <a href="{html.escape(r["diagnostics_link"],quote=True)}">판정 기록</a></td>'
        page.append('<tr>'+cells+'</tr>')
    page.append('</table><h2>실제 대표 수가 달라진 그룹</h2>')
    for method,title in methods:
        s=selection_summary[method]
        page.append(f'<p>{html.escape(title)}: {s["groups"]}개 초기 그룹 중 {len(s["groups_with_changed_representative_count"])}개. 전체 대표 {s["old_selected"]} → {s["new_selected"]}장.</p>')
    page.append('<h2>미확인 얼굴별 상위 후보</h2><table><tr><th>얼굴</th><th>이전: 후보 / 점수 / 지지 / 판정</th><th>변경: 후보 / 점수 / 지지 / 판정</th></tr>')
    for c in candidate_changes:
        if c['method']!='representative_reassignment':continue
        def details(prefix):
            score=c[prefix+'_top_score'];score='없음' if score is None else f'{score:.4f}'
            return f"{c[prefix+'_top_group'] or '없음'} / {score} / {c[prefix+'_top_support']}장 / {c[prefix+'_reason']}"
        fid=c['item_id']
        page.append(f'<tr><td><a href="representative_reassignment/index.html#{fid}">{html.escape(fid)}</a></td><td>{html.escape(details("old"))}</td><td>{html.escape(details("new"))}</td></tr>')
    page.append('</table><h2>얼굴 소속 변화</h2><table><tr><th>실험</th><th>얼굴</th><th>이전 그룹 → 새 그룹</th></tr>')
    for f in face_changes:
        if not f['membership_changed'] and f['old_status']==f['new_status']:continue
        page.append(f'<tr><td>{html.escape(f["method"])}</td><td><a href="{f["method"]}/index.html#{f["face_id"]}">{html.escape(f["face_id"])}</a></td><td>{html.escape(f["old_group"] or "미확인")} → {html.escape(f["new_group"] or "미확인")}</td></tr>')
    page.append('</table></html>')
    (parent/'representative_count_comparison.html').write_text('\n'.join(page),encoding='utf-8')
