"""Frozen automatic anchors: predictions never receive human identity annotations."""
import csv
import html
import json
import os
import time
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path

import numpy as np

from .evaluation import pair_metrics, read_annotations
from .grouping import mark_same_photo
from .models import sha256
from .pipeline import load_vectors, new_run, write_csv, write_json
from .representative_experiment import quality, representatives


SETTINGS = dict(anchor_min_faces=3, anchor_min_photos=3, anchor_median_similarity=.50,
                anchor_p10_similarity=.30, anchor_mean_competitor_gap=.05,
                anchor_median_quality=.60, representatives=3, diversity_weight=.30,
                max_small_group=2, score_threshold=.45, support_threshold=.40,
                min_support=2, margin=.03, singleton_score_threshold=.50, singleton_margin=.05)

REASONS = dict(no_measured_candidate='비교 가능한 후보 없음',
               top_candidate_not_automatic_anchor='1순위가 안정적인 자동 대표 그룹이 아님',
               insufficient_distinct_photo_support='서로 다른 대표 사진 2장의 지지 부족',
               score_below_threshold='평균 또는 그룹 구성원 점수 기준 미달',
               competing_group_margin_insufficient='경쟁 그룹과 점수 차이 부족',
               same_photo_review='같은 사진 얼굴 충돌 검토 필요',
               assigned_to_frozen_automatic_anchor='고정 자동 대표 그룹에 연결')


def typed_faces(faces):
    result = deepcopy(faces)
    for face in result:
        for key in ('x','y','width','height','detection_score','blur_variance'):
            value=face.get(key)
            face[key]=float(value) if value not in ('',None) else None
    return result


def freeze_automatic_anchors(faces, vectors, settings=SETTINGS):
    groups=defaultdict(list)
    for face in faces:
        if face['group_id'] and not face.get('quality_reason') and face['face_id'] in vectors and face['status']!='false_detection':
            groups[face['group_id']].append(face)
    refs, selections, selection_audit = {}, {}, []
    for gid,members in sorted(groups.items()):
        chosen,audit=representatives(members,vectors,settings['representatives'],settings['diversity_weight'])
        refs[gid]=chosen;selections[gid]=audit
    for gid,members in sorted(groups.items()):
        ids=[f['face_id'] for f in sorted(members,key=lambda x:x['face_id'])]
        photos=[f['photo_id'] for f in members]
        pair_scores=[float(np.clip(vectors[a]@vectors[b],-1,1)) for i,a in enumerate(ids) for b in ids[i+1:]]
        own_mean=float(np.mean(pair_scores)) if pair_scores else None
        median=float(np.median(pair_scores)) if pair_scores else None
        p10=float(np.quantile(pair_scores,.1)) if pair_scores else None
        competitors=[]
        for other,other_members in sorted(groups.items()):
            if other==gid:continue
            score=float(np.mean([float(np.clip(vectors[a]@vectors[b['face_id']],-1,1))
                                 for a in ids for b in other_members]))
            competitors.append(dict(group_id=other,score=score))
        competitors.sort(key=lambda x:(-x['score'],x['group_id']))
        gap=own_mean-competitors[0]['score'] if own_mean is not None and competitors else None
        q=float(np.median([quality(f) for f in members]))
        reasons=[]
        if len(members)<settings['anchor_min_faces']:reasons.append('too_few_faces')
        if len(set(photos))<settings['anchor_min_photos']:reasons.append('too_few_distinct_photos')
        if len(photos)!=len(set(photos)):reasons.append('same_photo_review')
        if median is None or median<settings['anchor_median_similarity']:reasons.append('median_similarity_below_anchor_threshold')
        if p10 is None or p10<settings['anchor_p10_similarity']:reasons.append('lower_tail_similarity_below_anchor_threshold')
        if gap is not None and gap<settings['anchor_mean_competitor_gap']:reasons.append('anchor_competition_gap_insufficient')
        if q<settings['anchor_median_quality']:reasons.append('median_quality_below_anchor_threshold')
        selection_audit.append(dict(group_id=gid,face_ids=ids,selected=not reasons,reasons=reasons,
                                     face_count=len(ids),distinct_photos=len(set(photos)),within_mean=own_mean,
                                     within_median=median,within_p10=p10,competitor_gap=gap,
                                     median_quality=q,top_competitor=competitors[0] if competitors else None,
                                     representative_selection=selections[gid]))
    anchor_ids={row['group_id'] for row in selection_audit if row['selected']}
    return dict(groups=dict(groups),references=refs,anchors=anchor_ids,audit=selection_audit)


def rank_candidates(members, snapshot, vectors, excluded_group=None, settings=SETTINGS):
    candidates=[]
    for gid,refs in sorted(snapshot['references'].items()):
        if gid==excluded_group:continue
        evidence=[]
        for face in members:
            by_photo={}
            for ref in refs:
                if ref['face_id']==face['face_id'] or ref['photo_id']==face['photo_id']:continue
                cosine=float(np.clip(vectors[face['face_id']]@vectors[ref['face_id']],-1,1))
                item=dict(face_id=ref['face_id'],photo_id=ref['photo_id'],cosine=cosine)
                old=by_photo.get(ref['photo_id'])
                if old is None or (-cosine,ref['face_id'])<(-old['cosine'],old['face_id']):by_photo[ref['photo_id']]=item
            best=sorted(by_photo.values(),key=lambda x:(-x['cosine'],x['face_id']))[:2]
            evidence.append(dict(face_id=face['face_id'],score=float(np.mean([r['cosine'] for r in best])) if best else None,
                                 support=sum(r['cosine']>=settings['support_threshold'] for r in best),references=best))
        measured=[row['score'] for row in evidence if row['score'] is not None]
        score=float(np.mean(measured)) if len(measured)==len(members) and measured else None
        same_photo=bool({f['photo_id'] for f in members}&{f['photo_id'] for f in snapshot['groups'][gid]})
        candidates.append(dict(group_id=gid,is_anchor=gid in snapshot['anchors'],score=score,
                               minimum_member_score=min(measured) if len(measured)==len(members) and measured else None,
                               minimum_support=min(row['support'] for row in evidence),same_photo_review=same_photo,
                               evidence=evidence))
    candidates.sort(key=lambda row:(row['score'] is None,-row['score'] if row['score'] is not None else 0,row['group_id']))
    return candidates


def decide(members, candidates, singleton=False, settings=SETTINGS):
    valid=[candidate for candidate in candidates if candidate['score'] is not None]
    top=valid[0] if valid else None
    gap=top['score']-valid[1]['score'] if len(valid)>1 else None
    minimum=settings['singleton_score_threshold'] if singleton else settings['score_threshold']
    margin=settings['singleton_margin'] if singleton else settings['margin']
    if not top:reason='no_measured_candidate'
    elif not top['is_anchor']:reason='top_candidate_not_automatic_anchor'
    elif top['minimum_support']<settings['min_support']:reason='insufficient_distinct_photo_support'
    elif top['score']<minimum or top['minimum_member_score']<minimum:reason='score_below_threshold'
    elif gap is not None and gap<margin:reason='competing_group_margin_insufficient'
    elif top['same_photo_review']:reason='same_photo_review'
    else:reason='assigned_to_frozen_automatic_anchor'
    return dict(face_ids=[f['face_id'] for f in members],reason=reason,
                target_group=top['group_id'] if reason=='assigned_to_frozen_automatic_anchor' else None,
                score_threshold=minimum,margin_threshold=margin,score_gap=gap,
                top1_group_id=top['group_id'] if top else None,top1_score=top['score'] if top else None,
                top2_group_id=valid[1]['group_id'] if len(valid)>1 else None,
                top2_score=valid[1]['score'] if len(valid)>1 else None,candidates=candidates)


def automatic_stage(faces, vectors, snapshot, stage, settings=SETTINGS):
    if stage not in {'unknown_reassignment','small_group_linking'}:raise ValueError('Unknown stage')
    result=deepcopy(faces);by_id={f['face_id']:f for f in result};queries=[]
    if stage=='unknown_reassignment':
        queries=[[f] for f in faces if f['status']=='unknown' and not f.get('quality_reason') and f['face_id'] in vectors]
    else:
        queries=[members for gid,members in sorted(snapshot['groups'].items())
                 if gid not in snapshot['anchors'] and len(members)<=settings['max_small_group']]
    decisions=[]
    for members in sorted(queries,key=lambda xs:tuple(sorted(f['face_id'] for f in xs))):
        own=members[0]['group_id'] if stage=='small_group_linking' else None
        candidates=rank_candidates(members,snapshot,vectors,own,settings)
        decision=decide(members,candidates,singleton=stage=='small_group_linking' and len(members)==1,settings=settings)
        decision['source_group']=own
        decisions.append(decision)
        if not decision['target_group']:continue
        for face in members:
            target=by_id[face['face_id']];before={k:target[k] for k in ('group_id','status','reason')}
            target.update(group_id=decision['target_group'],status='grouped',reason='',
                          label='인물'+decision['target_group'][6:],decision_state='assigned')
            target.setdefault('diagnostic',{}).setdefault('stages',[]).append(dict(stage=stage,
                     state='assigned',reason=decision['reason'],before=before,group_id=decision['target_group']))
    mark_same_photo(result)
    return result,decisions


def _counts(faces):
    sizes=Counter(f['group_id'] for f in faces if f['group_id'])
    return dict(detected_faces=len(faces),grouping_eligible_faces=sum(not f.get('quality_reason') and f['status']!='false_detection' for f in faces),
                groups=len(sizes),singleton_groups=sum(n==1 for n in sizes.values()),
                unknown_faces=sum(f['status']=='unknown' for f in faces),
                quality_excluded_faces=sum(bool(f.get('quality_reason')) and f['status']!='false_detection' for f in faces),
                manual_false_excluded=sum(f['status']=='false_detection' for f in faces))


def run_auto_comparison(args):
    source=Path(args.source).resolve();truth_path=Path(args.ground_truth).resolve()
    cache=json.loads((source/'cache.json').read_text(encoding='utf-8'))
    with (source/'faces.csv').open(encoding='utf-8-sig',newline='') as file:raw=typed_faces(list(csv.DictReader(file)))
    if {f['face_id'] for f in raw}!={f['face_id'] for f in cache['faces']} or len(raw)!=len(cache['faces']):raise ValueError('Source face mapping mismatch')
    vectors=load_vectors(source,cache)
    source_sha=sha256(source/'faces.csv')
    summary=json.loads((source/'summary.json').read_text(encoding='utf-8'))
    if summary['actual_grouping_config']!=dict(linkage='average',similarity_threshold=.4,min_group_size=1,ambiguity_margin=.01,reassign_unknown=False):
        raise ValueError('Source must use the fixed average .40 baseline')
    # Make raw predictions BEFORE loading human identity annotations.
    raw_snapshot=freeze_automatic_anchors(raw,vectors)
    t=time.perf_counter();raw_unknown,raw_unknown_decisions=automatic_stage(raw,vectors,raw_snapshot,'unknown_reassignment');raw_unknown_time=time.perf_counter()-t
    t=time.perf_counter();raw_small,raw_small_decisions=automatic_stage(raw,vectors,raw_snapshot,'small_group_linking');raw_small_time=time.perf_counter()-t
    annotations=read_annotations(truth_path,cache['faces'],cache)
    # This lane is explicitly user-assisted: exclusions are human supplied, never identity IDs.
    excluded={fid for fid,a in annotations.items() if a['include']=='0'}
    clean=deepcopy(raw)
    for face in clean:
        if face['face_id'] in excluded:
            face.update(group_id='',status='false_detection',label='오검출',reason='user_confirmed_false_detection',review='')
    clean_snapshot=freeze_automatic_anchors(clean,vectors)
    clean_unknown,clean_unknown_decisions=automatic_stage(clean,vectors,clean_snapshot,'unknown_reassignment')
    clean_small,clean_small_decisions=automatic_stage(clean,vectors,clean_snapshot,'small_group_linking')
    lanes=[('fully_automatic',source,raw,vectors,raw_snapshot,annotations,
            [(raw_unknown,raw_unknown_decisions,raw_unknown_time),(raw_small,raw_small_decisions,raw_small_time)]),
           ('human_false_exclusion_control',source,clean,vectors,clean_snapshot,annotations,
            [(clean_unknown,clean_unknown_decisions,None),(clean_small,clean_small_decisions,None)])]
    if getattr(args,'target_source',None):
        target=Path(args.target_source).resolve();tc=json.loads((target/'cache.json').read_text(encoding='utf-8'))
        with (target/'faces.csv').open(encoding='utf-8-sig',newline='') as file:target_faces=typed_faces(list(csv.DictReader(file)))
        if len(target_faces)!=len(tc['faces']) or {f['face_id'] for f in target_faces}!={f['face_id'] for f in tc['faces']}:
            raise ValueError('Target source face mapping mismatch')
        target_summary=json.loads((target/'summary.json').read_text(encoding='utf-8'))
        if target_summary['actual_grouping_config']!=summary['actual_grouping_config']:
            raise ValueError('Target source baseline grouping configuration differs')
        tv=load_vectors(target,tc);ts=freeze_automatic_anchors(target_faces,tv)
        tu,tud=automatic_stage(target_faces,tv,ts,'unknown_reassignment')
        tm,tmd=automatic_stage(target_faces,tv,ts,'small_group_linking')
        lanes.append(('ljm2_fully_automatic',target,target_faces,tv,ts,None,[(tu,tud,None),(tm,tmd,None)]))
    output=new_run(args.output,Path(cache['input_root']));rows=[];evaluations={};provenance=[]
    for lane,asset_source,baseline,lv,snapshot,truth,stages in lanes:
        source_cache=json.loads((asset_source/'cache.json').read_text(encoding='utf-8'))
        provenance.append(dict(lane=lane,source=str(asset_source),faces_sha256=sha256(asset_source/'faces.csv'),
                               features_sha256=source_cache['features_sha256']))
        lane_dir=output/lane;lane_dir.mkdir()
        audit=[]
        for item in snapshot['audit']:
            row=deepcopy(item)
            # Purity is measured ONLY AFTER anchor selection and both stages have finished.
            if truth:
                labels={truth[fid]['manual_person_id'] for fid in item['face_ids'] if truth[fid]['include']=='1' and truth[fid]['manual_person_id']}
                row.update(evaluation_identity_ids=sorted(labels),evaluation_mixed=len(labels)>1)
            audit.append(row)
        write_json(lane_dir/'anchor_selection.json',dict(settings=SETTINGS,groups=audit,
                   warning='selected is computed without identity labels; evaluation fields are post-hoc only'))
        variants=[('baseline',baseline,[],None),('unknown_reassignment',*stages[0]),('small_group_linking',*stages[1])]
        for name,predicted,decisions,seconds in variants:
            folder=lane_dir/name;folder.mkdir()
            fields=['face_id','photo_id','relative_path','group_id','label','status','reason','quality_reason','width','height','detection_score','blur_variance','review','crop_path']
            write_csv(folder/'faces.csv',predicted,fields)
            write_json(folder/'decisions.json',decisions)
            metric=pair_metrics(predicted,truth) if truth else None
            if metric:write_json(folder/'evaluation.json',metric)
            counts=_counts(predicted)
            changed=[f['face_id'] for f,b in zip(predicted,baseline) if (f['group_id'],f['status'])!=(b['group_id'],b['status'])]
            result=dict(lane=lane,variant=name,**counts,changed_faces=changed,changed_count=len(changed),
                        query_count=len(decisions),
                        accepted_queries=sum(d['target_group'] is not None for d in decisions),
                        held_reasons=dict(Counter(d['reason'] for d in decisions if not d['target_group'])),
                        evaluation=metric,compute_seconds=seconds,link=f'{lane}/{name}/index.html')
            write_json(folder/'summary.json',result);rows.append(result)
            _render_faces(folder,asset_source,predicted,baseline,decisions)
        selected=[row for row in audit if row['selected']]
        evaluations[lane]=dict(selected_anchor_groups=len(selected),selected_anchors=[r['group_id'] for r in selected],
                                evaluated_mixed_anchors=sum(r.get('evaluation_mixed',False) for r in selected) if truth else None,
                                selected_representatives=sum(len(snapshot['references'][r['group_id']]) for r in selected))
    report=dict(source=str(source),source_faces_sha256=source_sha,features_sha256=cache['features_sha256'],sources=provenance,
                truth_sha256=sha256(truth_path),settings=SETTINGS,anchor_evaluation=evaluations,results=rows,
                policy='Raw predictions do not receive identities or exclusions. Control lane uses explicit exclusions only. Fixed settings, no sweep or best selection. All stages use frozen snapshots independently.',
                score_definition='For each query face: mean of best two distinct-photo representative cosines (at most three refs per group); group query is mean over all members, all members must pass score and support. All groups compete, top group must be an automatic anchor.',
                limitation='Same previously inspected ljm collection. Raw metrics exclude human-labeled nonfaces only during evaluation. ljm2 has no target truth and receives no accuracy metrics. No change to source results or model.')
    write_json(output/'comparison.json',report)
    write_csv(output/'comparison.csv',[dict(lane=r['lane'],variant=r['variant'],groups=r['groups'],unknown=r['unknown_faces'],
                 changed=r['changed_count'],fp=r['evaluation']['false_positive_pairs'] if r['evaluation'] else '',
                 fn=r['evaluation']['false_negative_pairs'] if r['evaluation'] else '') for r in rows],
              ['lane','variant','groups','unknown','changed','fp','fn'])
    _render_comparison(output,report)
    if sha256(source/'faces.csv')!=source_sha:raise ValueError('Source changed during comparison')
    print(f'Automatic representative comparison: {output / "index.html"}')
    return output


def _render_faces(folder,source,faces,baseline,decisions):
    changes={f['face_id'] for f,b in zip(faces,baseline) if (f['group_id'],f['status'])!=(b['group_id'],b['status'])}
    by_query={fid:d for d in decisions for fid in d['face_ids']};groups=defaultdict(list)
    for face in faces:groups[face['group_id'] or ('false_detection' if face['status']=='false_detection' else 'unknown')].append(face)
    sections=[]
    for gid,members in sorted(groups.items()):
        cards=[]
        for face in members:
            fid=face['face_id'];d=by_query.get(fid);text=''
            if d:
                top=[c for c in d['candidates'] if c['score'] is not None][:2]
                scores=', '.join(f"{c['group_id']} {c['score']:.3f}/지지{c['minimum_support']}/{'대표그룹' if c['is_anchor'] else '비대표그룹'}" for c in top)
                text=f'<p>{html.escape(REASONS.get(d["reason"],d["reason"]))}</p><small>{html.escape(scores)}</small>'
            crop=os.path.relpath(source/'crops'/f'{fid}.jpg',folder);photo=os.path.relpath(source/'annotated'/f"{face['photo_id']}.jpg",folder)
            image=f'<img loading="lazy" src="{html.escape(crop,quote=True)}" alt="{fid}">' if face.get('crop_path') else '<p>정렬얼굴없음</p>'
            cards.append(f'<div class="card">{image}<small>{fid}</small><p>{"변경됨" if fid in changes else "기존 배정 유지"}</p>{text}<small>{html.escape(face.get("reason", ""))}</small><a href="{html.escape(photo,quote=True)}">원본 위치(기준선 표시)</a></div>')
        sections.append(f'<h2>{html.escape(gid)} ({len(members)})</h2><div class="gallery">{"".join(cards)}</div>')
    page='<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>자동대표 비교 결과</title><style>body{font:15px sans-serif;line-height:1.6;margin:24px}.gallery{display:flex;flex-wrap:wrap;gap:10px}.card{width:235px;padding:10px;border:1px solid #ccc}img{width:112px;height:112px;object-fit:contain}small{display:block;overflow-wrap:anywhere}</style><h1>자동 대표 단계 결과</h1><p>원본 위치 링크의 그룹 표시는 기준선의 표시입니다. 현재 배정은 이 페이지와 faces.csv를 확인하세요.</p><p><a href="../../index.html">비교표</a> · <a href="summary.json">요약</a> · <a href="decisions.json">판정·모든후보</a></p>'+''.join(sections)+'</html>'
    (folder/'index.html').write_text(page,encoding='utf-8')


def _render_comparison(output,report):
    names={'baseline':'기준선','unknown_reassignment':'미확인 재배정만','small_group_linking':'단독·2개 그룹 연결만'}
    lanes={'fully_automatic':'ljm 원본 완전 자동','human_false_exclusion_control':'ljm 사용자 오검출 제외 대조','ljm2_fully_automatic':'ljm2 원본 완전 자동'}
    rows=[]
    for row in report['results']:
        m=row['evaluation'];fp=m['false_positive_pairs'] if m else '정답 없음';fn=m['false_negative_pairs'] if m else '정답 없음'
        f1=f"{100*m['f1']:.2f}%" if m and m['f1'] is not None else '미계산'
        rows.append('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in [lanes[row['lane']],names[row['variant']],row['detected_faces'],row['groups'],row['singleton_groups'],row['unknown_faces'],row['quality_excluded_faces'],row['changed_count'],fp,fn,f1])+f'<td><a href="{row["link"]}">얼굴 결과</a></td></tr>')
    audits=[]
    for lane,a in report['anchor_evaluation'].items():
        audits.append(f'<li>{lanes[lane]}: 자동 대표 그룹{a["selected_anchor_groups"]}개, 대표얼굴{a["selected_representatives"]}개, 정답상 혼합 대표그룹 {a["evaluated_mixed_anchors"] if a["evaluated_mixed_anchors"] is not None else "미검증"}개. <a href="{lane}/anchor_selection.json">선정·탈락 근거</a></li>')
    reasons=[]
    for row in report['results']:
        if row['variant']=='baseline':continue
        readable='; '.join(f'{REASONS.get(reason,reason)} {count}件' for reason,count in row['held_reasons'].items()).replace('件','개')
        if not readable:readable='추가 비교 대상 없음(품질 제외 얼굴은 대상에 포함하지 않음)' if row.get('query_count',0)==0 else '보류 없음'
        reasons.append(f'<li>{lanes[row["lane"]]} / {names[row["variant"]]}: {html.escape(readable)}</li>')
    conclusion = ('<p><strong>이번 고정 조건에서는 추가 배정·연결된 얼굴이 없었습니다.</strong> '
                  '원래 그룹의 혼합·분리 오류가 유지되어 성능 향상은 확인하지 못했습니다. '
                  '자동 대표를 골랐다는 것과 분리 문제를 해결했다는 것은 다릅니다.</p>'
                  if all(row['changed_count']==0 for row in report['results']) else
                  '<p>변경된 얼굴과 혼합·분리 지표를 함께 확인해야 합니다. 변경 수만으로 성능 향상을 판단하지 않습니다.</p>')
    page=f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>자동 대표 그룹화 비교</title>
<style>body{{font:15px sans-serif;line-height:1.8;margin:24px}}table{{border-collapse:collapse}}td,th{{border:1px solid #ccc;padding:7px}}th{{background:#edf4f8}}</style>
<h1>수동 인물 ID 없는 자동 대표 그룹화 비교</h1><p>최초 average-linkage 0.40·모호성0.01·최소그룹1은 고정했습니다. 미확인 배정과 작은그룹 연결은 기준선의 서로 다른 사본에서 독립 실행하며 두 효과를 이어 적용하지 않았습니다.</p>
{conclusion}
<p>원본 완전 자동은 사람의 인물ID·오검출 목록을 예측에 쓰지 않았습니다. 정답은 예측이 끝난 뒤 평가에만 사용했습니다. 오검출 제외 대조는 사용자 정보3개를 활용한 별도 조건입니다. 원본 평가에서도 비인물 정답은 쌍평가에서 제외되므로 검출 정확도를 의미하지 않습니다.</p>
<table><tr><th>데이터/조건</th><th>단계</th><th>검출</th><th>그룹</th><th>단독</th><th>미확인</th><th>품질제외</th><th>변경얼굴</th><th>혼합쌍</th><th>분리쌍</th><th>F1</th><th>결과</th></tr>{''.join(rows)}</table>
<h2>自動 대표 선정</h2><p>서로 다른사진3장/얼굴3개 이상, 같은사진 그룹중복 검토없음, 내부 유사도 중앙값0.50·하위10%0.30 이상, 내부평균−가장가까운다른그룹평균0.05 이상, 품질 중앙값0.60 이상을 모두 요구합니다. 정답상 정확한 그룹만 미리 고르지 않았습니다. 선정 후에만 정답으로 혼합 여부를 검사했습니다.</p><ul>{''.join(audits)}</ul>
<h2>연결 판정</h2><p>그룹당 대표최대3개는 품질·다양성으로 고정합니다. 모든 그룹을 경쟁후보로 비교하고, 1순위가 자동 대표그룹일 때만 연결할 수 있습니다. 비대표그룹이1순위이면 다른 대표그룹으로 강제 승격시키지 않습니다.</p>
<p>한 얼굴의 점수는 서로다른사진 대표 중 가장가까운2개 유사도 평균입니다. 대표2개 모두0.40 이상, 얼굴평균0.45 이상, 경쟁차이0.03을 요구합니다. 단독그룹은 점수0.50·차이0.05를 요구합니다. 2개얼굴 그룹은 전원 조건통과 시에만 함께 연결합니다. 같은사진 충돌은 검토보류이며, 기존그룹에 절대금지 규칙으로 적용하지 않습니다. 새로배정된 얼굴은 대표에 추가하지 않습니다.</p>
<p>후보0개/1개는 정상 처리하며 2순위점수·차이를 만들지 않습니다. 존재하는 단독후보 점수는 단일비교값으로 기록하되 지지2개 조건은 통과할 수 없습니다. 품질제외 얼굴은 자동 복귀시키지 않습니다.</p>
<h2>변경되지 않은 사유</h2><ul>{''.join(reasons)}</ul><p>이 설정은 고정된 첫 실험값이며 최적값으로 선정하지 않았습니다. ljm은 기존에 검토한 자료라 탐색 결과이고, ljm2에는 새 정답이 없어 정확도·F1을 계산하지 않았습니다. 그룹수·미확인 감소만으로 성능 향상을 결론내리지 않습니다.</p>
<p><a href="comparison.json">설정과 평가 전체</a> · <a href="comparison.csv">요약 CSV</a></p></html>'''.replace('自動','자동')
    (output/'index.html').write_text(page,encoding='utf-8')
