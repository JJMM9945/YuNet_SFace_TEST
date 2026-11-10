"""Create a local, human-only pair review from already-computed representative scores."""
import csv
import json
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from .models import sha256
from .pipeline import load_vectors, new_run, write_csv, write_json
from .review import context_hash, read_truth

SAFE_ID = re.compile(r'[A-Za-z0-9_-]+\Z')


def read_rows(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as file:
        return list(csv.DictReader(file))


def attach_prior_labels(data, annotations):
    """Show previously confirmed IDs without using them to rank or select candidates."""
    def label(fid):
        a=annotations.get(fid,{})
        return a.get('manual_person_id','') if a.get('include')=='1' and a.get('annotation_status')=='labeled' else ''
    for q in data['queries']:
        q['manual_person_id']=label(q['face_id'])
        for candidate in q['candidates']:
            for rep in candidate['representatives']:
                rep['manual_person_id']=label(rep['face_id'])
                rep['previously_confirmed_same']=bool(q['manual_person_id'] and q['manual_person_id']==rep['manual_person_id'])
    for pair in data['pairs']:
        pair['query_manual_id']=label(pair['query_face_id'])
        pair['candidate_manual_id']=label(pair['candidate_face_id'])
        pair['previously_confirmed_same']=bool(pair['query_manual_id'] and pair['query_manual_id']==pair['candidate_manual_id'])
    data['previously_labeled_queries']=sum(bool(q['manual_person_id']) for q in data['queries'])


def build_review_data(faces, source_faces, merge, reassignment, vectors, top_candidates=3):
    """Review candidates are ranked existing cosine scores, not predicted identity."""
    if not 1 <= top_candidates <= 10:
        raise ValueError('top_candidates must be between 1 and 10')
    byid = {f['face_id']:f for f in faces}
    source = {f['face_id']:f for f in source_faces}
    if len(byid) != len(faces) or len(source) != len(source_faces) or set(byid) != set(source):
        raise ValueError('Face mappings differ or have duplicates')
    if any(not SAFE_ID.fullmatch(f['face_id']) or not SAFE_ID.fullmatch(f['photo_id']) for f in faces):
        raise ValueError('Invalid generated face/photo ID')
    if merge['settings'] != reassignment['settings'] or merge['representatives'] != reassignment['representatives']:
        raise ValueError('Representative comparison settings or snapshot differ')
    groups=defaultdict(list)
    for f in faces:
        if f['group_id'] and f['status'] == 'grouped': groups[f['group_id']].append(f)
    reps=merge['representatives']
    if set(groups) != set(reps):
        raise ValueError('Representative groups do not match the false-excluded result')
    for gid,rs in reps.items():
        members={f['face_id'] for f in groups[gid]}
        ids=[r['face_id'] for r in rs]
        if not ids or len(ids)!=len(set(ids)) or not set(ids)<=members:
            raise ValueError(f'Invalid representative IDs for {gid}')
        if any(fid not in vectors for fid in ids):raise ValueError('Missing representative vector')
    merge_decisions={d['group_id']:d for d in merge['decisions']}
    unknown_decisions={d['face_id']:d for d in reassignment['decisions']}
    if len(merge_decisions)!=len(merge['decisions']) or len(unknown_decisions)!=len(reassignment['decisions']):
        raise ValueError('Duplicate representative decisions')
    if set(merge_decisions)!=set(groups) or set(unknown_decisions)-set(byid):
        raise ValueError('Representative decision IDs do not match result faces/groups')
    queries=[];flat=[]
    todo=[('singleton',members[0]) for _,members in sorted(groups.items()) if len(members)==1]
    todo += [('unknown',f) for f in sorted(faces,key=lambda f:f['face_id']) if f['status']=='unknown']
    flagged=set(merge.get('flagged_groups', []))
    settings=merge['settings']
    for kind,f in todo:
        fid=f['face_id']; own_group=f['group_id']
        decision=merge_decisions[own_group] if kind=='singleton' else unknown_decisions.get(fid)
        if decision is None:
            candidates=[];hold='품질 제외 또는 특징값 부족으로 대표 자동 비교에서 제외됨'
        else:
            candidates=decision['candidates'];hold=decision['reason']
        checked=[]
        for candidate in candidates[:top_candidates]:
            gid=candidate['group_id']
            if gid not in groups or gid==own_group: raise ValueError('Invalid candidate group')
            rep_ids=[r['face_id'] for r in reps[gid]]
            if kind=='unknown':
                scored=candidate.get('representatives',[])
                if [r['face_id'] for r in scored] != rep_ids: raise ValueError('Candidate representative mapping differs')
                scores=[r['cosine'] for r in scored]
                support=candidate['support']
            else:
                matrix=candidate.get('similarities',[])
                if len(matrix)!=1 or len(matrix[0])!=len(rep_ids):raise ValueError('Singleton score matrix differs')
                scores=matrix[0];support=candidate['other_support']
            if fid not in vectors:raise ValueError('Candidate query is missing its vector')
            expected=[float(np.clip(vectors[fid] @ vectors[rid],-1,1)) for rid in rep_ids]
            if not np.allclose(scores,expected,atol=1e-5,rtol=0) or not np.isclose(candidate['score'],np.mean(expected),atol=1e-5,rtol=0):
                raise ValueError('Stored representative scores do not match feature vectors')
            rep_faces=[]
            for rid,score in zip(rep_ids,scores):
                target=byid[rid]
                item=dict(face_id=rid,photo_id=target['photo_id'],relative_path=target['relative_path'],
                          source_group=source[rid]['group_id'],similarity=float(score),
                          same_photo=target['photo_id']==f['photo_id'])
                rep_faces.append(item)
                flat.append(dict(query_type=kind,query_face_id=fid,query_photo_id=f['photo_id'],
                    query_group=own_group,candidate_group=gid,candidate_size=len(groups[gid]),
                    group_mean_score=float(candidate['score']),candidate_face_id=rid,
                    individual_cosine=float(score),same_photo=item['same_photo'],reason=hold))
            checked.append(dict(group_id=gid,member_count=len(groups[gid]),score=float(candidate['score']),
                                support=support,flagged=gid in flagged,representatives=rep_faces))
        queries.append(dict(kind=kind,face_id=fid,photo_id=f['photo_id'],relative_path=f['relative_path'],
                            group_id=own_group,source_group=source[fid]['group_id'],quality_reason=f.get('quality_reason',''),
                            automatic_reason=f.get('reason',''),review_reason=hold,candidates=checked))
    return dict(queries=queries,pairs=flat,settings=settings,top_candidates=top_candidates,
                singleton_groups=sum(q['kind']=='singleton' for q in queries),
                unknown_faces=sum(q['kind']=='unknown' for q in queries))


def create_singleton_review(args):
    comparison=Path(args.comparison).resolve()
    report=json.loads((comparison/'comparison.json').read_text(encoding='utf-8'))
    source=Path(args.source or report['source']).resolve()
    cache=json.loads((source/'cache.json').read_text(encoding='utf-8'))
    if (report['source_faces_sha256']!=sha256(source/'faces.csv') or
        report['features_sha256']!=cache['features_sha256']):
        raise ValueError('Comparison does not match the original automatic run')
    vectors=load_vectors(source,cache)
    base=comparison/'false_excluded'
    faces=read_rows(base/'faces.csv');source_faces=read_rows(source/'faces.csv')
    merge=json.loads((comparison/'representative_merge'/'representative_diagnostics.json').read_text(encoding='utf-8'))
    reassignment=json.loads((comparison/'representative_reassignment'/'representative_diagnostics.json').read_text(encoding='utf-8'))
    if merge['settings'] != report['settings'] or reassignment['settings'] != report['settings']:
        raise ValueError('Representative settings do not match comparison')
    data=build_review_data(faces,source_faces,merge,reassignment,vectors,args.top_candidates)
    # Validate each result face against source extraction geometry and photo mapping.
    extracted={f['face_id']:f for f in cache['faces']}
    if set(extracted)!={f['face_id'] for f in faces}:raise ValueError('Extraction face mapping differs')
    if any(f['photo_id']!=extracted[f['face_id']]['photo_id'] for f in faces):raise ValueError('Extraction photo mapping differs')
    images={q['face_id']:q['photo_id'] for q in data['queries']}
    for q in data['queries']:
        for candidate in q['candidates']:
            for rep in candidate['representatives']:images[rep['face_id']]=rep['photo_id']
    missing=[fid for fid,pid in images.items() if not (base/'crops'/f'{fid}.jpg').is_file() or not (base/'annotated'/f'{pid}.jpg').is_file()]
    if missing:raise ValueError(f'Missing review image for {missing[0]}')
    truth_path=Path(args.ground_truth).resolve() if getattr(args,'ground_truth',None) else None
    annotations=read_truth(truth_path,{f['face_id'] for f in faces},context_hash(cache),True) if truth_path else {}
    attach_prior_labels(data,annotations)
    provenance=dict(schema=1,source_context_sha256=context_hash(cache),source_faces_sha256=sha256(source/'faces.csv'),
        experiment_faces_sha256=sha256(base/'faces.csv'),features_sha256=cache['features_sha256'],
        comparison_sha256=sha256(comparison/'comparison.json'),ground_truth_sha256=sha256(truth_path) if truth_path else None,
        representative_settings=report['settings'])
    run=new_run(args.output,Path(cache['input_root']))
    for fid,pid in images.items():
        shutil.copy2(base/'crops'/f'{fid}.jpg',run/'crops'/f'{fid}.jpg')
        dst=run/'annotated'/f'{pid}.jpg'
        if not dst.exists():shutil.copy2(base/'annotated'/f'{pid}.jpg',dst)
    data['provenance']=provenance
    data['source_run']=str(source);data['comparison_run']=str(comparison)
    data['face_source_groups']={f['face_id']:f['group_id'] for f in source_faces}
    write_json(run/'review_manifest.json',data)
    fields=['query_type','query_face_id','query_photo_id','query_group','candidate_group','candidate_size',
            'group_mean_score','candidate_face_id','individual_cosine','same_photo','reason',
            'query_manual_id','candidate_manual_id','previously_confirmed_same']
    write_csv(run/'candidate_pairs.csv',data['pairs'],fields)
    render_review(run,data)
    print(f"Singleton review: {run}/index.html\nSingleton groups={data['singleton_groups']}, unknown faces={data['unknown_faces']}")
    return run


def render_review(run,data):
    run=Path(run)
    payload=json.dumps(data,ensure_ascii=False,allow_nan=False).replace('&','\\u0026').replace('<','\\u003c').replace('>','\\u003e')
    template=Path(__file__).with_name('singleton_review_ui.html').read_text(encoding='utf-8')
    js=Path(__file__).with_name('singleton_review_ui.js').read_text(encoding='utf-8')
    (run/'index.html').write_text(template.replace('__REVIEW_DATA__',payload).replace('__REVIEW_SCRIPT__',js),encoding='utf-8')
    reasons=Counter(q['review_reason'] for q in data['queries'])
    (run/'summary.md').write_text(f"# 단독 그룹 수동 검토\n\n- 단독 그룹: {data['singleton_groups']}개\n- 미확인 얼굴: {data['unknown_faces']}개\n- 이전에 ID를 확인한 검토 대상: {data.get('previously_labeled_queries',0)}개\n- 후보 그룹: 얼굴당 최대 {data['top_candidates']}개\n- 대표 최대: {data['settings']['representatives']}장\n- 기존 판정 사유: {dict(reasons)}\n\n"
        "그룹 평균은 후보 그룹 대표 전체에 대한 코사인 유사도의 평균입니다. 개별 유사도는 두 얼굴만의 점수이며 어느 것도 정답 확률이 아닙니다. "
        "같음·다름·판단 보류는 선택한 두 얼굴에만 적용됩니다. JSON 저장은 자동 배정이나 병합을 수행하지 않습니다. "
        "기존 점수를 보여 주는 수동 검토 화면이며 자동 그룹·임계값은 변경하지 않습니다.\n",encoding='utf-8')
