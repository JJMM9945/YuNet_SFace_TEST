"""Local paired recognition comparison. Detection/eligibility are frozen, labels evaluate only."""
import copy
import hashlib
import html
import json
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from .auraface import AuraFace, SPEC, align_face
from .evaluation import read_annotations, pair_metrics
from .grouping import group_faces, validate_vectors
from .models import load, sha256, MODELS
from .pipeline import new_run, inventory, load_vectors, write_json, write_csv, runtime
from .preprocess import decode_oriented, extract, save_bgr


def variant_label(name):
    model,scope=name.split('_',1)
    return ('SFace' if model=='sface' else 'AuraFace')+' — '+{
        'all_040':'전체 사진 · 0.40 고정', 'development_selected':'개발 사진 · 선택 기준',
        'evaluation_040':'평가 사진 · 0.40 고정', 'evaluation_selected':'평가 사진 · 개발에서 선택한 기준'}[scope]


def photo_fold(content_sha):
    # Exact duplicate photo contents stay together, irrespective of filename.
    value=int(hashlib.sha256(('auraface-photo-split-v1:'+content_sha).encode()).hexdigest()[:8],16)
    return 'development' if value%10<7 else 'evaluation'


def predict(faces,vectors,threshold):
    result=copy.deepcopy(faces)
    group_faces(result,vectors,threshold=threshold,min_group=1,margin=.01,method='average')
    return result


def select_threshold(rows,fp_budget=0):
    """Only DEVELOPMENT metrics enter this function; no evaluation metrics or labels."""
    safe=[r for r in rows if r['false_positive_pairs']<=fp_budget]
    if not safe:raise ValueError('No development threshold satisfies the declared false-positive budget')
    return max(safe,key=lambda r:(r['recall'] if r['recall'] is not None else -1,
                                 r['precision'] if r['precision'] is not None else -1,r['threshold']))['threshold']


def save_features(run,name,vectors,model,source_hash):
    ids=sorted(vectors);dimension=len(vectors[ids[0]])
    validate_vectors(vectors,dimension=dimension)
    path=run/(name+'_features.npz')
    np.savez_compressed(path,face_ids=np.array(ids),vectors=np.stack([vectors[fid] for fid in ids]))
    write_json(run/(name+'_features_manifest.json'),dict(schema='recognition-comparison-v1',model=model,
        dimension=dimension,sha256=sha256(path),source_cache_sha256=source_hash,face_ids=ids,
        preprocessing='See experiment.json; never interchange models or use the SFace-only regroup cache loader'))


def write_variant(run,name,faces,vectors,threshold,annotations):
    started=time.perf_counter();predicted=predict(faces,vectors,threshold)
    elapsed=time.perf_counter()-started
    folder=run/name;folder.mkdir()
    metrics=pair_metrics(predicted,annotations)
    counts=Counter(f['group_id'] for f in predicted if f['group_id'])
    summary=dict(name=name,threshold=threshold,linkage='average',distance_threshold=1-threshold,
        ambiguity_margin=.01,min_group_size=1,faces=len(faces),eligible_faces=sum(not f.get('quality_reason') for f in faces),
        groups=len(counts),singleton_groups=sum(n==1 for n in counts.values()),
        unknown_faces=sum(not f['group_id'] and not f.get('quality_reason') for f in predicted),
        quality_excluded_faces=sum(bool(f.get('quality_reason')) for f in predicted),
        grouping_seconds=elapsed,metrics=metrics)
    write_json(folder/'summary.json',summary)
    write_json(folder/'diagnostics.json',{f['face_id']:f['diagnostic'] for f in predicted})
    fields=['face_id','photo_id','relative_path','group_id','status','reason','quality_reason','width','height','detection_score','blur_variance']
    write_csv(folder/'faces.csv',predicted,fields)
    render_variant(folder,summary,predicted,annotations)
    return summary,predicted


def render_variant(folder,summary,predicted,annotations):
    name=summary['name'];threshold=summary['threshold'];metrics=summary['metrics']
    counts=Counter(f['group_id'] for f in predicted if f['group_id'])
    e=html.escape
    body=[f'<h1>{e(variant_label(name))}</h1><a href="../index.html">비교 요약</a><p>average-linkage {threshold:.2f}, 모호성 폭 0.01, 최소 그룹 1. 그룹 번호는 이 결과 안에서만 유효합니다.</p>',
          '<p><a href="summary.json">평가·혼합/분리 목록</a> · <a href="diagnostics.json">얼굴별 후보 점수·판정 이유</a> · <a href="faces.csv">얼굴 CSV</a></p>']
    body.append('<h2>다른 사람이 섞인 그룹</h2>')
    if not metrics['mixed_groups']:body.append('<p>평가 범위에서 발견되지 않았습니다.</p>')
    for group in metrics['mixed_groups']:
        body.append(f'<p><a href="#{group["group_id"]}">{group["group_id"]}</a>: '+e(', '.join(group['actual_person_ids']))+'</p>')
    body.append('<details><summary>같은 사람이 나뉜 경우</summary>')
    for person in metrics['split_people']:
        body.append('<p>'+e(person['actual_person_id'])+': '+e(', '.join(p['id'] for p in person['predictions']))+'</p>')
    body.append('</details>')
    gids=sorted(counts)+['']
    for gid in gids:
        members=[f for f in predicted if f['group_id']==gid]
        if not members:continue
        body.append(f'<h2 id="{e(gid)}">{e(gid or "미확인 / 품질 제외")} ({len(members)}장)</h2><div class="grid">')
        for f in members:
            fid=f['face_id'];d=f['diagnostic'];a=annotations.get(fid,{})
            candidates=' / '.join(f"{c['group_id']} {c['score']:.3f}" for c in d['candidates'][:2]) or '후보 없음'
            body.append(f'<article><a href="../annotated/{f["photo_id"]}.jpg"><img loading="lazy" src="../crops/{name.split("_")[0]}/{fid}.png"></a><div>{e(fid)}</div><div>정답: {e(a.get("manual_person_id", ""))}</div><div>{e(f["reason"] or "배정")}</div><small>{e(candidates)}</small></article>')
        body.append('</div>')
    (folder/'index.html').write_text(page(name,''.join(body)),encoding='utf-8')


def page(title,body):
    return '<!doctype html><html lang="ko"><meta charset="utf-8"><title>'+html.escape(title)+'</title><style>body{font:16px/1.6 system-ui;margin:30px;max-width:1400px}table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:8px}img{width:112px;height:112px;object-fit:contain;background:#eee}.grid{display:flex;flex-wrap:wrap;gap:12px}article{border:1px solid #ddd;padding:10px;width:230px;overflow-wrap:anywhere}small{font-size:12px}a{color:#075ab5}</style>'+body+'</html>'


def run_experiment(args):
    started=time.perf_counter();source=Path(args.source).resolve();root=Path(args.input).resolve()
    cache=json.loads((source/'cache.json').read_text(encoding='utf-8'))
    source_hash=sha256(source/'cache.json');truth_hash=sha256(Path(args.ground_truth))
    originals=inventory(root)
    expected={(p['relative_path'],p['sha256']) for p in cache['input_files']}
    if expected!={(p['relative_path'],p['sha256']) for p in originals} or any(p['read_error'] for p in originals):
        raise ValueError('Original photo inventory/hash differs from the frozen detection source')
    old=load_vectors(source,cache)
    annotations=read_annotations(args.ground_truth,cache['faces'],cache)
    # Explicit human non-face exclusions are fixed for BOTH models, not learned by either model.
    faces=copy.deepcopy(cache['faces'])
    for f in faces:
        if annotations.get(f['face_id'],{}).get('annotation_status')=='excluded':
            f['quality_reason']='human_confirmed_nonface'
    folds={p['photo_id']:photo_fold(p['sha256']) for p in cache['photos']}
    partitions={fold:[f for f in faces if folds[f['photo_id']]==fold] for fold in ('development','evaluation')}
    # Fail before inference if there is no usable ground truth in either partition.
    for part in partitions.values():pair_metrics([dict(f,group_id='') for f in part],annotations)
    aura=AuraFace(args.aura_models);_,sface=load(args.models)
    run=new_run(args.output,root)
    for model in ('sface','auraface'):(run/'crops'/model).mkdir()
    write_json(run/'photo_split.json',dict(rule='SHA256(auraface-photo-split-v1:+photo content SHA256), first 8 hex modulo 10; <7 development',
        photos=[dict(photo_id=p['photo_id'],relative_path=p['relative_path'],sha256=p['sha256'],split=folds[p['photo_id']]) for p in cache['photos']]))
    vectors={'sface':{},'auraface':{}};seconds={'sface':0.,'auraface':0.};alignment={};max_difference=0.;processed=0
    for p in cache['photos']:
        bgr=decode_oriented((root/p['relative_path']).read_bytes())
        if bgr.shape[:2]!=(p['height'],p['width']):raise ValueError('Original-coordinate image dimensions changed')
        annotated=bgr.copy()
        for f in [f for f in faces if f['photo_id']==p['photo_id']]:
            fid=f['face_id'];landmarks=np.array(f['landmarks'],np.float64).reshape(5,2)
            row=np.array([f['x'],f['y'],f['width'],f['height'],*f['landmarks'],f['detection_score']],np.float32)
            t=time.perf_counter();scrop,sv=extract(sface,bgr,row);seconds['sface']+=time.perf_counter()-t
            t=time.perf_counter();acrop,matrix=align_face(bgr,landmarks);av=aura.feature(acrop);seconds['auraface']+=time.perf_counter()-t
            max_difference=max(max_difference,float(np.max(np.abs(sv-old[fid]))))
            for model,crop,v in (('sface',scrop,sv),('auraface',acrop,av)):
                vectors[model][fid]=v;save_bgr(run/'crops'/model/(fid+'.png'),crop)
            alignment[fid]=dict(photo_id=f['photo_id'],landmarks=landmarks.tolist(),auraface_matrix=matrix.tolist(),
                               original_box=[f[k] for k in ('x','y','width','height')])
            x,y,w,h=[round(f[k]) for k in ('x','y','width','height')]
            cv2.rectangle(annotated,(x,y),(x+w,y+h),(0,255,0),3)
            for lx,ly in landmarks:cv2.circle(annotated,(round(lx),round(ly)),4,(0,0,255),-1)
            cv2.putText(annotated,fid[-4:],(x,max(20,y-8)),cv2.FONT_HERSHEY_SIMPLEX,.7,(0,255,0),2)
            processed+=1
            if processed%20==0:print(f'Extracted both models: {processed}/{len(faces)} faces',flush=True)
        scale=min(1,1600/max(annotated.shape[:2]))
        if scale<1:annotated=cv2.resize(annotated,None,fx=scale,fy=scale,interpolation=cv2.INTER_AREA)
        save_bgr(run/'annotated'/(p['photo_id']+'.jpg'),annotated)
    if max_difference>1e-5:raise ValueError(f'SFace replay differs from frozen source: {max_difference}; investigate before comparison')
    for model in vectors:save_features(run,model,vectors[model],SPEC if model=='auraface' else MODELS['sface'],source_hash)
    write_json(run/'alignment.json',alignment)
    grid=[round(t/100,2) for t in range(20,81,2)]+[1.0]
    variants=[];predictions={};scans={};selected={}
    for model in ('sface','auraface'):
        v=vectors[model]
        # Scan only development photos. Evaluation labels cannot affect this choice.
        scans[model]=[]
        for threshold in grid:
            metric=pair_metrics(predict(partitions['development'],v,threshold),annotations)
            scans[model].append(dict(threshold=threshold,**metric))
        selected[model]=select_threshold(scans[model],fp_budget=0)
        print(f'{model}: development selected threshold {selected[model]}',flush=True)
        for scope,subset,threshold in [('all_040',faces,.4),('development_selected',partitions['development'],selected[model]),
                                       ('evaluation_040',partitions['evaluation'],.4),('evaluation_selected',partitions['evaluation'],selected[model])]:
            name=model+'_'+scope
            summary,pred=write_variant(run,name,subset,v,threshold,annotations)
            variants.append(summary);predictions[name]=pred
    write_json(run/'development_threshold_scan.json',dict(fp_budget=0,grid=grid,selection='Maximum development recall among FP<=0; tie precision then higher threshold',results=scans))
    # Membership change uses sets of face IDs, never equality of temporary group numbers.
    def peer_map(pred):
        return {f['face_id']:sorted(g['face_id'] for g in pred if f['group_id'] and g['group_id']==f['group_id']) for f in pred}
    first=peer_map(predictions['sface_all_040']);second=peer_map(predictions['auraface_all_040'])
    changes=[dict(face_id=fid,sface_members=first[fid],auraface_members=second[fid]) for fid in sorted(first) if first[fid]!=second[fid]]
    write_json(run/'membership_changes.json',changes)
    metadata=dict(status='complete',input=str(root),photos=len(originals),detected_faces=len(faces),
        explicit_nonface_exclusions=[f['face_id'] for f in faces if f.get('quality_reason')=='human_confirmed_nonface'],
        fixed_eligibility='Original SFace quality decisions retained for both models; three human-confirmed nonfaces excluded equally',
        source=str(source),source_cache_sha256=source_hash,source_features_sha256=cache['features_sha256'],
        truth_path=str(Path(args.ground_truth).resolve()),truth_sha256=truth_hash,source_detection_settings=cache['config'],
        runtime=runtime(),auraface=aura.metadata,sface=MODELS['sface'],sface_replay_max_component_error=max_difference,
        selected_thresholds=selected,feature_seconds=seconds,feature_timing='alignment+feature, excludes image decode/file output/model load; CPU runtimes differ',
        total_seconds=time.perf_counter()-started,variants=variants,membership_changed_faces=len(changes),
        limitations=['Previously inspected photos: exploratory development/evaluation split, not an untouched independent test',
                    'Photo-level split can share people, events, near-duplicates; exact duplicates stay together',
                    '0.40 is the same numeric threshold, not the same false acceptance operating point',
                    'Detection is frozen; no missed-face detection improvement is measured',
                    'No Android latency/memory/accuracy measurement; desktop CPU only',
                    'Unknown faces remain distinct singleton predictions in pairwise evaluation; no manual correction of predicted groups'])
    if sha256(source/'cache.json')!=source_hash or sha256(Path(args.ground_truth))!=truth_hash or inventory(root)!=originals:
        raise ValueError('Source/truth/photo contents changed during experiment')
    write_json(run/'experiment.json',metadata)
    render_comparison(run,metadata,changes)
    print(f'Comparison complete: {run}/index.html',flush=True)
    return run


def render_comparison(run,metadata,changes):
    variants=metadata['variants'];seconds=metadata['feature_seconds']
    body=['<h1>AuraFace / SFace 비교 — ljm_dataset</h1>',
          '<p>YuNet 검출·랜드마크·얼굴 집합 고정, average-linkage, 모호성 폭 0.01, 최소 그룹 1. 수동 확인된 비얼굴은 두 모델 모두 제외합니다. 사진을 클릭하면 검출 상자·랜드마크를 볼 수 있습니다.</p>',
          '<p><b>전체 사진 0.40 비교는 탐색 결과입니다.</b> 모델마다 점수 분포가 달라 같은 숫자가 같은 안전도를 뜻하지 않습니다. selected는 개발 사진에서 혼합 쌍 0을 만족하는 후보 중 재현율이 가장 높은 기준을 선택한 결과입니다. 평가 사진의 정답은 기준 선택에 사용하지 않았습니다. 이 사진들은 과거에 검토한 자료여서 독립 검증으로 간주할 수 없습니다.</p>',
          '<table><tr><th>결과</th><th>기준</th><th>얼굴/평가</th><th>그룹</th><th>미확인/제외</th><th>혼합 쌍 FP</th><th>분리 쌍 FN</th><th>Precision</th><th>Recall</th><th>F1</th></tr>']
    def pct(value):return '—' if value is None else f'{100*value:.2f}%'
    for r in variants:
        m=r['metrics'];body.append(f'<tr><td><a href="{r["name"]}/index.html">{variant_label(r["name"])}</a></td><td>{r["threshold"]:.2f}</td><td>{r["faces"]}/{m["evaluated_faces"]}</td><td>{r["groups"]}</td><td>{r["unknown_faces"]}/{r["quality_excluded_faces"]}</td><td>{m["false_positive_pairs"]}</td><td>{m["false_negative_pairs"]}</td><td>{pct(m["precision"])}</td><td>{pct(m["recall"])}</td><td>{pct(m["f1"])}</td></tr>')
    body.extend(['</table><p>FP는 다른 사람의 얼굴을 같은 그룹에 넣은 얼굴 쌍, FN은 같은 사람을 다른 그룹 또는 미확인으로 남긴 얼굴 쌍입니다. 인원 수가 아닙니다. 미확인끼리 같은 사람으로 계산하지 않습니다.</p>',
        f'<p>얼굴 정렬+특징 추출 합계: SFace {seconds["sface"]:.2f}초 / AuraFace {seconds["auraface"]:.2f}초. 런타임이 달라 모델 구조만의 속도 차이로 해석할 수 없습니다. AuraFace 모델 약 261MB, 512차원; Android 실기기 속도·메모리는 별도 측정해야 합니다.</p>',
        '<p><a href="experiment.json">실제 설정·시간·근거</a> · <a href="development_threshold_scan.json">개발 기준 비교 전체</a> · <a href="photo_split.json">사진별 개발/평가 구분</a> · <a href="membership_changes.json">얼굴 소속 변경</a> · <a href="alignment.json">원본 좌표·정렬 행렬</a></p>',
        f'<h2>0.40에서 소속이 바뀐 얼굴 ({len(changes)}개)</h2><div class="grid">'])
    for change in changes:
        fid=change['face_id'];body.append(f'<article>{html.escape(fid)}<br>SFace / AuraFace<br><img loading="lazy" src="crops/sface/{fid}.png"><img loading="lazy" src="crops/auraface/{fid}.png"><div>동료 얼굴 수 {len(change["sface_members"])} → {len(change["auraface_members"])}</div></article>')
    body.append('</div>')
    (run/'index.html').write_text(page('AuraFace 비교',''.join(body)),encoding='utf-8')
