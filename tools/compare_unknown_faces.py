"""Offline side-by-side diagnostics for unknown faces; never assigns identities."""
import argparse
import csv
from datetime import datetime,timezone
from html import escape
import json
from pathlib import Path
import re
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image,ImageDraw,ImageOps
from face_lab import GROUPING_VERSION
from face_lab.grouping import group_faces
from face_lab.models import sha256
from face_lab.pipeline import extraction_signature,write_csv,write_json


def analyze(run):
    cache=json.loads((run/'cache.json').read_text(encoding='utf-8'))
    config=json.loads((run/'run_config.json').read_text(encoding='utf-8'))
    if config.get('grouping_version')!=GROUPING_VERSION:
        raise ValueError('Use a current automatic run; grouping rule mismatch or manually corrected result.')
    if cache['signature']!=extraction_signature(cache['config']):
        raise ValueError('Extraction runtime mismatch; use the environment that generated this run.')
    if sha256(run/'features.npz')!=cache['features_sha256']:
        raise ValueError('Feature checksum mismatch')
    with (run/'faces.csv').open(encoding='utf-8-sig',newline='') as file:
        faces=list(csv.DictReader(file))
    originals={f['face_id']:f for f in cache['faces']}
    if len(faces)!=len(originals) or {f['face_id'] for f in faces}!=set(originals):
        raise ValueError('Prediction/cache face mapping mismatch')
    for f in faces:
        if not re.fullmatch(r'[A-Za-z0-9_-]+',f['face_id']) or not re.fullmatch(r'[A-Za-z0-9_-]+',f['photo_id']):
            raise ValueError('Invalid generated ID')
    with np.load(run/'features.npz',allow_pickle=False) as data:
        vectors=dict(zip(data['face_ids'].tolist(),data['vectors']))
    options=config['options']
    working=json.loads(json.dumps(cache['faces']))
    trace={}
    group_faces(working,vectors,options['similarity_threshold'],options['min_group_size'],options['ambiguity_margin'],trace)
    predictions={f['face_id']:f for f in working}
    for f in faces:
        for k in ('group_id','status','reason'):
            if f[k]!=predictions[f['face_id']][k]:
                raise ValueError('Reconstructed grouping differs from stored predictions')
    by_id={f['face_id']:f for f in faces}
    records=[]
    for f in faces:
        if f['status']!='unknown':continue
        fid=f['face_id']; nearest=[]
        if fid in vectors:
            for other in faces:
                oid=other['face_id']
                if oid!=fid and oid in vectors:
                    nearest.append(dict(face_id=oid,similarity=float(np.clip(vectors[fid]@vectors[oid],-1,1)),
                                        label=other['label'],status=other['status']))
            nearest.sort(key=lambda x:(-x['similarity'],x['face_id']))
        records.append(dict(face_id=fid,photo_id=f['photo_id'],relative_path=f['relative_path'],reason=f['reason'],
            width=float(f['width']),height=float(f['height']),blur_variance=float(f['blur_variance']) if f['blur_variance'] else None,
            detection_score=float(f['detection_score']),trace=trace.get(fid),nearest_faces=nearest[:5]))
    return cache,options,by_id,originals,records,vectors


def roi(run,out,f):
    source=run/'photos'/(f['photo_id']+'.jpg')
    with Image.open(source) as image:
        image=image.convert('RGB')
    x,y,w,h=(float(f[k]) for k in ('x','y','width','height'))
    pad=max(w,h)*.35
    box=(max(0,int(x-pad)),max(0,int(y-pad)),min(image.width,int(x+w+pad)+1),min(image.height,int(y+h+pad)+1))
    draw=ImageDraw.Draw(image)
    draw.rectangle((x,y,x+w,y+h),outline='orange',width=max(2,int(max(w,h)/90)))
    points=f.get('landmarks',[])
    radius=max(2,int(max(w,h)/100))
    for i in range(0,len(points),2):
        px,py=points[i:i+2]
        draw.ellipse((px-radius,py-radius,px+radius,py+radius),fill='lime')
    crop=image.crop(box)
    crop=ImageOps.contain(crop,(360,300))
    name=f['face_id']+'_roi.jpg';crop.save(out/name,quality=92)
    return name


def render(run,out,options,by_id,originals,records,vectors):
    def tile(fid,query=None):
        f=by_id[fid]
        score=''
        if query and query in vectors and fid in vectors:
            score=f'<p>cosine {float(np.clip(vectors[query]@vectors[fid],-1,1)):.4f}</p>'
        image=f'<img src="../crops/{fid}.jpg" alt="{fid}">' if f['crop_path'] else '<p>정렬 얼굴 없음</p>'
        return f'<figure>{image}<figcaption>{fid}<br>{escape(f["label"])}{score}<a href="../annotated/{f["photo_id"]}.jpg">검출 확인</a></figcaption></figure>'
    html=['<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width">',
        '<title>미확인 얼굴 비교</title><style>body{font:16px system-ui,sans-serif;background:#f5f6f8;color:#232a31;margin:24px}section{background:white;border:1px solid #ccc;border-radius:12px;padding:20px;margin:24px 0}.row{display:flex;flex-wrap:wrap;gap:12px}figure{width:190px;margin:0;padding:10px;background:#f4f4f4;overflow-wrap:anywhere}figure img{width:112px;height:112px;object-fit:contain}figcaption{font-size:13px}.roi{max-width:360px;max-height:300px}td,th{padding:8px;border:1px solid #ccc}table{border-collapse:collapse}.warn{color:#9c4600}a{color:#145aaa}p{line-height:1.5}</style>',
        '<h1>미확인 얼굴 비교</h1><p><a href="../index.html">기존 결과</a> · <a href="diagnostics.json">판정 수치 JSON</a> · <a href="comparison.csv">비교 CSV</a></p>',
        '<p>원본 내용 사본의 얼굴 위치/랜드마크, 정렬 얼굴, 초기 소속 클러스터와 경쟁 클러스터를 비교합니다. 점수는 코사인 유사도이며 확률이 아닙니다. 실제 인물이 같은지 자동 확정하지 않습니다.</p>',
        f'<p>비교 임계값 {options["similarity_threshold"]:.2f}, margin {options["ambiguity_margin"]:.2f}, 최소 그룹 크기 {options["min_group_size"]}. 검출·특징 추출 재실행 없이 저장 벡터를 사용했습니다.</p>',
        '<p class="warn">경쟁 후보는 미확인 제외 전의 초기 클러스터입니다. 최종 인물 그룹과 일치하지 않을 수 있습니다. 가까운 얼굴 한 장의 점수는 실제 보류 규칙의 그룹 최저 점수와 다릅니다.</p>',
        '<p>위치 비교는 저장된 방향 보정 JPEG 사본에서 잘랐습니다. 녹색 점은 저장된 5개 랜드마크입니다. 특징 추출에 사용한 원본 픽셀을 다시 추론한 결과는 아닙니다.</p>']
    overview=Image.new('RGB',(3*580,((len(records)+2)//3)*218),'white') if records else Image.new('RGB',(580,100),'white')
    od=ImageDraw.Draw(overview)
    for number,record in enumerate(records,1):
        fid=record['face_id'];f=by_id[fid];trace=record['trace']
        region=roi(run,out,originals[fid])
        html.append(f'<section id="{fid}"><h2>{number}. {fid}</h2><p>{escape(f["relative_path"])}</p><p>사유: {escape(record["reason"])} · 박스 {record["width"]:.1f}×{record["height"]:.1f}px · blur 분산 {record["blur_variance"]} · 검출 점수 {record["detection_score"]:.4f}</p>')
        html.append(f'<div class="row"><div><h3>원본 위치와 랜드마크</h3><a href="../annotated/{f["photo_id"]}.jpg"><img class="roi" src="{region}" alt="face location"></a></div><div><h3>정렬 얼굴 112×112</h3>{tile(fid)}</div></div>')
        if trace:
            html.append(f'<h3>보류 조건 재현</h3><table><tr><th>자기 클러스터 최저</th><th>최강 경쟁 클러스터 최저</th><th>차이</th></tr><tr><td>{trace["own_min_similarity"]:.4f}</td><td>{trace["competing_min_similarity"]:.4f}</td><td>{trace["gap"]:.4f}</td></tr></table>')
            if trace['own_singleton']:
                html.append('<p>자기 클러스터는 단독 얼굴이므로 자기 값 1.0은 코드의 기본값입니다. 다른 얼굴과의 일치 근거가 아닙니다.</p>')
            gate=options['similarity_threshold']-options['ambiguity_margin']
            html.append(f'<p>경쟁값 ≥ {gate:.4f}, 차이 ≤ {options["ambiguity_margin"]:.4f}이면 모호성 보류. 이 얼굴의 조건 충족: {trace["ambiguity_rule_matched"]}.</p><h3>제외 전 자기 클러스터</h3><div class="row">')
            html.extend(tile(other,fid) for other in trace['own_face_ids'] if other!=fid)
            if trace['own_singleton']:html.append('<p>다른 구성원 없음</p>')
            html.append('</div>')
            for rank,competitor in enumerate(trace['competitors'][:3],1):
                html.append(f'<h3>경쟁 후보 {rank}: 최저 {competitor["min_similarity"]:.4f}, 최고 {competitor["max_similarity"]:.4f}</h3><div class="row">')
                html.extend(tile(other,fid) for other in competitor['face_ids'])
                html.append('</div>')
        else:
            html.append('<p class="warn">품질/추출 사유로 군집화 입력에서 제외됐습니다. 아래 유사도는 참고용이며, 보류 사유가 경쟁 그룹이라는 뜻이 아닙니다.</p>')
        html.append('<h3>가장 가까운 개별 얼굴 5개 — 참고용</h3><div class="row">')
        html.extend(tile(other['face_id'],fid) for other in record['nearest_faces'])
        html.append('</div></section>')
        # A compact overview shows the query and closest individual, without claiming identity.
        ox,oy=((number-1)%3)*580,((number-1)//3)*218
        od.text((ox+5,oy+5),f'{number}. {fid}',fill='black')
        od.text((ox+5,oy+24),record['reason'],fill='black')
        for col,target in enumerate([fid]+[n['face_id'] for n in record['nearest_faces'][:1]]):
            if by_id[target]['crop_path']:
                with Image.open(run/'crops'/(target+'.jpg')) as im:overview.paste(im.convert('RGB'),(ox+5+col*230,oy+48))
            od.text((ox+5+col*230,oy+164),target,fill='black')
            if col:od.text((ox+5+col*230,oy+184),f'nearest cosine={record["nearest_faces"][0]["similarity"]:.4f}',fill='black')
    html.append('</html>')
    (out/'index.html').write_text('\n'.join(html),encoding='utf-8')
    overview.save(out/'overview.jpg',quality=92)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',required=True)
    args=p.parse_args();run=Path(args.run).resolve()
    cache,options,by_id,originals,records,vectors=analyze(run)
    out=run/datetime.now(timezone.utc).strftime('comparison_%Y%m%dT%H%M%S_%fZ');out.mkdir()
    render(run,out,options,by_id,originals,records,vectors)
    write_json(out/'diagnostics.json',dict(source=str(run),grouping_version=GROUPING_VERSION,options=options,faces=records,
        note='Numerical and visual comparison only; no ground truth, identity assignment, or model retraining.'))
    rows=[]
    for r in records:
        t=r['trace'] or {};near=r['nearest_faces'][0] if r['nearest_faces'] else {}
        rows.append(dict(face_id=r['face_id'],reason=r['reason'],width=r['width'],height=r['height'],blur_variance=r['blur_variance'],
            own_min=t.get('own_min_similarity'),competing_min=t.get('competing_min_similarity'),gap=t.get('gap'),
            nearest_face_id=near.get('face_id',''),nearest_label=near.get('label',''),nearest_similarity=near.get('similarity')))
    write_csv(out/'comparison.csv',rows,list(rows[0]) if rows else ['face_id','reason'])
    print('Comparison:',out/'index.html')
    for r in records:
        t=r['trace'] or {};n=r['nearest_faces'][0] if r['nearest_faces'] else {}
        print(json.dumps(dict(face_id=r['face_id'],reason=r['reason'],own=t.get('own_min_similarity'),other=t.get('competing_min_similarity'),gap=t.get('gap'),nearest=n),ensure_ascii=False))


if __name__=='__main__':main()
