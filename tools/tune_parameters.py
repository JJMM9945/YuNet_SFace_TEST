"""Local diagnostic sweep; detections are proposals for manual review, not ground truth."""
import argparse
from collections import Counter
from datetime import datetime, timezone
from html import escape
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps
from face_lab.models import load, sha256
from face_lab.pipeline import write_csv, write_json, inventory
from face_lab.preprocess import decode_oriented, detection_input, to_original
from face_lab.grouping import group_faces


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',required=True)
    p.add_argument('--input',default='ljm_dataset')
    p.add_argument('--models',default='models')
    p.add_argument('--output',default='results/parameter_checks')
    a=p.parse_args()
    cv2.setNumThreads(1); cv2.setRNGSeed(0); cv2.ocl.setUseOpenCL(False)
    source,root=Path(a.source),Path(a.input).resolve()
    cache=json.loads((source/'cache.json').read_text(encoding='utf-8'))
    if inventory(root)!=cache['input_files']:
        raise ValueError('Input differs from source run; analyze again first.')
    if sha256(source/'features.npz')!=cache['features_sha256']:
        raise ValueError('Feature cache checksum mismatch')
    out=Path(a.output)/datetime.now(timezone.utc).strftime('check_%Y%m%dT%H%M%S_%fZ')
    out.mkdir(parents=True)
    groups=[]
    with np.load(source/'features.npz',allow_pickle=False) as z:
        vectors=dict(zip(z['face_ids'].tolist(),z['vectors']))
    for threshold in (.35,.40,.45,.50):
        for margin in (.01,.03):
            faces=json.loads(json.dumps(cache['faces']))
            group_faces(faces,vectors,threshold,2,margin)
            reasons=Counter(f['reason'] for f in faces if f['status']=='unknown')
            groups.append(dict(similarity_threshold=threshold,ambiguity_margin=margin,min_group_size=2,
                groups=len({f['group_id'] for f in faces if f['group_id']}),
                unknown_faces=sum(f['status']=='unknown' for f in faces),
                review_faces=sum(bool(f['review']) for f in faces),reasons=dict(reasons)))
    write_json(out/'grouping_sweep.json',groups)
    configs=[(1600,.9),(1600,.8),(1600,.7),(2400,.9),(2400,.8),(2400,.7),(960,.8)]
    detector,_=load(a.models)
    rows=[]
    html=['<!doctype html><meta charset="utf-8"><title>Parameter review</title>',
          '<style>body{font:16px sans-serif;margin:24px}img{max-width:100%}td,th{padding:8px;border:1px solid #ccc}table{border-collapse:collapse}</style>',
          '<h1>얼굴 검출 설정 비교</h1>',
          '<p>Previously zero-face photos only. More boxes do not establish better accuracy. Check false detections manually.</p>']
    zero=[x for x in cache['photos'] if x['status']=='success' and x['detected_faces']==0]
    for number,photo in enumerate(zero,1):
        bgr=decode_oriented((root/photo['relative_path']).read_bytes())
        rgb=Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB))
        preview=ImageOps.contain(rgb,(480,360))
        sheet=Image.new('RGB',(480*4,400*2),'white')
        draw=ImageDraw.Draw(sheet)
        for index,(side,score) in enumerate(configs):
            small,sx,sy=detection_input(bgr,side)
            detector.setInputSize((small.shape[1],small.shape[0])); detector.setScoreThreshold(score)
            start=time.perf_counter(); _,found=detector.detect(small); elapsed=time.perf_counter()-start
            detections=[] if found is None else [to_original(f,sx,sy) for f in found]
            item=dict(photo_id=photo['photo_id'],relative_path=photo['relative_path'],max_side=side,
                score_threshold=score,detected_faces=len(detections),detection_seconds=elapsed,
                detections=[d.tolist() for d in detections])
            rows.append(item)
            cell=preview.copy(); cd=ImageDraw.Draw(cell)
            for d in detections:
                x,y,w,h=map(float,d[:4]); ax,ay=cell.width/rgb.width,cell.height/rgb.height
                cd.rectangle((x*ax,y*ay,(x+w)*ax,(y+h)*ay),outline='lime',width=2)
                cd.text((max(0,x*ax),max(0,y*ay-12)),f'{d[14]:.3f}',fill='red')
            ox,oy=(index%4)*480,(index//4)*400
            sheet.paste(cell,(ox,oy+25))
            draw.text((ox+5,oy+5),f'side={side}, score>={score}, boxes={len(detections)}',fill='black')
        name=photo['photo_id']+'.jpg'; sheet.save(out/name,quality=90)
        html.append(f'<h2>{escape(photo["relative_path"])}</h2><img src="{name}" alt="Detection comparison">')
        print(f'{number}/{len(zero)} completed',flush=True)
    write_json(out/'detection_sweep.json',rows)
    write_csv(out/'detection_sweep.csv',rows,['photo_id','relative_path','max_side','score_threshold','detected_faces','detection_seconds'])
    summary=[dict(max_side=side,score_threshold=score,
                 originally_zero_photos_with_boxes=sum(r['detected_faces']>0 for r in rows if r['max_side']==side and r['score_threshold']==score),
                 proposed_faces=sum(r['detected_faces'] for r in rows if r['max_side']==side and r['score_threshold']==score)) for side,score in configs]
    write_json(out/'summary.json',dict(source=str(source),input=str(root),detection=summary,grouping=groups,
        note='No manual ground truth: counts are not accuracy; no identity assessment performed.'))
    (out/'index.html').write_text('\n'.join(html),encoding='utf-8')
    print(json.dumps(dict(output=str(out),detection=summary,grouping=groups),ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
