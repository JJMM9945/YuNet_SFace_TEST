"""역할: 분석 결과를 검색·그룹 탐색·확대·얼굴 비교가 가능한 단일 로컬 HTML로 구성."""
from collections import Counter
from html import escape
import json
from pathlib import Path
import re


def render_html(run,photos,faces,options=None):
    run=Path(run)
    for f in faces:
        if not re.fullmatch(r'[A-Za-z0-9_-]+',f['face_id']) or not re.fullmatch(r'[A-Za-z0-9_-]+',f['photo_id']):
            raise ValueError('Invalid generated report face/photo ID')
    data_faces=[];cards=[]
    for f in faces:
        fid,pid=f['face_id'],f['photo_id']
        crop=f'crops/{fid}.jpg' if f.get('crop_path') and (run/'crops'/f'{fid}.jpg').exists() else ''
        geometry=f'geometry/{fid}.jpg' if (run/'geometry'/f'{fid}.jpg').exists() else ''
        record={k:f.get(k,'') for k in ['face_id','photo_id','relative_path','group_id','label','status','reason','review','quality_reason','manual_person_id','label_origin','detection_score','blur_variance','width','height','correction_note']}
        record.update(crop=crop,geometry=geometry,photo=f'photos/{pid}.jpg' if (run/'photos'/f'{pid}.jpg').exists() else '',
                      annotated=f'annotated/{pid}.jpg' if (run/'annotated'/f'{pid}.jpg').exists() else '',diagnostic=f.get('diagnostic',{}))
        data_faces.append(record)
        image=f'<img src="{crop}" alt="{fid} 정렬 얼굴" loading="lazy">' if crop else '<span class="no-image">정렬 이미지 없음</span>'
        badge='오검출' if f.get('status') == 'false_detection' else '미확인' if not f.get('group_id') else f['label']
        provenance='사용자 확인 오검출' if f.get('status') == 'false_detection' else '수동 확인 '+f['manual_person_id'] if f.get('manual_person_id') else '사용자 관계 확인' if f.get('label_origin','').startswith('user_') else '자동 추정' if f.get('group_id') else '배정 보류'
        cards.append(f'<figure id="{fid}"><button class="face-open" data-face="{fid}" aria-label="{escape(fid)} 확대">{image}</button><figcaption><div class="card-line"><span class="badge {"warning" if not f.get("group_id") else ""}">{escape(badge)}</span><label class="compare-check"><input type="checkbox" data-compare="{fid}">비교</label></div><strong>{escape(f["relative_path"])}</strong><span class="face-id">{fid}</span><span class="origin">{escape(provenance)}</span><span class="reason" data-reason="{fid}"></span><a class="photo-link" href="#{fid}" data-face="{fid}">얼굴과 원본 보기 ↗</a></figcaption></figure>')
    data_photos=[]
    for p in photos:
        pid=p['photo_id']
        if not re.fullmatch(r'[A-Za-z0-9_-]+',pid):raise ValueError('Invalid report photo ID')
        data_photos.append(dict(p,photo=f'photos/{pid}.jpg' if (run/'photos'/f'{pid}.jpg').exists() else '',
            annotated=f'annotated/{pid}.jpg' if (run/'annotated'/f'{pid}.jpg').exists() else ''))
    counts=Counter(f['group_id'] for f in faces if f.get('group_id'))
    payload=dict(version=2,run_name=run.name,photos=data_photos,faces=data_faces,options=options or {},
                 counts=dict(photos=len(photos),faces=len(faces),groups=len(counts),unknown=sum(not f.get('group_id') and f.get('status') != 'false_detection' for f in faces), false_detections=sum(f.get('status') == 'false_detection' for f in faces)))
    safe=json.dumps(payload,ensure_ascii=False,allow_nan=False).replace('&','\\u0026').replace('<','\\u003c').replace('>','\\u003e')
    template=Path(__file__).with_name('results_ui.html').read_text(encoding='utf-8')
    result=template.replace('__RESULT_DATA__',safe).replace('__FACE_CARDS__','\n'.join(cards))
    result=result.replace('__RUN_NAME__',escape(run.name))
    (run/'index.html').write_text(result,encoding='utf-8')
