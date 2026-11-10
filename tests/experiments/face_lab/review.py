# 역할: 추출 출처 해시·수동 정답 검증·오프라인 정답 입력 화면 생성
"""Human pseudonymous face annotations. Never infer corrections from guessed identities."""
import csv
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path


ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}\Z")


def context_hash(cache):
    value = dict(signature=cache['signature'], input_files=cache['input_files'],
                 faces=[{k:f[k] for k in ('face_id','photo_id','x','y','width','height')} for f in cache['faces']])
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode('utf-8')).hexdigest()


def read_truth(path, face_ids, context=None, require_context=False):
    with Path(path).open(encoding='utf-8-sig',newline='') as file:
        reader=csv.DictReader(file)
        if not {'face_id','manual_person_id','include'} <= set(reader.fieldnames or []):
            raise ValueError('Ground truth needs face_id, manual_person_id, include columns.')
        has_context='_context_sha256' in reader.fieldnames
        if require_context and not has_context:
            raise ValueError('Review provenance missing. Export CSV from the review HTML for this extraction.')
        rows=list(reader)
    result={}
    for r in rows:
        fid=r['face_id']
        if fid not in face_ids or fid in result:
            raise ValueError(f'Unknown or duplicate ground-truth face_id: {fid}')
        if has_context and context is not None and r.get('_context_sha256')!=context:
            raise ValueError('Ground-truth extraction mismatch. Use the exact run used by the review screen.')
        include=r['include'].strip()
        if include not in {'0','1'}:
            raise ValueError(f'include must be 0 or 1: {fid}')
        manual=r['manual_person_id'].strip()
        if require_context and manual and not ID_PATTERN.fullmatch(manual):
            raise ValueError('Use pseudonymous IDs of 1-64 ASCII letters/digits/_/-, starting with a letter or digit.')
        status = r.get('annotation_status', '').strip()
        if status not in {'','labeled','uncertain','excluded','unlabeled'}:
            raise ValueError(f'Invalid annotation_status: {fid}')
        if status in {'uncertain','excluded','unlabeled'} and manual:
            raise ValueError('Clear manual_person_id for uncertain/excluded/unlabeled rows')
        if status == 'labeled' and (not manual or include != '1'):
            raise ValueError('labeled needs manual_person_id and include=1')
        result[fid]=dict(manual_person_id=manual,include=include,notes=r.get('notes',''),
                         annotation_status=status or ('excluded' if include == '0' else 'labeled' if manual else 'unlabeled'))
    return result


def make_review(args):
    run=Path(args.run).resolve()
    cache=json.loads((run/'cache.json').read_text(encoding='utf-8'))
    with (run/'faces.csv').open(encoding='utf-8-sig',newline='') as f:
        faces=list(csv.DictReader(f))
    expected = {f['face_id'] for f in cache['faces']}
    if {f['face_id'] for f in faces}!=expected or len(faces)!=len(expected):
        raise ValueError('faces.csv and cache have different face mappings.')
    for f in faces:
        if not re.fullmatch(r'[A-Za-z0-9_-]+',f['face_id']) or not re.fullmatch(r'[A-Za-z0-9_-]+',f['photo_id']):
            raise ValueError('Invalid generated face/photo ID')
        # Use generated names only; never interpolate original filenames in URLs.
        f['crop_path']=f"crops/{f['face_id']}.jpg" if f.get('crop_path') else ''
    template=(Path(__file__).with_name('review_ui.html')).read_text(encoding='utf-8')
    payload=json.dumps(dict(context=context_hash(cache),source=str(run),faces=faces),ensure_ascii=False)
    payload=payload.replace('&','\\u0026').replace('<','\\u003c').replace('>','\\u003e')
    html=template.replace('__REVIEW_PAYLOAD__',payload)
    target=run/('review_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')+'.html')
    target.write_text(html,encoding='utf-8')
    print(f'Review: {target}\nNo IDs prefilled from automatic group numbers. Export CSV after manual annotation.')
    return target
