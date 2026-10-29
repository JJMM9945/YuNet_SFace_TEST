"""Human pseudonymous face annotations. Never infer corrections from guessed identities."""
import csv
import hashlib
import json
import re
import shutil
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from .pipeline import extraction_signature, inventory, new_run, write_csv, write_json
from .models import MODELS, sha256, verify

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


def validate_cache(source, root, models):
    cache=json.loads((source/'cache.json').read_text(encoding='utf-8'))
    if cache.get('schema')!=1 or cache['signature']!=extraction_signature(cache['config']):
        raise ValueError('Cache incompatible: model/preprocessing/runtime changed; analyze again.')
    for spec in MODELS.values():
        verify(Path(models)/spec['name'],spec)
    if inventory(root)!=cache['input_files']:
        raise ValueError('Cache input mismatch: photo content/path/list changed; analyze again.')
    if sha256(source/'features.npz')!=cache['features_sha256']:
        raise ValueError('Feature cache checksum mismatch.')
    return cache


def assign_review(faces, annotations):
    manuals=sorted({r['manual_person_id'] for r in annotations.values() if r['include']=='1' and r['manual_person_id']})
    numbering={manual:i+1 for i,manual in enumerate(manuals)}
    for f in faces:
        row=annotations.get(f['face_id'],{})
        manual=row.get('manual_person_id','') if row.get('include')=='1' else ''
        f.update(group_id='',label='미확인',status='unknown',review='',manual_person_id=manual,label_origin='unreviewed',
                 reason='manual_review_missing')
        if row.get('include')=='0':
            f.update(reason='manual_review_excluded',label_origin='excluded')
        elif manual:
            n=numbering[manual]
            f.update(group_id=f'person{n:03d}',label=f'인물{n:03d}',status='grouped',reason='',label_origin='manual')
            if f.get('quality_reason'):
                f['review']='manual_override_quality'
    counts=Counter((f['photo_id'],f['group_id']) for f in faces if f['group_id'])
    for f in faces:
        if f['group_id'] and counts[(f['photo_id'],f['group_id'])]>1:
            f['review']=';'.join(filter(None,[f['review'],'same_photo_same_group']))


def apply_review(args):
    source,root=Path(args.source).resolve(),Path(args.input).resolve()
    start=time.perf_counter()
    cache=validate_cache(source,root,args.models)
    annotations=read_truth(args.ground_truth,{f['face_id'] for f in cache['faces']},context_hash(cache),True)
    if not any(r['include']=='1' and r['manual_person_id'] for r in annotations.values()):
        raise ValueError('No included manual IDs. No corrected result was generated.')
    run=new_run(args.output,root)
    for name in ('photos','crops'):
        shutil.copytree(source/name,run/name,dirs_exist_ok=True)
    shutil.copy2(source/'features.npz',run/'features.npz')
    # Preserve extraction-only cache so future regroup remains an automatic experiment.
    write_json(run/'cache.json',cache)
    shutil.copy2(args.ground_truth,run/'applied_ground_truth.csv')
    write_json(run/'manual_review.json',dict(source=str(source),ground_truth_sha256=sha256(Path(args.ground_truth)),
        context_sha256=context_hash(cache),policy='only explicit reviewed IDs; unreviewed/excluded stay unknown',
        note='Human correction, not automatic model improvement or retraining.'))
    from .pipeline import finish
    return finish(run,cache['photos'],cache['faces'],{},args,start,0,0,cache,'apply-review',annotations)


def make_review(args):
    run=Path(args.run).resolve()
    cache=json.loads((run/'cache.json').read_text(encoding='utf-8'))
    with (run/'faces.csv').open(encoding='utf-8-sig',newline='') as f:
        faces=list(csv.DictReader(f))
    experiment_path = run/'experiment.json'
    expected = set(json.loads(experiment_path.read_text())['face_ids']) if experiment_path.exists() else {f['face_id'] for f in cache['faces']}
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
