"""Apply only explicitly named face-ID corrections to an immutable automatic run."""
import csv
import json
import re
import shutil
import time
from collections import Counter
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

from .models import sha256
from .pipeline import finish, load_vectors, new_run, write_csv, write_json
from .review import context_hash

GROUP_ID = re.compile(r'person[0-9]{3,}\Z')


def create_template(args):
    source = Path(args.source).resolve()
    cache = json.loads((source/'cache.json').read_text(encoding='utf-8'))
    summary = json.loads((source/'summary.json').read_text(encoding='utf-8'))
    if summary.get('mode') not in {'analyze', 'regroup'}:
        raise ValueError('Template requires an original automatic analyze/regroup run')
    load_vectors(source, cache)
    result = dict(schema=1, context_sha256=context_hash(cache), source_faces_sha256=sha256(source/'faces.csv'),
                  features_sha256=cache['features_sha256'], assignments={}, false_detections={},
                  description='Only explicitly named face IDs are changed; all others stay automatic and unreviewed.')
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('x', encoding='utf-8') as file:
        json.dump(result, file, ensure_ascii=False, indent=2, allow_nan=False)
        file.write('\n')
    print(f'Correction template: {target}')
    return target


def apply_corrections(faces, feedback):
    """Group IDs refer to the bound source run; omitted faces keep their prediction."""
    by_id = {f['face_id']: f for f in faces}
    if len(by_id) != len(faces):
        raise ValueError('Duplicate source face_id')
    assignments = feedback.get('assignments', {})
    false_detections = feedback.get('false_detections', {})
    if not isinstance(assignments, dict) or not isinstance(false_detections, dict):
        raise ValueError('assignments and false_detections must be objects keyed by face_id')
    if not assignments and not false_detections:
        raise ValueError('No explicit corrections')
    if set(assignments) & set(false_detections):
        raise ValueError('A face cannot be assigned and excluded as a false detection')
    unknown = (set(assignments) | set(false_detections)) - set(by_id)
    if unknown:
        raise ValueError(f'Unknown face_id: {sorted(unknown)[0]}')
    if any(not isinstance(g, str) or not GROUP_ID.fullmatch(g) for g in assignments.values()):
        raise ValueError('Target group IDs must be person followed by at least three digits')
    if any(not isinstance(note, str) or not note.strip() for note in false_detections.values()):
        raise ValueError('False detections need a nonempty explanation')
    changes = []
    for f in faces:
        fid = f['face_id']
        before = {k: f.get(k, '') for k in ('group_id', 'label', 'status', 'reason', 'review')}
        f['label_origin'] = 'automatic_unreviewed'
        if fid in assignments:
            target = assignments[fid]
            f.update(group_id=target, label='인물'+target[6:], status='grouped', reason='',
                     label_origin='user_confirmed', decision_state='assigned', correction_note='사용자 확인 인물')
        elif fid in false_detections:
            f.update(group_id='', label='오검출', status='false_detection', reason='false_detection',
                     label_origin='user_confirmed_false_detection', decision_state='false_detection',
                     correction_note=false_detections[fid])
        if fid in assignments or fid in false_detections:
            after = {k: f.get(k, '') for k in ('group_id', 'label', 'status', 'reason', 'review')}
            f['diagnostic']['manual_correction'] = dict(before=before, after=after, note=f['correction_note'])
            changes.append(dict(face_id=fid, photo_id=f['photo_id'], relative_path=f['relative_path'],
                                before_group=before['group_id'], before_status=before['status'],
                                after_group=f['group_id'], after_status=f['status'], note=f['correction_note']))
    # Recalculate the review marker after explicit moves; preserve other flags.
    counts = Counter((f['photo_id'], f['group_id']) for f in faces if f['group_id'])
    for f in faces:
        flags = [v for v in f.get('review', '').split(';') if v and v != 'same_photo_same_group']
        if f['group_id'] and counts[(f['photo_id'], f['group_id'])] > 1:
            flags.append('same_photo_same_group')
        if f.get('quality_reason') and f['face_id'] in assignments:
            flags.append('manual_override_quality')
        f['review'] = ';'.join(dict.fromkeys(flags))
        if 'manual_correction' in f['diagnostic']:
            f['diagnostic']['manual_correction']['after']['review'] = f['review']
    return changes


def correct_groups(args):
    start = time.perf_counter()
    source = Path(args.source).resolve()
    feedback_path = Path(args.corrections).resolve()
    feedback = json.loads(feedback_path.read_text(encoding='utf-8'))
    cache = json.loads((source/'cache.json').read_text(encoding='utf-8'))
    summary = json.loads((source/'summary.json').read_text(encoding='utf-8'))
    if summary.get('mode') not in {'analyze', 'regroup'}:
        raise ValueError('Corrections require an original automatic analyze/regroup run')
    if feedback.get('schema') != 1 or feedback.get('context_sha256') != context_hash(cache):
        raise ValueError('Correction extraction context mismatch')
    if feedback.get('source_faces_sha256') != sha256(source/'faces.csv'):
        raise ValueError('Correction source faces.csv mismatch')
    if feedback.get('features_sha256') != cache['features_sha256']:
        raise ValueError('Correction feature provenance mismatch')
    vectors = load_vectors(source, cache)
    with (source/'faces.csv').open(encoding='utf-8-sig', newline='') as file:
        rows = list(csv.DictReader(file))
    source_rows = {r['face_id']: r for r in rows}
    if len(source_rows) != len(rows) or set(source_rows) != {f['face_id'] for f in cache['faces']}:
        raise ValueError('Source faces and cache mappings differ')
    diagnostics = json.loads((source/'diagnostics.json').read_text(encoding='utf-8'))
    if set(diagnostics) != set(source_rows):
        raise ValueError('Source diagnostics and face mappings differ')
    faces = deepcopy(cache['faces'])
    for f in faces:
        r = source_rows[f['face_id']]
        if r['photo_id'] != f['photo_id'] or r['relative_path'] != f['relative_path']:
            raise ValueError('Source face/photo mapping differs from extraction cache')
        for key in ('group_id','label','status','reason','review','decision_state'):
            f[key] = r.get(key, '')
        f['diagnostic'] = diagnostics[f['face_id']]
    changes = apply_corrections(faces, feedback)
    # Validate everything before creating any output folder.
    for folder in ('photos', 'crops'):
        if not (source/folder).is_dir():
            raise ValueError(f'Missing source image folder: {folder}')
    run = new_run(args.output, Path(cache['input_root']))
    for folder in ('photos','crops'):
        shutil.copytree(source/folder, run/folder, dirs_exist_ok=True)
    shutil.copy2(source/'features.npz', run/'features.npz')
    write_json(run/'cache.json', cache)
    shutil.copy2(feedback_path, run/'corrections.json')
    write_csv(run/'correction_changes.csv', changes,
              ['face_id','photo_id','relative_path','before_group','before_status','after_group','after_status','note'])
    truth = []
    context = context_hash(cache)
    for f in faces:
        fid = f['face_id']
        manual = feedback.get('assignments', {}).get(fid, '')
        excluded = fid in feedback.get('false_detections', {})
        truth.append(dict(face_id=fid, photo_id=f['photo_id'], relative_path=f['relative_path'],
                          manual_person_id=manual, include='0' if excluded else '1',
                          annotation_status='excluded' if excluded else 'labeled' if manual else 'unlabeled',
                          review_type='false_detection' if excluded else '',
                          notes=feedback.get('false_detections', {}).get(fid, ''),
                          _context_sha256=context))
    write_csv(run/'confirmed_ground_truth.csv', truth,
              ['face_id','photo_id','relative_path','manual_person_id','include','annotation_status',
               'review_type','notes','_context_sha256'])
    write_json(run/'correction_provenance.json', dict(source=str(source), context_sha256=context_hash(cache),
               source_faces_sha256=feedback['source_faces_sha256'], features_sha256=cache['features_sha256'],
               corrections_sha256=sha256(feedback_path), assignment_count=len(feedback.get('assignments', {})),
               false_detection_count=len(feedback.get('false_detections', {})),
               policy='Only explicitly listed faces changed; unlisted predictions remain automatic and unreviewed.'))
    source_config = json.loads((source/'run_config.json').read_text(encoding='utf-8'))['options']
    source_config.update(command='correct-groups', source=str(source), corrections=str(feedback_path),
                         output=str(Path(args.output)), input=cache['input_root'])
    return finish(run, deepcopy(cache['photos']), faces, vectors, SimpleNamespace(**source_config),
                  start, 0.0, 0.0, cache, 'correct-groups')
