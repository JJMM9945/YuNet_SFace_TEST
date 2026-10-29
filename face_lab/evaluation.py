"""Pairwise evaluation using explicit identities, independent of temporary group IDs."""
import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from .pipeline import write_json


def choose2(n):
    return n * (n - 1) // 2


def read_split(path, photos):
    """Photo-level disjoint split; blank/unassigned is excluded from dev and eval."""
    with Path(path).open(encoding='utf-8-sig', newline='') as file:
        reader = csv.DictReader(file)
        if not {'photo_id', 'split'} <= set(reader.fieldnames or []):
            raise ValueError('Split manifest needs photo_id,split')
        rows = list(reader)
    expected = {p['photo_id'] for p in photos}
    result = {}
    for row in rows:
        pid, split = row['photo_id'], row['split'].strip()
        if pid not in expected or pid in result or split not in {'', 'unassigned', 'dev', 'eval'}:
            raise ValueError(f'Invalid/duplicate photo split: {pid}')
        result[pid] = split or 'unassigned'
    if set(result) != expected:
        raise ValueError('Split manifest must list every source photo exactly once')
    return result


def split_hash(splits):
    return hashlib.sha256(json.dumps(splits, sort_keys=True).encode()).hexdigest()


def read_annotations(path, faces, cache=None):
    from .review import read_truth, context_hash
    annotations = read_truth(path, {f['face_id'] for f in faces}, context_hash(cache) if cache else None)
    with Path(path).open(encoding='utf-8-sig', newline='') as file:
        rows = list(csv.DictReader(file))
    for row in rows:
        status = row.get('annotation_status', '').strip()
        if status not in {'', 'labeled', 'uncertain', 'excluded', 'unlabeled'}:
            raise ValueError(f"Invalid annotation_status: {row['face_id']}")
        a = annotations[row['face_id']]
        if status == 'labeled' and (not a['manual_person_id'] or a['include'] != '1'):
            raise ValueError('labeled needs manual_person_id and include=1')
        if status in {'uncertain','excluded','unlabeled'} and a['manual_person_id']:
            raise ValueError('Clear manual_person_id for uncertain/excluded/unlabeled rows')
        a['annotation_status'] = status or ('excluded' if a['include'] == '0' else 'labeled' if a['manual_person_id'] else 'unlabeled')
    return annotations


def pair_metrics(faces, annotations):
    selected = []
    for f in faces:
        a = annotations.get(f['face_id'], {})
        if a.get('include') == '1' and a.get('manual_person_id') and a.get('annotation_status', 'labeled') == 'labeled':
            pred = ('group', f['group_id']) if f['group_id'] else ('unknown', f['face_id'])
            selected.append((f['face_id'], a['manual_person_id'], pred))
    if not selected:
        raise ValueError('No filled, included ground truth; metrics are not computed.')
    truth = Counter(t for _,t,_ in selected)
    predicted = Counter(p for _,_,p in selected)
    joint = Counter((t,p) for _,t,p in selected)
    tp = sum(choose2(n) for n in joint.values())
    predicted_pairs, true_pairs = sum(map(choose2, predicted.values())), sum(map(choose2, truth.values()))
    fp, fn = predicted_pairs-tp, true_pairs-tp
    mixed = []
    for p in sorted(predicted):
        ids = sorted({t for _,t,q in selected if q == p})
        if len(ids) > 1:
            mixed.append(dict(group_id=p[1], actual_person_ids=ids,
                              faces=[dict(face_id=fid, actual_person_id=t) for fid,t,q in selected if q == p]))
    split = []
    for t in sorted(truth):
        ps = sorted({p for _,q,p in selected if q == t})
        if len(ps) > 1:
            split.append(dict(actual_person_id=t, predictions=[dict(kind=p[0], id=p[1]) for p in ps],
                              faces=[dict(face_id=fid, kind=p[0], predicted_id=p[1]) for fid,q,p in selected if q == t]))
    n = len(faces)
    return dict(evaluated_faces=len(selected), total_detected_faces=n, omitted_faces=n-len(selected),
                ground_truth_coverage=len(selected)/n if n else None,
                uncertain_faces=sum(annotations.get(f['face_id'],{}).get('annotation_status') == 'uncertain' for f in faces),
                annotation_status_counts=dict(Counter(annotations.get(f['face_id'],{}).get('annotation_status','unlabeled') for f in faces)),
                automatic_assignment_rate=sum(bool(f['group_id']) for f in faces)/n if n else None,
                evaluated_assignment_rate=sum(p[0] == 'group' for _,_,p in selected)/len(selected),
                evaluated_unknown_faces=sum(p[0] == 'unknown' for _,_,p in selected),
                true_positive_pairs=tp, false_positive_pairs=fp, false_negative_pairs=fn,
                precision=tp/predicted_pairs if predicted_pairs else None,
                recall=tp/true_pairs if true_pairs else None,
                f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
                mixed_groups=mixed, split_people=split,
                unknown_policy='Each unknown is a distinct singleton; included same-person unknown pairs are false negatives. Quality exclusions remain in the labeled evaluation scope.',
                undefined_policy='zero denominator is null',
                limitation='Only annotated detected faces; missed detections and unlabeled pairs are outside evaluation.')


def evaluate(args):
    run = Path(args.run)
    summary_path = run/'summary.json'
    summary = json.loads(summary_path.read_text(encoding='utf-8')) if summary_path.exists() else {}
    if summary.get('mode') in {'apply-review','correct-groups'}:
        raise ValueError('This run has human-corrected labels. Evaluate the original automatic run to measure model/grouping performance.')
    with (run/'faces.csv').open(encoding='utf-8-sig', newline='') as file:
        faces = list(csv.DictReader(file))
    cache = json.loads((run/'cache.json').read_text(encoding='utf-8')) if (run/'cache.json').exists() else None
    annotations = read_annotations(args.ground_truth, cache['faces'] if cache else faces, cache)
    subset = getattr(args, 'subset', 'all')
    experiment_path = run/'experiment.json'
    experiment = json.loads(experiment_path.read_text()) if experiment_path.exists() else {}
    if experiment.get('subset', 'all') != 'all' and subset != experiment['subset']:
        raise ValueError('Use the same evaluation subset as the experiment')
    if subset != 'all':
        if experiment.get('subset') != subset:
            raise ValueError('First group only the requested subset using compare; filtering groups built with both dev/eval leaks cross-set membership')
        if not getattr(args, 'split_manifest', None) or not cache:
            raise ValueError('dev/eval requires --split-manifest and extraction cache')
        splits = read_split(args.split_manifest, cache['photos'])
        if experiment.get('split_sha256') and split_hash(splits) != experiment['split_sha256']:
            raise ValueError('Evaluation split differs from grouping experiment')
        faces = [f for f in faces if splits[f['photo_id']] == subset]
    result = pair_metrics(faces, annotations)
    result['subset'] = subset
    result['ground_truth_sha256'] = hashlib.sha256(Path(args.ground_truth).read_bytes()).hexdigest()
    output = Path(args.output) if args.output else run/'evaluations'
    output.mkdir(parents=True, exist_ok=True)
    target = output/('evaluation_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')+'.json')
    write_json(target, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f'Evaluation: {target}')
    return result
