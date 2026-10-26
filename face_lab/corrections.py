"""Apply user-confirmed same/different constraints without guessing unmentioned identities."""
import csv
import json
import shutil
import time
from pathlib import Path
from .review import context_hash, validate_cache
from .pipeline import new_run, write_json
from .models import sha256


def constraint_check(faces, feedback):
    by_id={f['face_id']:f for f in faces}
    def same(a,b):
        return bool(by_id[a]['group_id']) and by_id[a]['group_id']==by_id[b]['group_id']
    positives=set()
    # Include transitive positive constraints when several sets overlap.
    components=[]
    for members in feedback.get('same_person_sets',[]):
        combined=set(members)
        changed=True
        while changed:
            changed=False
            for old in components[:]:
                if old & combined:
                    combined |= old;components.remove(old);changed=True
        components.append(combined)
    from itertools import combinations
    for members in components:
        positives.update(combinations(sorted(members),2))
    negative={tuple(sorted(pair)) for pair in feedback.get('different_person_pairs',[])}
    split=[list(p) for p in sorted(positives) if not same(*p)]
    mixed=[list(p) for p in sorted(negative) if same(*p)]
    return dict(explicit_same_person_pairs=len(positives),split_pair_violations=len(split),
                explicit_different_person_pairs=len(negative),mixed_pair_violations=len(mixed),
                split_pairs=split,mixed_pairs=mixed,
                note='Only explicit user constraints; not whole-dataset accuracy or an independent evaluation.')


def apply_constraints(faces, feedback):
    by_id={f['face_id']:f for f in faces}
    parent={fid:fid for fid in by_id}
    def find(fid):
        while parent[fid]!=fid:
            parent[fid]=parent[parent[fid]]
            fid=parent[fid]
        return fid
    involved=set()
    for members in feedback.get('same_person_sets',[]):
        if len(set(members))<2:
            raise ValueError('Same-person set must contain at least two distinct face IDs.')
        if any(fid not in by_id for fid in members):
            raise ValueError('Unknown face ID in same-person set')
        involved.update(members)
        for fid in members[1:]:
            a,b=find(members[0]),find(fid)
            parent[max(a,b)]=min(a,b)
    pairs=feedback.get('different_person_pairs',[])
    for pair in pairs:
        if len(pair)!=2 or any(fid not in by_id for fid in pair):
            raise ValueError('Different-person pair needs two known face IDs.')
        if find(pair[0])==find(pair[1]):
            raise ValueError('Conflicting feedback: same-person and different-person constraints overlap.')
    # Keep source groups for unmentioned faces; never propagate identity to them.
    number=max((int(f['group_id'][6:]) for f in faces if f.get('group_id','').startswith('person')),default=0)
    for f in faces:
        if not f.get('label_origin'):
            f['label_origin']='automatic_unreviewed'
        f.setdefault('manual_person_id','')
    components={}
    for fid in sorted(involved):
        components.setdefault(find(fid),[]).append(fid)
    def assign(ids,origin,manual=''):
        nonlocal number
        number+=1
        for fid in ids:
            f=by_id[fid]
            f.update(group_id=f'person{number:03d}',label=f'인물{number:03d}',status='grouped',reason='',
                     label_origin=origin,manual_person_id=manual,review='manual_override_quality' if f.get('quality_reason') else '')
    # Distinct positive components have no explicit negative relationship. Do not
    # manufacture manual ground truth asserting they are different people.
    for n,members in enumerate(components.values(),1):
        assign(members,'user_same_person',f'confirmed_set_{n:03d}' if len(components)==1 else '')
    for a,b in pairs:
        if by_id[a]['group_id'] and by_id[a]['group_id']==by_id[b]['group_id']:
            # Detach both endpoints unless one belongs to an explicit positive set.
            # Other original group members keep their automatic, unreviewed label.
            for fid in (a,b):
                if fid not in involved:
                    assign([fid],'user_different_person')
    for a,b in pairs:
        if not by_id[a]['group_id'] or not by_id[b]['group_id']:
            for fid in (a,b):
                if not by_id[fid]['group_id']:
                    assign([fid],'user_different_person')
        if by_id[a]['group_id']==by_id[b]['group_id']:
            raise ValueError('Cannot satisfy different-person constraint')
    # Recompute same-photo collision flags after merges and splits.
    from collections import Counter
    counts=Counter((f['photo_id'],f['group_id']) for f in faces if f['group_id'])
    for f in faces:
        flags=[s for s in f.get('review','').split(';') if s and s!='same_photo_same_group']
        if f['group_id'] and counts[(f['photo_id'],f['group_id'])]>1:
            flags.append('same_photo_same_group')
        f['review']=';'.join(flags)


def correct_groups(args):
    source,root=Path(args.source).resolve(),Path(args.input).resolve()
    start=time.perf_counter()
    cache=validate_cache(source,root,args.models)
    feedback=json.loads(Path(args.corrections).read_text(encoding='utf-8'))
    if feedback.get('context_sha256')!=context_hash(cache):
        raise ValueError('Correction extraction mismatch')
    if feedback.get('source_faces_sha256')!=sha256(source/'faces.csv'):
        raise ValueError('Correction source grouping mismatch; use the exact source run.')
    if not feedback.get('same_person_sets') and not feedback.get('different_person_pairs'):
        raise ValueError('No user-confirmed constraints')
    with (source/'faces.csv').open(encoding='utf-8-sig',newline='') as file:
        predictions={f['face_id']:f for f in csv.DictReader(file)}
    if set(predictions)!={f['face_id'] for f in cache['faces']}:
        raise ValueError('Source prediction/cache face mapping mismatch')
    faces=json.loads(json.dumps(cache['faces']))
    for face in faces:
        pred=predictions[face['face_id']]
        for k in ('group_id','label','status','reason','review','manual_person_id','label_origin'):
            if k in pred:
                face[k]=pred[k]
    before=constraint_check(faces,feedback)
    apply_constraints(faces,feedback)
    after=constraint_check(faces,feedback)
    run=new_run(args.output,root)
    for name in ('photos','crops'):
        shutil.copytree(source/name,run/name,dirs_exist_ok=True)
    shutil.copy2(source/'features.npz',run/'features.npz')
    write_json(run/'cache.json',cache)
    write_json(run/'applied_constraints.json',feedback)
    write_json(run/'constraint_checks.json',dict(before=before,after=after))
    write_json(run/'manual_review.json',dict(source=str(source),feedback_sha256=sha256(Path(args.corrections)),
        policy='only explicitly named faces merged/split; unmentioned predictions preserved, unreviewed',
        note='Human feedback correction, not automatic improvement or model retraining.'))
    from .pipeline import finish
    return finish(run,cache['photos'],faces,{},args,start,0,0,cache,'correct-groups')
