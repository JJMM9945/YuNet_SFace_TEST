"""Compare thresholds against explicit user pair constraints, without inventing other labels."""
import argparse
import json
from pathlib import Path
import sys
from datetime import datetime,timezone
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from face_lab.review import validate_cache,context_hash
from face_lab.grouping import group_faces
from face_lab.corrections import constraint_check
from face_lab.models import sha256
from face_lab.pipeline import write_csv,write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',required=True);p.add_argument('--corrections',required=True)
    p.add_argument('--input',default='ljm_dataset');p.add_argument('--models',default='models')
    p.add_argument('--output',default='results/parameter_checks')
    a=p.parse_args();source=Path(a.source)
    cache=validate_cache(source,Path(a.input),a.models)
    feedback=json.loads(Path(a.corrections).read_text(encoding='utf-8'))
    if feedback.get('context_sha256')!=context_hash(cache) or feedback.get('source_faces_sha256')!=sha256(source/'faces.csv'):
        raise ValueError('Feedback source mismatch')
    with np.load(source/'features.npz',allow_pickle=False) as z:vectors=dict(zip(z['face_ids'].tolist(),z['vectors']))
    rows=[]
    for minimum in (1,2):
        for threshold in (.20,.25,.30,.35,.40,.45,.50,.55,.60):
            for margin in (.01,.03):
                faces=json.loads(json.dumps(cache['faces']))
                group_faces(faces,vectors,threshold,minimum,margin)
                check=constraint_check(faces,feedback)
                rows.append(dict(similarity_threshold=threshold,min_group_size=minimum,ambiguity_margin=margin,
                    same_person_pairs=check['explicit_same_person_pairs'],split_pairs=check['split_pair_violations'],
                    different_person_pairs=check['explicit_different_person_pairs'],mixed_pairs=check['mixed_pair_violations'],
                    unknown_faces=sum(f['status']=='unknown' for f in faces),
                    groups=len({f['group_id'] for f in faces if f['group_id']})))
    out=Path(a.output)/datetime.now(timezone.utc).strftime('feedback_%Y%m%dT%H%M%S_%fZ');out.mkdir(parents=True)
    write_csv(out/'feedback_sweep.csv',rows,list(rows[0]))
    write_json(out/'feedback_sweep.json',dict(source=str(source),feedback_sha256=sha256(Path(a.corrections)),results=rows,
        note='Only explicit user pairs; sparse feedback is not a representative evaluation. No automatic best setting selected.'))
    print('Feedback sweep:',out)
    for t in (.30,.40,.50,.55):
        print(next(r for r in rows if r['similarity_threshold']==t and r['min_group_size']==1 and r['ambiguity_margin']==.01))


if __name__=='__main__':main()
