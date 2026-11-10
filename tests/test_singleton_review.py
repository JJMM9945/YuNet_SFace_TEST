"""Singleton review exposes existing scores without assigning identities."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from tests.experiments.face_lab.representative_experiment import merge_groups, unknown_reassignment, SETTINGS
from tests.experiments.face_lab.singleton_review import attach_prior_labels, build_review_data, render_review


def make_fixture():
    angles=[0,25,28,10,75,80]
    gids=['person001','person002','person002','','person003','']
    faces=[]; vectors={}
    for i,(angle,gid) in enumerate(zip(angles,gids)):
        fid=f'p{i}_f001';theta=np.deg2rad(angle);vectors[fid]=np.array([np.cos(theta),np.sin(theta)],np.float32)
        status='grouped' if gid else 'unknown'
        faces.append(dict(face_id=fid,photo_id=f'p{i}',relative_path=f'{i}.jpg',group_id=gid,status=status,
            reason='' if gid else 'ambiguous',quality_reason='',review='',label=gid or '미확인',
            width=160,height=160,detection_score=.9,blur_variance=200,diagnostic=dict(stages=[])))
    faces[-1].update(status='false_detection',reason='false_detection')
    settings=dict(SETTINGS,score_threshold=.99)
    merge=merge_groups(copy.deepcopy(faces),vectors,settings)
    reassignment=unknown_reassignment(copy.deepcopy(faces),vectors,settings)
    return faces,vectors,merge,reassignment


class SingletonReviewTests(unittest.TestCase):
    def test_singletons_unknown_and_false_detections(self):
        faces,vectors,merge,reassignment=make_fixture();before=copy.deepcopy(faces)
        result=build_review_data(faces,faces,merge,reassignment,vectors)
        self.assertEqual((result['singleton_groups'],result['unknown_faces']),(2,1))
        self.assertEqual(faces,before)
        self.assertNotIn('p5_f001',[q['face_id'] for q in result['queries']])
        self.assertTrue(result['pairs'])
        for p in result['pairs']:
            self.assertNotEqual(p['query_face_id'],p['candidate_face_id'])
            expected=float(vectors[p['query_face_id']] @ vectors[p['candidate_face_id']])
            self.assertAlmostEqual(p['individual_cosine'],expected,places=5)

    def test_broken_representative_mapping_and_scores_rejected(self):
        faces,vectors,merge,reassignment=make_fixture()
        altered=copy.deepcopy(reassignment)
        altered['settings']['score_threshold']=.1
        with self.assertRaisesRegex(ValueError,'settings'):
            build_review_data(faces,faces,merge,altered,vectors)
        altered=copy.deepcopy(merge)
        first=next(d for d in altered['decisions'] if d['group_id']=='person001')
        first['candidates'][0]['similarities'][0][0]=-.99
        with self.assertRaisesRegex(ValueError,'scores'):
            build_review_data(faces,faces,altered,reassignment,vectors)

    def test_no_candidates_and_quality_exclusion(self):
        faces,vectors,_,_=make_fixture();faces=[faces[0],faces[3]]
        faces[1]['quality_reason']='blurry_face';faces[1]['reason']='blurry_face'
        merge=merge_groups(copy.deepcopy(faces),vectors)
        reassignment=unknown_reassignment(copy.deepcopy(faces),vectors)
        result=build_review_data(faces,faces,merge,reassignment,vectors)
        self.assertEqual(result['queries'][0]['candidates'],[])
        unknown=next(q for q in result['queries'] if q['kind']=='unknown')
        self.assertEqual(unknown['candidates'],[])
        self.assertIn('품질',unknown['review_reason'])

    def test_review_payload_escapes_untrusted_filename(self):
        faces,vectors,merge,reassignment=make_fixture()
        faces[0]['relative_path']='</script><script>alert(1)</script>.jpg'
        data=build_review_data(faces,faces,merge,reassignment,vectors)
        with tempfile.TemporaryDirectory() as folder:
            run=Path(folder);render_review(run,data)
            html=(run/'index.html').read_text(encoding='utf-8')
            self.assertNotIn('</script><script>alert(1)',html)
            payload=html.split('<script type="application/json" id="review-data">')[1].split('</script>')[0]
            self.assertEqual(json.loads(payload)['queries'][0]['relative_path'],faces[0]['relative_path'])

    def test_prior_confirmations_are_display_only_and_do_not_rank_candidates(self):
        faces,vectors,merge,reassignment=make_fixture()
        data=build_review_data(faces,faces,merge,reassignment,vectors)
        before=[[(c['group_id'],c['score']) for c in q['candidates']] for q in data['queries']]
        first=data['queries'][0];rep=first['candidates'][0]['representatives'][0]
        truth={fid:dict(manual_person_id='A01',include='1',annotation_status='labeled')
               for fid in (first['face_id'],rep['face_id'])}
        attach_prior_labels(data,truth)
        self.assertTrue(rep['previously_confirmed_same'])
        self.assertEqual(before,[[(c['group_id'],c['score']) for c in q['candidates']] for q in data['queries']])


if __name__=='__main__':unittest.main()
