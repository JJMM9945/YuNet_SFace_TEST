import copy
import csv
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from face_lab.corrections import apply_constraints, constraint_check
from face_lab.review import assign_review, context_hash, make_review, read_truth
from face_lab.pipeline import write_csv


def face(fid,gid,photo=None):
    return dict(face_id=fid,photo_id=photo or fid,group_id=gid,label=gid,status='grouped' if gid else 'unknown',
                reason='',review='',quality_reason='',x=0,y=0,width=10,height=10,crop_path='',relative_path='sample.jpg')


class ReviewTests(unittest.TestCase):
    def test_named_faces_only_merge_and_split(self):
        faces=[face('a','person001'),face('b','person008'),face('unmentioned','person008'),
               face('c','person054'),face('d','person054')]
        feedback=dict(same_person_sets=[['a','b']],different_person_pairs=[['c','d']])
        before=constraint_check(faces,feedback)
        apply_constraints(faces,feedback)
        self.assertEqual(faces[0]['group_id'],faces[1]['group_id'])
        self.assertEqual(faces[2]['group_id'],'person008')
        self.assertNotEqual(faces[2]['group_id'],faces[1]['group_id'])
        self.assertNotEqual(faces[3]['group_id'],faces[4]['group_id'])
        self.assertEqual(before['split_pair_violations'],1)
        self.assertEqual(before['mixed_pair_violations'],1)
        self.assertEqual(constraint_check(faces,feedback)['split_pair_violations'],0)
        self.assertEqual(constraint_check(faces,feedback)['mixed_pair_violations'],0)
        self.assertEqual(faces[2]['label_origin'],'automatic_unreviewed')

    def test_conflicting_transitive_constraints_rejected(self):
        faces=[face(x,'person001') for x in 'abc']
        before=copy.deepcopy(faces)
        with self.assertRaisesRegex(ValueError,'Conflicting'):
            apply_constraints(faces,dict(same_person_sets=[['a','b'],['b','c']],different_person_pairs=[['a','c']]))
        self.assertEqual(faces,before)

    def test_unknown_constraint_ids_rejected(self):
        with self.assertRaisesRegex(ValueError,'Unknown'):
            apply_constraints([face('a','person001')],dict(same_person_sets=[['a','missing']]))

    def test_manual_ids_only_unreviewed_not_propagated(self):
        faces=[face('a','person001'),face('b','person002'),face('c','person001'),face('d','person002')]
        assign_review(faces,dict(a=dict(manual_person_id='A01',include='1'),b=dict(manual_person_id='A01',include='1'),
                               c=dict(manual_person_id='B01',include='1')))
        self.assertEqual(faces[0]['group_id'],faces[1]['group_id'])
        self.assertNotEqual(faces[0]['group_id'],faces[2]['group_id'])
        self.assertEqual(faces[3]['status'],'unknown')
        self.assertEqual(faces[3]['reason'],'manual_review_missing')

    def test_review_context_rejects_wrong_extraction(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'truth.csv'
            write_csv(path,[dict(face_id='a',manual_person_id='A01',include='1',_context_sha256='wrong')],
                      ['face_id','manual_person_id','include','_context_sha256'])
            with self.assertRaisesRegex(ValueError,'extraction mismatch'):
                read_truth(path,{'a'},'expected',True)

    def test_review_html_escapes_embedded_json(self):
        with tempfile.TemporaryDirectory() as d:
            run=Path(d);f=face('a','person001');f['relative_path']='</script><script>alert(1)</script>.jpg'
            cache=dict(signature={},input_files=[],faces=[f])
            (run/'cache.json').write_text(json.dumps(cache),encoding='utf-8')
            write_csv(run/'faces.csv',[f],list(f))
            result=make_review(Namespace(run=d))
            html=result.read_text(encoding='utf-8')
            self.assertNotIn('</script><script>alert(1)',html)
            self.assertIn('\\u003c/script\\u003e',html)
            payload=html.split('<script type="application/json" id="data">')[1].split('</script>')[0]
            self.assertEqual(json.loads(payload)['context'],context_hash(cache))


if __name__=='__main__':
    unittest.main()
