"""Explicit human corrections preserve provenance and unmentioned predictions."""
import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
from tests.experiments.face_lab.corrections import apply_corrections, correct_groups, create_template
from tests.experiments.face_lab.evaluation import evaluate
from tests.experiments.face_lab.models import sha256
from tests.experiments.face_lab.pipeline import write_csv, write_json, regroup
from tests.experiments.face_lab.review import context_hash


def face(fid, photo, group, reason=''):
    return dict(face_id=fid, photo_id=photo, relative_path=photo+'.jpg', group_id=group,
                label='인물'+group[6:] if group else '미확인', status='grouped' if group else 'unknown',
                reason=reason, review='', quality_reason='', decision_state='assigned' if group else 'held',
                x=10., y=10., width=20., height=20., landmarks=[15.,15.]*5,
                detection_score=.9, blur_variance=100., crop_path='crops/'+fid+'.jpg',
                diagnostic=dict(stages=[],candidates=[]))


class CorrectionTests(unittest.TestCase):
    def test_explicit_split_merge_and_false_preserve_unmentioned(self):
        faces=[face('a','p','person001'),face('b','p','person001'),face('c','q','person002'),
               face('d','r',''),face('e','s','person003'),face('f','t','person004')]
        feedback=dict(assignments={'a':'person002','b':'person005','d':'person002'},
                      false_detections={'e':'손'})
        before=faces[-1].copy()
        changes=apply_corrections(faces,feedback)
        self.assertEqual([f['group_id'] for f in faces],['person002','person005','person002','person002','','person004'])
        self.assertEqual(faces[4]['status'],'false_detection')
        self.assertEqual(faces[-1]['group_id'],before['group_id'])
        self.assertEqual(faces[-1]['label_origin'],'automatic_unreviewed')
        self.assertEqual(faces[0]['diagnostic']['manual_correction']['before']['group_id'],'person001')
        self.assertEqual(len(changes),4)

    def test_bad_ids_and_conflicts_are_rejected_before_mutation(self):
        for feedback in (dict(assignments={'missing':'person001'}),
                         dict(assignments={'a':'person002'},false_detections={'a':'hand'}),
                         dict(assignments={'a':'../group'}),dict(false_detections={'a':''})):
            faces=[face('a','p','person001')]
            with self.assertRaises(ValueError):apply_corrections(faces,feedback)
            self.assertEqual(faces[0]['group_id'],'person001')

    def test_output_bound_to_source_and_evaluation_blocked(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);source=root/'source';source.mkdir();inputs=root/'input';inputs.mkdir()
            for name in ('photos','crops'):(source/name).mkdir()
            faces=[face('p_f001','p','person001'),face('p_f002','p','person001'),face('q_f001','q','')]
            for f in faces:
                Image.new('RGB',(112,112),'white').save(source/f['crop_path'])
            photos=[]
            for pid,count in [('p',2),('q',1)]:
                Image.new('RGB',(100,100),'white').save(source/'photos'/f'{pid}.jpg')
                photos.append(dict(photo_id=pid,relative_path=pid+'.jpg',status='success',error='',detected_faces=count))
            np.savez_compressed(source/'features.npz',face_ids=np.array([f['face_id'] for f in faces]),
                                vectors=np.tile(np.eye(1,128,dtype=np.float32),(3,1)))
            raw=[{k:v for k,v in f.items() if k not in {'diagnostic','group_id','label','status','reason','review','decision_state'}} for f in faces]
            cache=dict(schema=1,signature={},input_root=str(inputs),input_files=[],config={},photos=photos,faces=raw,
                       features_sha256=sha256(source/'features.npz'))
            write_json(source/'cache.json',cache)
            write_json(source/'diagnostics.json',{f['face_id']:f['diagnostic'] for f in faces})
            write_csv(source/'faces.csv',faces,['face_id','photo_id','relative_path','group_id','label','status','reason','review','decision_state'])
            write_json(source/'summary.json',dict(mode='analyze'))
            write_json(source/'run_config.json',dict(options=dict(linkage='average',similarity_threshold=.4,min_group_size=1,
                                                                  ambiguity_margin=.01,reassign_unknown=False)))
            template_path = root/'template.json'
            create_template(SimpleNamespace(source=str(source), output=str(template_path)))
            template = json.loads(template_path.read_text())
            self.assertEqual(template['assignments'], {})
            self.assertEqual(template['context_sha256'], context_hash(cache))
            with self.assertRaises(FileExistsError):
                create_template(SimpleNamespace(source=str(source), output=str(template_path)))
            feedback=dict(schema=1,context_sha256=context_hash(cache),source_faces_sha256=sha256(source/'faces.csv'),
                          features_sha256=cache['features_sha256'],assignments={'p_f001':'person002'},
                          false_detections={'q_f001':'손'})
            path=root/'feedback.json';write_json(path,feedback)
            original=sha256(source/'faces.csv')
            output=correct_groups(SimpleNamespace(source=str(source),corrections=str(path),output=str(root/'out')))
            self.assertEqual(sha256(source/'faces.csv'),original)
            summary=json.loads((output/'summary.json').read_text())
            self.assertEqual((summary['groups'],summary['unknown_faces'],summary['false_detections']),(2,0,1))
            with (output/'faces.csv').open(encoding='utf-8-sig',newline='') as f: result={r['face_id']:r for r in csv.DictReader(f)}
            self.assertEqual(result['p_f002']['group_id'],'person001')
            self.assertEqual(result['q_f001']['status'],'false_detection')
            with (output/'ground_truth_template.csv').open(encoding='utf-8-sig',newline='') as f: truth={r['face_id']:r for r in csv.DictReader(f)}
            self.assertEqual(truth['q_f001']['include'],'0')
            with (output/'confirmed_ground_truth.csv').open(encoding='utf-8-sig',newline='') as f: confirmed={r['face_id']:r for r in csv.DictReader(f)}
            self.assertEqual(confirmed['p_f001']['manual_person_id'],'person002')
            self.assertEqual(confirmed['p_f002']['manual_person_id'],'')
            self.assertEqual(confirmed['q_f001']['annotation_status'],'excluded')
            with self.assertRaisesRegex(ValueError,'human-corrected'):
                evaluate(SimpleNamespace(run=str(output),ground_truth=str(path)))
            with self.assertRaisesRegex(ValueError,'original automatic'):
                regroup(SimpleNamespace(source=str(output),input=str(inputs)))
            feedback['source_faces_sha256']='wrong';write_json(path,feedback)
            with self.assertRaisesRegex(ValueError,'faces.csv mismatch'):
                correct_groups(SimpleNamespace(source=str(source),corrections=str(path),output=str(root/'other')))
            self.assertFalse((root/'other').exists())
