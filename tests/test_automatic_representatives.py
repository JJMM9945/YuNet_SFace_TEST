import unittest
from copy import deepcopy
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from tests.experiments.face_lab.automatic_representatives import (SETTINGS, automatic_stage, decide,
                                                freeze_automatic_anchors, rank_candidates, run_auto_comparison)
from tests.experiments.face_lab.evaluation import read_annotations
from tests.experiments.face_lab.models import sha256
from tests.experiments.face_lab.pipeline import write_csv, write_json
from tests.experiments.face_lab.review import context_hash


def fixture(angles, groups):
    faces=[];vectors={}
    for i,(angle,group) in enumerate(zip(angles,groups)):
        fid=f'f{i:02d}';theta=np.deg2rad(angle)
        vectors[fid]=np.array([np.cos(theta),np.sin(theta)],np.float32)
        faces.append(dict(face_id=fid,photo_id=f'p{i:02d}',relative_path=f'{i}.jpg',group_id=group,
                          label=group or '미확인',status='grouped' if group else 'unknown',reason='' if group else 'ambiguous',
                          review='',quality_reason='',width=160,height=160,detection_score=.95,blur_variance=200,
                          diagnostic=dict(stages=[])))
    return faces,vectors


class AutomaticAnchorTests(unittest.TestCase):
    def test_end_to_end_predictions_precede_truth_loading_and_preserve_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'source';source.mkdir();inputs=root/'inputs';inputs.mkdir()
            faces,vectors=fixture([0,2,-2,35],['person001']*3+[''])
            for f in faces:f.update(x=0.,y=0.,crop_path='')
            matrix=np.zeros((4,128),np.float32)
            for i,f in enumerate(faces):matrix[i,:2]=vectors[f['face_id']]
            np.savez_compressed(source/'features.npz',face_ids=np.array([f['face_id'] for f in faces]),vectors=matrix)
            cache=dict(schema=1,signature={},input_files=[],input_root=str(inputs),photos=[],faces=deepcopy(faces),
                       features_sha256=sha256(source/'features.npz'))
            write_json(source/'cache.json',cache)
            write_csv(source/'faces.csv',faces,[k for k in faces[0] if k!='diagnostic'])
            write_json(source/'summary.json',dict(actual_grouping_config=dict(linkage='average',similarity_threshold=.4,
                       min_group_size=1,ambiguity_margin=.01,reassign_unknown=False)))
            truth=root/'truth.csv'
            annotations=[dict(face_id=f['face_id'],manual_person_id='A',include='1',annotation_status='labeled',
                              _context_sha256=context_hash(cache)) for f in faces]
            write_csv(truth,annotations,list(annotations[0]))
            before=sha256(source/'faces.csv');order=[]
            def stage(*args,**kwargs):
                order.append('predict');return automatic_stage(*args,**kwargs)
            def labels(*args,**kwargs):
                order.append('truth');return read_annotations(*args,**kwargs)
            with patch('tests.experiments.face_lab.automatic_representatives.automatic_stage',side_effect=stage), \
                 patch('tests.experiments.face_lab.automatic_representatives.read_annotations',side_effect=labels):
                output=run_auto_comparison(SimpleNamespace(source=str(source),ground_truth=str(truth),
                                            output=str(root/'results'),target_source=None))
            self.assertEqual(order[:3],['predict','predict','truth'])
            self.assertEqual(sha256(source/'faces.csv'),before)
            report=json.loads((output/'comparison.json').read_text())
            self.assertEqual(len(report['results']),6)
            self.assertEqual(report['anchor_evaluation']['fully_automatic']['evaluated_mixed_anchors'],0)
            self.assertTrue((output/'index.html').is_file())

    def test_selects_by_measurements_not_identity_fields(self):
        faces,vectors=fixture([0,2,-2,130],['person001']*3+['person002'])
        first=freeze_automatic_anchors(faces,vectors)
        self.assertIn('person001',first['anchors'])
        self.assertNotIn('person002',first['anchors'])
        for i,f in enumerate(faces):f['manual_person_id']='inconsistent-'+str(i)
        second=freeze_automatic_anchors(faces,vectors)
        self.assertEqual(first['anchors'],second['anchors'])
        self.assertEqual(first['audit'],second['audit'])

    def test_anchor_with_repeated_photo_is_review_only(self):
        faces,vectors=fixture([0,2,-2],['person001']*3)
        faces[1]['photo_id']=faces[0]['photo_id']
        result=freeze_automatic_anchors(faces,vectors)
        self.assertFalse(result['anchors'])
        self.assertIn('same_photo_review',result['audit'][0]['reasons'])

    def test_unknown_reassignment_is_frozen_order_independent_and_quality_safe(self):
        faces,vectors=fixture([0,2,-2,30,35,20],['person001']*3+['','',''])
        faces[-1]['quality_reason']='blurry_face'
        snapshot=freeze_automatic_anchors(faces,vectors);before=deepcopy(snapshot)
        result,decisions=automatic_stage(faces,vectors,snapshot,'unknown_reassignment')
        self.assertEqual(result[3]['group_id'],'person001')
        self.assertEqual(result[4]['group_id'],'person001')
        self.assertEqual(result[5]['group_id'],'')
        self.assertEqual(faces[3]['group_id'],'')
        self.assertEqual(snapshot,before)
        _,reverse=automatic_stage(list(reversed(faces)),vectors,snapshot,'unknown_reassignment')
        self.assertEqual(decisions,reverse)
        self.assertEqual({f['face_id'] for f in snapshot['references']['person001']},{'f00','f01','f02'})

    def test_small_group_members_all_must_pass(self):
        faces,vectors=fixture([0,2,-2,30,120],['person001']*3+['person002']*2)
        snapshot=freeze_automatic_anchors(faces,vectors)
        result,decisions=automatic_stage(faces,vectors,snapshot,'small_group_linking')
        self.assertEqual([f['group_id'] for f in result[-2:]],['person002','person002'])
        self.assertTrue(all(d['target_group'] is None for d in decisions))

    def test_nonanchor_competes_and_is_not_silently_skipped(self):
        faces,vectors=fixture([0,2,-2,35,36],['person001']*3+['person002',''])
        snapshot=freeze_automatic_anchors(faces,vectors)
        _,decisions=automatic_stage(faces,vectors,snapshot,'unknown_reassignment')
        self.assertEqual(decisions[0]['top1_group_id'],'person002')
        self.assertEqual(decisions[0]['reason'],'top_candidate_not_automatic_anchor')

    def test_missing_candidates_and_runner_up_are_null(self):
        faces,vectors=fixture([0,2,-2,35],['person001']*3+[''])
        snap=freeze_automatic_anchors(faces,vectors)
        candidates=rank_candidates([faces[-1]],snap,vectors)
        d=decide([faces[-1]],candidates)
        self.assertIsNone(d['top2_group_id']);self.assertIsNone(d['top2_score']);self.assertIsNone(d['score_gap'])
        self.assertEqual(decide([faces[-1]],[])['reason'],'no_measured_candidate')

    def test_same_photo_collision_is_held_not_forced(self):
        faces,vectors=fixture([0,2,-2,30],['person001']*3+[''])
        faces[-1]['photo_id']=faces[0]['photo_id']
        snap=freeze_automatic_anchors(faces,vectors)
        _,decisions=automatic_stage(faces,vectors,snap,'unknown_reassignment')
        self.assertEqual(decisions[0]['reason'],'same_photo_review')
        self.assertIsNone(decisions[0]['target_group'])

    def test_singleton_threshold_is_stricter(self):
        faces,vectors=fixture([0,2,-2,62],['person001']*3+['person002'])
        snap=freeze_automatic_anchors(faces,vectors)
        candidates=rank_candidates([faces[-1]],snap,vectors,excluded_group='person002')
        self.assertIsNotNone(decide([faces[-1]],candidates,singleton=False)['target_group'])
        self.assertIsNone(decide([faces[-1]],candidates,singleton=True)['target_group'])
