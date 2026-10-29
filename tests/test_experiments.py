import copy
import itertools
import json
from argparse import Namespace
from io import BytesIO
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
from face_lab.grouping import group_faces, reassign_unknown, select_representatives, validate_vectors
from face_lab.preprocess import extract, decode_oriented, detection_input, to_original
from face_lab.evaluation import pair_metrics, read_annotations, read_split, evaluate
from face_lab.pipeline import load_vectors, write_csv, write_json
from face_lab.models import sha256
from face_lab.experiments import compare, select_config, memberships


def sample(angles):
    faces = [dict(face_id=f'f{i}', photo_id=f'p{i}', quality_reason='', width=100, height=120,
                  detection_score=.95, blur_variance=250) for i in range(len(angles))]
    vectors = {f['face_id']: np.array([np.cos(np.deg2rad(a)),np.sin(np.deg2rad(a))], np.float32)
               for f,a in zip(faces,angles)}
    return faces, vectors


class GroupExperimentTests(unittest.TestCase):
    def test_average_and_complete_use_correct_distance_direction(self):
        faces, vectors = sample([0,20,60])
        a,b = copy.deepcopy(faces),copy.deepcopy(faces)
        group_faces(a,vectors,.6,1,0,method='complete')
        group_faces(b,vectors,.6,1,0,method='average')
        self.assertEqual(len({f['group_id'] for f in a}),2)
        self.assertEqual(len({f['group_id'] for f in b}),1)
        self.assertAlmostEqual(b[2]['diagnostic']['own_group_score'],(np.cos(np.deg2rad(60))+np.cos(np.deg2rad(40)))/2,places=6)
        self.assertLess(b[2]['diagnostic']['own_min_similarity'],.6)
        # At a stricter threshold average also breaks the chain.
        group_faces(b,vectors,.7,1,0,method='average')
        self.assertEqual(len({f['group_id'] for f in b}),2)

    def test_permutations_and_exact_ties_are_stable(self):
        faces,vectors = sample([0,0,45,45])
        for method in ('complete','average'):
            baseline=None
            for order in itertools.permutations(faces):
                shuffled=copy.deepcopy(order)
                d=group_faces(shuffled,vectors,.8,1,.01,method=method)
                state={f['face_id']:(f['group_id'],f['reason']) for f in shuffled}
                if baseline is None: baseline=(state,d)
                self.assertEqual((state,d),baseline)

    def test_no_candidates_and_single_candidate_are_null(self):
        faces,vectors=sample([0])
        d=group_faces(faces,vectors,.4,1,.01)['f0']
        self.assertIsNone(d['top1_score']);self.assertIsNone(d['top2_score'])
        self.assertIsNone(d['candidate_gap']);self.assertIsNone(d['nearest_face_id'])
        self.assertIsNone(d['own_group_score']);self.assertEqual(d['singleton_prior'],1)
        faces,vectors=sample([0,10])
        d=group_faces(faces,vectors,.4,1,.01)['f0']
        self.assertIsNotNone(d['top1_score']);self.assertIsNone(d['top2_score'])
        self.assertIsNone(d['gap'])
        self.assertAlmostEqual(d['top1_score'],np.cos(np.deg2rad(10)),places=6)

    def test_quality_exclusion_is_diagnostic_only(self):
        faces,vectors=sample([0,0,5])
        faces[2]['quality_reason']='blurry_face'
        d=group_faces(faces,vectors,.8,1,.01)
        reassign_unknown(faces,vectors,.5,.01)
        self.assertEqual(faces[2]['group_id'],'')
        self.assertEqual(d['f2']['state'],'quality_excluded')
        self.assertTrue(d['f2']['candidates_diagnostic_only'])
        self.assertIsNotNone(d['f2']['nearest_face_id'])
        self.assertFalse(d['f2']['reassignment']['applied'])

    def test_missing_invalid_vectors_rejected_or_held(self):
        faces,_=sample([0]);d=group_faces(faces,{},.4,1,.01)
        self.assertEqual(d['f0']['reason'],'missing_embedding')
        for v in ([0.,0.],[2.,0.],[np.nan,0.]):
            with self.assertRaises(ValueError): validate_vectors({'x':np.array(v)})

    def test_ambiguity_and_follow_on_size_reason_are_distinct(self):
        faces,vectors=sample([0,30,60])
        d=group_faces(faces,vectors,.8,2,.03)
        self.assertTrue(any(f['reason']=='ambiguous_competing_cluster' for f in faces))
        self.assertTrue(any(f['reason']=='below_min_group_size_after_ambiguity' for f in faces))
        ambiguous=next(v for v in d.values() if v['ambiguity_rule_matched'])
        self.assertGreaterEqual(ambiguous['competitors'][0]['score'],.8-.03)
        self.assertLessEqual(ambiguous['own_competitor_gap'],.03)

    def seed_recovery(self):
        faces,vectors=sample([0,5,135,140,25,40])
        group_faces(faces,vectors,1,2,0)
        for f,gid in zip(faces,['person001','person001','person002','person002','','']):
            f.update(group_id=gid,status='grouped' if gid else 'unknown')
            f['diagnostic']['state']='assigned' if gid else 'held'
        return faces,vectors

    def test_recovery_frozen_order_independent_and_preserves_before(self):
        faces,vectors=self.seed_recovery();reverse=copy.deepcopy(faces[::-1])
        audit=reassign_unknown(faces,vectors,.7,.1)
        other=reassign_unknown(reverse,vectors,.7,.1)
        self.assertEqual(audit,other)
        self.assertEqual({f['face_id']:f['group_id'] for f in faces},{f['face_id']:f['group_id'] for f in reverse})
        self.assertEqual(faces[4]['group_id'],'person001')
        self.assertEqual(faces[5]['group_id'],'person001')
        refs={r['face_id'] for members in audit['representatives'].values() for r in members}
        self.assertNotIn('f4',refs);self.assertNotIn('f5',refs)
        self.assertEqual(faces[4]['diagnostic']['before_reassignment']['group_id'],'')
        self.assertTrue(faces[4]['diagnostic']['reassignment']['applied'])

    def test_recovery_needs_multiple_supports_and_competition_margin(self):
        faces,vectors=self.seed_recovery()
        faces[1]['group_id']='person003'  # near-identical competing singletons
        reassign_unknown(faces,vectors,.6,.1)
        self.assertEqual(faces[4]['group_id'],'')
        self.assertEqual(faces[4]['diagnostic']['reassignment']['reason'],'insufficient_representative_support')
        faces,vectors=self.seed_recovery()
        reassign_unknown(faces,vectors,.6,2)
        self.assertEqual(faces[4]['group_id'],'')
        self.assertEqual(faces[4]['diagnostic']['reassignment']['reason'],'representative_candidate_margin_insufficient')
        # Exactly one candidate group: no invented runner-up, still >=2 supports.
        faces,vectors=self.seed_recovery()
        faces=[f for f in faces if f['face_id'] not in {'f2','f3'}]
        reassign_unknown(faces,vectors,.7,.1)
        self.assertEqual(faces[2]['group_id'],'person001')
        self.assertIsNone(faces[2]['diagnostic']['reassignment']['top2_score'])

    def test_representative_selection_uses_quality_then_diversity(self):
        faces,vectors=sample([0,0,60])
        faces[2]['blur_variance']=160
        refs,audit=select_representatives(faces,vectors,2,.9)
        self.assertEqual(refs[0]['face_id'],'f0')
        self.assertEqual(refs[1]['face_id'],'f2')
        self.assertGreater(audit[1]['diversity'],0)

    def test_membership_comparison_ignores_group_renumbering(self):
        faces,_=sample([0,0]);faces[0]['group_id']=faces[1]['group_id']='A'
        other=copy.deepcopy(faces)
        for f in other:f['group_id']='Z'
        self.assertEqual(memberships(faces),memberships(other))


class PreprocessAndEvaluationTests(unittest.TestCase):
    def test_oriented_original_coordinates_and_feature_contract(self):
        image=Image.new('RGB',(203,101),(230,20,5));exif=image.getexif();exif[274]=6
        stream=BytesIO();image.save(stream,format='JPEG',exif=exif)
        bgr=decode_oriented(stream.getvalue());self.assertEqual(bgr.shape[:2],(203,101))
        small,sx,sy=detection_input(bgr,64)
        row=np.arange(15,dtype=np.float32);row[14]=.95
        mapped=to_original(row,sx,sy)
        for k in range(14): self.assertAlmostEqual(mapped[k],row[k]*(sx if k%2==0 else sy),places=5)
        class Recognizer:
            def alignCrop(self,im,r):
                np.testing.assert_array_equal(im,bgr);np.testing.assert_array_equal(r,mapped)
                assert im[0,0,2]>im[0,0,0]
                return np.zeros((112,112,3),np.uint8)
            def feature(self,crop):return np.arange(1,129,dtype=np.float32).reshape(1,128)
        crop,v=extract(Recognizer(),bgr,mapped)
        self.assertEqual(crop.shape,(112,112,3));self.assertAlmostEqual(np.linalg.norm(v),1,places=6)
        class Bad(Recognizer):
            def feature(self,crop):return np.zeros((1,128),np.float32)
        with self.assertRaisesRegex(ValueError,'invalid_feature'):extract(Bad(),bgr,mapped)

    def test_cache_shape_norm_and_checksum_are_validated(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)
            cache={'faces':[{'face_id':'f','quality_reason':''}]}
            for matrix in (np.ones((1,128)),np.ones((1,2))):
                np.savez(p/'features.npz',face_ids=['f'],vectors=matrix)
                cache['features_sha256']=sha256(p/'features.npz')
                with self.assertRaises(ValueError):load_vectors(p,cache)
            cache['features_sha256']='bad'
            with self.assertRaisesRegex(ValueError,'checksum'):load_vectors(p,cache)

    def test_pair_metrics_do_not_match_identity_strings_and_keep_unknown_fn(self):
        faces=[{'face_id':str(i),'group_id':g} for i,g in enumerate(['X','X','',''])]
        ann={str(i):dict(include='1',manual_person_id=t) for i,t in enumerate(['X','Y','X','X'])}
        m=pair_metrics(faces,ann)
        self.assertEqual(m['false_positive_pairs'],1);self.assertEqual(m['false_negative_pairs'],3)
        self.assertEqual(len(m['mixed_groups']),1);self.assertEqual(len(m['split_people']),1)
        with self.assertRaises(ValueError):pair_metrics(faces,{})

    def test_explicit_uncertain_and_missing_labels(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'truth.csv'
            rows=[dict(face_id='a',manual_person_id='',include='1',annotation_status='uncertain'),
                  dict(face_id='b',manual_person_id='',include='1',annotation_status='')]
            write_csv(path,rows,list(rows[0]))
            a=read_annotations(path,[{'face_id':'a'},{'face_id':'b'}])
            self.assertEqual(a['a']['annotation_status'],'uncertain');self.assertEqual(a['b']['annotation_status'],'unlabeled')

    def test_photo_split_rejects_duplicate_and_missing_photos(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'splits.csv';photos=[{'photo_id':'a'},{'photo_id':'b'}]
            write_csv(path,[{'photo_id':'a','split':'dev'},{'photo_id':'a','split':'eval'}],['photo_id','split'])
            with self.assertRaises(ValueError):read_split(path,photos)
            write_csv(path,[{'photo_id':'a','split':'dev'}],['photo_id','split'])
            with self.assertRaises(ValueError):read_split(path,photos)

    def test_evaluate_does_not_filter_groups_built_across_sets(self):
        with tempfile.TemporaryDirectory() as d:
            run=Path(d)
            write_csv(run/'faces.csv',[dict(face_id='a',group_id='g',photo_id='p')],['face_id','group_id','photo_id'])
            write_csv(run/'truth.csv',[dict(face_id='a',manual_person_id='A',include='1')],['face_id','manual_person_id','include'])
            with self.assertRaisesRegex(ValueError,'First group only'):
                evaluate(Namespace(run=d,ground_truth=str(run/'truth.csv'),subset='eval',split_manifest='unused'))

    def test_eval_sweep_requires_frozen_dev_config_before_writing(self):
        with tempfile.TemporaryDirectory() as d:
            args=Namespace(source=d,input=d,models=d,subset='eval',split_manifest=None,selected_config=None)
            cache=dict(photos=[dict(photo_id='p')],faces=[])
            with patch('face_lab.experiments.validate_cache',return_value=cache),patch('face_lab.experiments.load_vectors',return_value={}):
                with self.assertRaisesRegex(ValueError,'split-manifest'):compare(args)
            write_json(Path(d)/'comparison.json',dict(subset='all'))
            with self.assertRaisesRegex(ValueError,'Only a dev'):select_config(Namespace(comparison=d,experiment='baseline'))


if __name__=='__main__':unittest.main()
