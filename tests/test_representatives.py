"""Frozen representatives must not propagate uncertain or single-edge assignments."""
import copy
import unittest
import numpy as np
from tests.experiments.face_lab.representative_experiment import (SETTINGS, representatives, unknown_reassignment,
                                                merge_groups, validate_count_reference)


def fixture(angles, gids, photos=None):
    faces=[];vectors={}
    for i,(angle,gid) in enumerate(zip(angles,gids)):
        fid=f'f{i:02d}';theta=np.deg2rad(angle)
        vectors[fid]=np.array([np.cos(theta),np.sin(theta)],np.float32)
        faces.append(dict(face_id=fid,photo_id=(photos or [str(n) for n in range(len(gids))])[i],
                          group_id=gid,label=gid or 'unknown',status='grouped' if gid else 'unknown',
                          reason='' if gid else 'ambiguous',review='',quality_reason='',
                          width=160,height=160,detection_score=.95,blur_variance=200,
                          diagnostic=dict(stages=[])))
    return faces,vectors


class RepresentativeTests(unittest.TestCase):
    def test_representative_mean_threshold_changes_assignment(self):
        faces, vectors = fixture([0, 0, 66], ['person001', 'person001', ''])
        strict = copy.deepcopy(faces)
        loose = copy.deepcopy(faces)
        a = unknown_reassignment(strict, vectors, dict(SETTINGS, score_threshold=.45))
        b = unknown_reassignment(loose, vectors, dict(SETTINGS, score_threshold=.40))
        self.assertEqual(a['decisions'][0]['reason'], 'score_below_threshold')
        self.assertEqual(b['decisions'][0]['reason'], 'assigned_to_frozen_representatives')
        self.assertEqual(strict[-1]['group_id'], '')
        self.assertEqual(loose[-1]['group_id'], 'person001')

    def test_quality_then_embedding_diversity(self):
        faces,vectors=fixture([0,0,60],['person001']*3)
        faces[2]['blur_variance']=160
        chosen,audit=representatives(faces,vectors,count=2,weight=.9)
        self.assertEqual([f['face_id'] for f in chosen],['f00','f02'])
        self.assertGreater(audit[1]['diversity'],0)

    def test_five_representatives_extend_the_same_three(self):
        faces,vectors=fixture([0,10,25,40,60,80],['person001']*6)
        three,audit_three=representatives(faces,vectors,count=3)
        five,audit_five=representatives(list(reversed(faces)),vectors,count=5)
        self.assertEqual([f['face_id'] for f in three],[f['face_id'] for f in five[:3]])
        self.assertEqual(audit_three,audit_five[:3])
        self.assertEqual(len({f['face_id'] for f in five}),5)

    def test_count_comparison_rejects_other_setting_or_source_changes(self):
        provenance=dict(source_faces_sha256='faces',features_sha256='features',
                        truth_sha256='truth',feedback_sha256='feedback')
        previous=dict(settings=dict(SETTINGS),**provenance)
        five=dict(SETTINGS,representatives=5)
        self.assertEqual(validate_count_reference(previous,five,provenance)['representatives'],3)
        with self.assertRaisesRegex(ValueError,'only in representative count'):
            validate_count_reference(previous,dict(five,margin=.05),provenance)
        with self.assertRaisesRegex(ValueError,'different source'):
            validate_count_reference(previous,five,dict(provenance,truth_sha256='other'))
        with self.assertRaisesRegex(ValueError,'only in representative count'):
            validate_count_reference(previous,dict(SETTINGS),provenance)

    def test_reassignment_frozen_order_independent_and_quality_excluded(self):
        faces,vectors=fixture([0,5,130,135,20,25,10],['person001','person001','person002','person002','','',''])
        faces[-1]['quality_reason']='blurry_face';faces[-1]['reason']='blurry_face'
        reverse=copy.deepcopy(faces[::-1])
        a=unknown_reassignment(faces,vectors);b=unknown_reassignment(reverse,vectors)
        self.assertEqual(a,b)
        self.assertEqual(faces[4]['group_id'],'person001');self.assertEqual(faces[5]['group_id'],'person001')
        self.assertEqual(faces[6]['group_id'],'')
        refs={f['face_id'] for group in a['representatives'].values() for f in group}
        self.assertFalse({'f04','f05','f06'} & refs)
        self.assertEqual({f['face_id']:f['group_id'] for f in faces},{f['face_id']:f['group_id'] for f in reverse})

    def test_one_representative_cannot_automatically_assign_and_no_fake_second_score(self):
        faces,vectors=fixture([0,5],['person001',''])
        audit=unknown_reassignment(faces,vectors)
        self.assertEqual(faces[1]['group_id'],'')
        self.assertEqual(audit['decisions'][0]['reason'],'insufficient_support')
        self.assertIsNone(audit['decisions'][0]['score_gap'])

    def test_flagged_group_competes_but_is_not_an_automatic_target(self):
        faces,vectors=fixture([0,5,7],['person001','person001',''],photos=['p','p','q'])
        for f in faces[:2]:f['review']='same_photo_same_group'
        audit=unknown_reassignment(faces,vectors)
        self.assertEqual(audit['decisions'][0]['reason'],'candidate_group_already_flagged')
        self.assertEqual(faces[2]['group_id'],'')

    def test_mutual_merge_is_disjoint_and_does_not_chain(self):
        faces,vectors=fixture([0,1,8,9,22,23],['person001','person001','person002','person002','person003','person003'])
        settings=dict(representatives=3,diversity_weight=.3,min_support=2,support_threshold=.4,score_threshold=.45,margin=.001)
        audit=merge_groups(faces,vectors,settings)
        self.assertEqual(audit['accepted_pairs'],[('person001','person002')])
        self.assertEqual([f['group_id'] for f in faces],['person001']*4+['person003']*2)
        reverse=fixture([0,1,8,9,22,23],['person001','person001','person002','person002','person003','person003'])[0][::-1]
        self.assertEqual(merge_groups(reverse,vectors,settings)['accepted_pairs'],audit['accepted_pairs'])

    def test_ambiguous_competitor_prevents_reassignment(self):
        faces,vectors=fixture([0,5,40,45,22],['person001','person001','person002','person002',''])
        audit=unknown_reassignment(faces,vectors)
        self.assertEqual(audit['decisions'][0]['reason'],'competing_group_margin_insufficient')
        self.assertEqual(faces[-1]['group_id'],'')


if __name__=='__main__':unittest.main()
