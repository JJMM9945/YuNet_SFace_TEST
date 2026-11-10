"""Model-independent tests of recognition contracts and leakage-safe selection."""
import tempfile
import unittest
from pathlib import Path
import numpy as np

from tests.experiments.face_lab.auraface import TEMPLATE, similarity_matrix, normalize_embedding, input_blob
from tests.experiments.face_lab.auraface_experiment import select_threshold, photo_fold, predict, save_features
from tests.experiments.face_lab.evaluation import pair_metrics


class AuraFaceTests(unittest.TestCase):
    def test_five_point_similarity_recovers_known_transform(self):
        angle=.37;rotation=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])*2.7
        original=TEMPLATE@rotation.T+np.array([231,154])
        matrix=similarity_matrix(original)
        np.testing.assert_allclose(original@matrix[:,:2].T+matrix[:,2],TEMPLATE,atol=1e-8)
        np.testing.assert_allclose(similarity_matrix(TEMPLATE),np.eye(2,3),atol=1e-10)
        for bad in (np.zeros((5,2)),np.full((5,2),np.nan),np.ones((4,2))):
            with self.assertRaises(ValueError):similarity_matrix(bad)

    def test_bgr_to_rgb_and_model_normalization(self):
        image=np.empty((112,112,3),np.uint8);image[:]=[10,20,30]
        np.testing.assert_allclose(input_blob(image,127.5,127.5)[0,:,0,0],(np.array([30,20,10])-127.5)/127.5,atol=1e-7)
        np.testing.assert_array_equal(input_blob(image,0,1)[0,:,0,0],[30,20,10])

    def test_embedding_dimensions_finiteness_and_normalization(self):
        vector=normalize_embedding(np.arange(512,dtype=np.float32))
        self.assertAlmostEqual(float(np.linalg.norm(vector)),1,places=6)
        for bad in (np.zeros(512),np.ones(128),np.full(512,np.nan),np.full(512,np.inf)):
            with self.assertRaises(ValueError):normalize_embedding(bad)

    def test_development_selection_obeys_false_mix_budget_and_ties(self):
        rows=[dict(threshold=.3,false_positive_pairs=1,recall=1,precision=.9),
              dict(threshold=.4,false_positive_pairs=0,recall=.8,precision=1),
              dict(threshold=.5,false_positive_pairs=0,recall=.8,precision=1)]
        self.assertEqual(select_threshold(rows),.5)
        self.assertEqual(select_threshold(list(reversed(rows))),.5)
        with self.assertRaises(ValueError):select_threshold(rows[:1])
        self.assertEqual(photo_fold('abc'),photo_fold('abc'))

    def test_grouping_never_uses_truth_and_unknowns_are_distinct(self):
        faces=[dict(face_id=f'f{i}',photo_id=str(i),width=120,height=120) for i in range(3)]
        vectors={f'f{i}':np.eye(3,dtype=np.float32)[i] for i in range(3)}
        pred=predict(faces,vectors,.4)
        truth={f'f{i}':dict(include='1',manual_person_id='same',annotation_status='labeled') for i in range(3)}
        metric=pair_metrics(pred,truth)
        self.assertEqual(metric['false_positive_pairs'],0);self.assertEqual(metric['false_negative_pairs'],3)
        for f in pred:f['group_id']=''
        self.assertEqual(pair_metrics(pred,truth)['false_negative_pairs'],3)

    def test_separate_feature_manifest_records_dimension_and_model(self):
        import json
        with tempfile.TemporaryDirectory() as d:
            vector=normalize_embedding(np.ones(512))
            save_features(Path(d),'auraface',{'f1':vector},{'sha256':'model-sha'},'source-sha')
            manifest=json.loads((Path(d)/'auraface_features_manifest.json').read_text())
            self.assertEqual(manifest['dimension'],512)
            self.assertEqual(manifest['model']['sha256'],'model-sha')
            self.assertFalse((Path(d)/'cache.json').exists())


if __name__=='__main__':unittest.main()
