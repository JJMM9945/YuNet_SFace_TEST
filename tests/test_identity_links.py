import json
import csv
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from tests.experiments.face_lab.identities import (apply_identity_changes, empty_store, identity_collections,
                                 register_identity, undo_last_change, validate_store)
from tests.experiments.face_lab.identity_experiment import (SETTINGS, frozen_identity_representatives,
                                          photo_partition, predict_identities, test_identity_links)
from tests.experiments.face_lab.models import sha256
from tests.experiments.face_lab.pipeline import write_csv, write_json
from tests.experiments.face_lab.review import context_hash
from PIL import Image


def fixture():
    faces = [dict(face_id=f'f{i}', photo_id=f'p{i}', relative_path=f'p{i}.jpg',
                  x=0, y=0, width=160, height=160, detection_score=.95, blur_variance=200,
                  quality_reason='', group_id=('g1' if i < 2 else 'g2')) for i in range(4)]
    cache = dict(signature={}, input_files=[], features_sha256='features', faces=deepcopy(faces))
    return faces, cache


def vector(degrees):
    radians = np.deg2rad(degrees)
    return np.array([np.cos(radians), np.sin(radians)], np.float32)


class IdentityLinkTests(unittest.TestCase):
    def test_real_file_workflow_preserves_source_and_creates_reloadable_store(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = root/'source'; source.mkdir(); (source/'crops').mkdir()
            inputs = root/'input'; inputs.mkdir()
            faces, cache = fixture()
            photo_ids = {}
            for name in ('seed', 'development', 'evaluation'):
                photo_ids[name] = next('photo'+str(i) for i in range(100) if photo_partition('photo'+str(i)) == name)
            for index, face in enumerate(faces):
                face['photo_id'] = photo_ids['seed' if index < 2 else 'development' if index == 2 else 'evaluation']
                face.update(status='grouped', reason='', crop_path='crops/'+face['face_id']+'.jpg')
                Image.new('RGB', (112,112), 'white').save(source/face['crop_path'])
            cache['faces'] = deepcopy(faces)
            cache.update(input_root=str(inputs), photos=[dict(photo_id=pid) for pid in set(photo_ids.values())])
            matrix = np.zeros((4,128), np.float32); matrix[:,0] = 1
            np.savez_compressed(source/'features.npz', face_ids=np.array([f['face_id'] for f in faces]), vectors=matrix)
            cache['features_sha256'] = sha256(source/'features.npz')
            write_json(source/'cache.json', cache)
            write_csv(source/'faces.csv', faces, list(faces[0]))
            truth = root/'truth.csv'
            rows = [dict(face_id=face['face_id'], manual_person_id='A' if i % 2 == 0 else 'B',
                         include='1', annotation_status='labeled', _context_sha256=context_hash(cache))
                    for i, face in enumerate(faces)]
            write_csv(truth, rows, list(rows[0]))
            before = sha256(source/'faces.csv')
            output = test_identity_links(SimpleNamespace(run=str(source), ground_truth=str(truth), output=str(root/'results')))
            self.assertEqual(sha256(source/'faces.csv'), before)
            store = json.loads((output/'identity_store.json').read_text())
            validate_store(store, cache)
            self.assertEqual(len(store['links']), 4)
            self.assertEqual(len(store['identities']), 2)
            self.assertTrue((output/'index.html').is_file())

    def test_face_links_span_groups_and_photo_search_deduplicates(self):
        faces, cache = fixture()
        faces[2]['photo_id'] = faces[0]['photo_id']
        faces[3]['group_id'] = ''
        store = empty_store(cache, 'original')
        target = register_identity(store, 'confirmed-A')
        before = deepcopy(faces)
        store = apply_identity_changes(store, cache, {'f0': target, 'f2': target, 'f3': target}, 'User confirmed')
        self.assertEqual(faces, before)
        result = identity_collections(faces, store)[target]
        self.assertEqual(result['automatic_groups'], ['g1', 'g2'])
        self.assertEqual(result['photo_ids'], ['p0', 'p3'])
        self.assertEqual(result['unknown_faces'], ['f3'])
        self.assertNotIn('f1', store['links'])

    def test_atomic_edit_exclusion_unknowns_and_model_mismatch(self):
        faces, cache = fixture()
        store = empty_store(cache, 'original', excluded=['f3'])
        target = register_identity(store, 'confirmed-A')
        for changes in ({'missing': target}, {'f3': target}, {'f0': 'missing_identity'}):
            before = deepcopy(store)
            with self.assertRaises(ValueError):
                apply_identity_changes(store, cache, changes, 'User change')
            self.assertEqual(store, before)
        other = deepcopy(cache); other['features_sha256'] = 'new-model-features'
        with self.assertRaisesRegex(ValueError, 'mismatch'):
            validate_store(store, other)

    def test_persistence_reassignment_detach_and_undo(self):
        _, cache = fixture()
        store = empty_store(cache, 'original')
        a = register_identity(store, 'confirmed-A'); b = register_identity(store, 'confirmed-B')
        store = apply_identity_changes(store, cache, {'f0': a, 'f1': a}, 'Confirm A')
        revised = apply_identity_changes(store, cache, {'f0': b}, 'Correct mixed face')
        self.assertEqual(store['links']['f0'], a)
        self.assertEqual(revised['links']['f1'], a)
        restored = json.loads(json.dumps(revised))
        validate_store(restored, cache)
        undo = undo_last_change(restored, cache)
        self.assertEqual(undo['links'], store['links'])
        self.assertEqual(undo['revision'], 3)
        detached = apply_identity_changes(undo, cache, {'f0': None}, 'Detach')
        self.assertNotIn('f0', detached['links'])
        self.assertEqual(undo_last_change(detached, cache)['links'], undo['links'])

    def test_group_renumbering_does_not_change_extraction_binding(self):
        faces, cache = fixture()
        store = empty_store(cache, 'original'); identity = register_identity(store, 'A')
        store = apply_identity_changes(store, cache, {'f0': identity}, 'Explicit confirmation')
        regrouped = deepcopy(cache)
        regrouped['faces'][0]['group_id'] = 'new-number'
        validate_store(store, regrouped)
        self.assertEqual(register_identity(store, 'A'), identity)


class SeedIdentityTests(unittest.TestCase):
    def test_single_representative_never_assigns_or_fabricates_second_score(self):
        faces, _ = fixture(); vectors = {face['face_id']: vector(0) for face in faces}
        reps = frozen_identity_representatives(faces[:1], vectors, {'f0': 'A'})
        decision = predict_identities(faces[2:3], vectors, reps)[0]
        self.assertIsNone(decision['predicted_identity'])
        self.assertIsNone(decision['score_gap'])
        self.assertEqual(decision['reason'], 'insufficient_distinct_photo_support')

    def test_multigroup_references_are_seed_only_frozen_and_order_independent(self):
        faces, _ = fixture(); faces[1]['group_id'] = 'other-subgroup'
        vectors = {face['face_id']: vector(i*3) for i, face in enumerate(faces)}
        links = {'f0': 'A', 'f1': 'A'}
        reps = frozen_identity_representatives(faces[:2], vectors, links)
        before = deepcopy(reps)
        decisions = predict_identities(faces[2:], vectors, reps)
        self.assertEqual([row['predicted_identity'] for row in decisions], ['A', 'A'])
        self.assertEqual(reps, before)
        self.assertEqual({rep['face_id'] for rep in reps['A']}, {'f0', 'f1'})
        self.assertEqual(predict_identities(list(reversed(faces[2:])), vectors, reps), decisions)

    def test_same_photo_cannot_supply_two_support_votes(self):
        faces, _ = fixture(); faces[1]['photo_id'] = faces[0]['photo_id']
        vectors = {face['face_id']: vector(0) for face in faces}
        reps = frozen_identity_representatives(faces[:2], vectors, {'f0': 'A', 'f1': 'A'})
        self.assertEqual(len(reps['A']), 1)
        result = predict_identities(faces[2:3], vectors, reps)[0]
        self.assertIsNone(result['predicted_identity'])

    def test_competition_and_quality_exclusion_hold_faces(self):
        faces, _ = fixture(); vectors = {face['face_id']: vector(0) for face in faces}
        representatives = [{'face_id': f'f{i}', 'photo_id': f'p{i}'} for i in (0, 1)]
        reps = {'A': representatives, 'B': deepcopy(representatives)}
        decisions = predict_identities(faces[2:3], vectors, reps)
        self.assertEqual(decisions[0]['reason'], 'competing_identity_margin_insufficient')
        faces[2]['quality_reason'] = 'blurry_face'
        self.assertEqual(predict_identities(faces[2:3], vectors, reps)[0]['reason'], 'quality_excluded_or_missing_embedding')

    def test_photo_split_is_deterministic(self):
        self.assertEqual(photo_partition('same-photo'), photo_partition('same-photo'))
        self.assertIn(photo_partition('same-photo'), {'seed', 'development', 'evaluation'})
