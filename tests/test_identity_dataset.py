import unittest
from copy import deepcopy
from unittest.mock import patch
import tempfile
from pathlib import Path
import json
import re

import numpy as np

from tests.experiments.face_lab.identity_dataset import content_photo_faces, exact_photo_face_matches, verify_reference_runtime, _make_review
from tests.experiments.face_lab.models import sha256


def face(fid, photo):
    return dict(face_id=fid, photo_id=photo, x=10., y=20., width=50., height=60.,
                landmarks=[15.,25.,45.,25.,30.,40.,20.,60.,40.,60.])


class IdentityDatasetTests(unittest.TestCase):
    def setUp(self):
        self.reference = [face('old-face', 'old-photo')]
        self.target = [face('new-face', 'renamed-photo')]
        self.old_cache = dict(photos=[dict(photo_id='old-photo', sha256='same-content')])
        self.new_cache = dict(photos=[dict(photo_id='renamed-photo', sha256='same-content')])
        self.old_vectors = {'old-face': np.array([1.,0.], np.float32)}
        self.new_vectors = {'new-face': np.array([1.,0.], np.float32)}

    def matches(self):
        return exact_photo_face_matches(self.reference, self.target, self.old_cache, self.new_cache,
                                        self.old_vectors, self.new_vectors)

    def test_renamed_exact_photo_can_transfer_face_but_not_by_id(self):
        self.assertEqual(self.matches(), {'new-face':'old-face'})
        self.new_cache['photos'][0]['sha256'] = 'different-content'
        self.assertEqual(self.matches(), {})

    def test_geometry_landmark_and_embedding_changes_block_transfer(self):
        original = deepcopy(self.target)
        self.target[0]['x'] += 1
        self.assertEqual(self.matches(), {})
        self.target = deepcopy(original)
        self.target[0]['landmarks'][0] += 1
        self.assertEqual(self.matches(), {})
        self.target = deepcopy(original)
        self.new_vectors['new-face'] = np.array([0.,1.], np.float32)
        self.assertEqual(self.matches(), {})

    def test_ambiguous_duplicate_reference_does_not_transfer(self):
        duplicate = deepcopy(self.reference[0]); duplicate['face_id'] = 'another-old-face'
        self.reference.append(duplicate)
        self.old_vectors['another-old-face'] = self.old_vectors['old-face']
        self.assertEqual(self.matches(), {})

    def test_content_hash_blocks_same_photo_self_support_across_renames(self):
        old = content_photo_faces(self.reference, self.old_cache)
        new = content_photo_faces(self.target, self.new_cache)
        self.assertEqual(old[0]['photo_id'], new[0]['photo_id'])
        self.assertEqual(self.reference[0]['photo_id'], 'old-photo')
        self.assertEqual(self.target[0]['photo_id'], 'renamed-photo')

    def test_only_platform_mismatch_is_eligible_for_runtime_replay(self):
        signature = dict(runtime=dict(platform='old', opencv='same'), models={}, extraction={})
        cache = dict(signature=signature)
        self.assertEqual(verify_reference_runtime(cache, signature, {}, 'unused')['status'], 'exact_runtime_signature')
        changed = deepcopy(signature);changed['runtime']['opencv'] = 'changed'
        with self.assertRaises(ValueError):
            verify_reference_runtime(cache, changed, {}, 'unused')

    def test_platform_change_requires_replaying_every_reference_vector(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); path = root/'photo.bin'
            path.write_bytes(b'original reference photo')
            face = self.reference[0]
            face['detection_score'] = .9
            signature = dict(runtime=dict(platform='old', opencv='same'), models={}, extraction={})
            cache = dict(signature=signature, faces=[face], input_root=str(root),
                         photos=[dict(photo_id=face['photo_id'],relative_path=path.name,sha256=sha256(path))])
            current = deepcopy(signature); current['runtime']['platform'] = 'new'
            with patch('tests.experiments.face_lab.identity_dataset.load', return_value=(None,object())), \
                 patch('tests.experiments.face_lab.identity_dataset.decode_oriented', return_value=np.zeros((10,10,3))), \
                 patch('tests.experiments.face_lab.identity_dataset.extract', return_value=(None,self.old_vectors['old-face'])):
                result=verify_reference_runtime(cache,current,self.old_vectors,'unused')
                self.assertEqual(result['faces_checked'],1)
                self.assertEqual(result['maximum_absolute_vector_error'],0.)
            with patch('tests.experiments.face_lab.identity_dataset.load', return_value=(None,object())), \
                 patch('tests.experiments.face_lab.identity_dataset.decode_oriented', return_value=np.zeros((10,10,3))), \
                 patch('tests.experiments.face_lab.identity_dataset.extract', return_value=(None,np.array([0.,1.],np.float32))):
                with self.assertRaisesRegex(ValueError,'vector replay differs'):
                    verify_reference_runtime(cache,current,self.old_vectors,'unused')

    def test_review_prefills_only_explicit_transfers_not_predictions(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            faces = [face('confirmed','p1'),face('predicted','p2')]
            cache = dict(signature={},input_files=[],faces=faces)
            store = dict(identities={'stable-id':dict(display_name='KnownPerson')})
            _make_review(output,faces,cache,{'confirmed':'stable-id'},[],store)
            text = (output/'정답입력.html').read_text()
            payload = json.loads(re.search(r'<script type="application/json" id="data">(.*?)</script>',text,re.S).group(1))
            by_id={f['face_id']:f for f in payload['faces']}
            self.assertEqual(by_id['confirmed']['manual_person_id'],'KnownPerson')
            self.assertEqual(by_id['predicted']['manual_person_id'],'')
            self.assertEqual(by_id['predicted']['annotation_status'],'unlabeled')
