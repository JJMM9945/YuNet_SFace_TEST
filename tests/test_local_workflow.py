# 역할: 모델 준비 후 빈 입력·손상 사진 격리·오프라인 실행·캐시 재검사 통합 검증
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image
from face_lab.__main__ import parser
from face_lab.models import MODELS, sha256
from face_lab.pipeline import analyze, regroup
from face_lab.review import context_hash, make_review, read_truth

MODELS_DIR = Path(__file__).resolve().parents[1] / 'models'


class LocalWorkflowTests(unittest.TestCase):
    @unittest.skipUnless(all((MODELS_DIR / m['name']).exists() for m in MODELS.values()),
                         'Local models absent; run prepare-models for integration check')
    def test_offline_empty_corrupt_regroup_and_cache_guard(self):
        with tempfile.TemporaryDirectory(prefix='face_lab_') as folder:
            root = Path(folder)
            inputs = root / '한글 입력'; inputs.mkdir()
            args = parser().parse_args(['analyze', '--input', str(inputs), '--output', str(root / 'out'),
                                       '--models', str(MODELS_DIR)])
            with patch('socket.socket', side_effect=AssertionError('Network forbidden')), contextlib.redirect_stdout(io.StringIO()):
                empty = analyze(args)
                self.assertEqual(json.loads((empty / 'summary.json').read_text())['total_photos'], 0)
                Image.new('RGB', (128, 96), 'white').save(inputs / '공백 사진.jpg')
                (inputs / '손상.jpg').write_bytes(b'not a jpeg')
                original = {p.name: sha256(p) for p in inputs.iterdir()}
                run = analyze(args)
                summary = json.loads((run / 'summary.json').read_text())
                self.assertEqual((summary['total_photos'], summary['successful_photos'], summary['failed_photos'],
                                  summary['zero_face_photos']), (2, 1, 1, 1))
                review = make_review(type('Args', (), {'run': str(run)})())
                self.assertTrue(review.exists())
                regroup_args = parser().parse_args(['regroup', '--input', str(inputs), '--source', str(run),
                                                    '--output', str(root / 'out'), '--models', str(MODELS_DIR)])
                second = regroup(regroup_args)
                self.assertEqual(sha256(run / 'features.npz'), sha256(second / 'features.npz'))
                self.assertEqual(original, {p.name: sha256(p) for p in inputs.iterdir()})
                (inputs / '손상.jpg').write_bytes(b'changed photo')
                with self.assertRaisesRegex(ValueError, 'Cache input mismatch'):
                    regroup(regroup_args)

    def test_truth_rejects_wrong_extraction_and_duplicate_faces(self):
        from face_lab.pipeline import write_csv
        cache = dict(signature={}, input_files=[], faces=[])
        context = context_hash(cache)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'truth.csv'
            row = dict(face_id='f1', manual_person_id='A01', include='1', _context_sha256='wrong')
            write_csv(path, [row], list(row))
            with self.assertRaisesRegex(ValueError, 'mismatch'):
                read_truth(path, {'f1'}, context)
            row['_context_sha256'] = context
            write_csv(path, [row, row], list(row))
            with self.assertRaisesRegex(ValueError, 'duplicate'):
                read_truth(path, {'f1'}, context)
