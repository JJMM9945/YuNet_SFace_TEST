from io import BytesIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from face_lab.engine import FaceEngine
from face_lab.photo_albums import run


class FakeEngine:
    initial_groups = FaceEngine.initial_groups

    def __init__(self, models, settings):
        self.settings = settings
        assert settings.score_threshold == .75

    def extract_photo(self, encoded, pid):
        with Image.open(BytesIO(encoded)) as image:
            count = image.getpixel((0, 0))[0]
        faces, vectors = [], {}
        for i in range(count):
            fid = f'{pid}_f{i}'
            faces.append(dict(face_id=fid,photo_id=pid,x=0,y=0,width=100,height=100,
                              landmarks=[],quality_reason='',detection_score=.99,blur_variance=100))
            vectors[fid] = np.eye(3,dtype=np.float32)[i]
        return faces, vectors


class WorkflowTests(unittest.TestCase):
    def test_arbitrary_folder_recursive_failed_image_and_report(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)/'내 사진'
            nested = folder/'하위 폴더'
            nested.mkdir(parents=True)
            for count in range(4):
                Image.new('RGB',(20,20),(count,0,0)).save(nested/f'{count}.png')
            (folder/'broken.jpg').write_bytes(b'not an image')
            with patch('face_lab.photo_albums.FaceEngine',FakeEngine):
                output = run(folder,Path(temp)/'출력')
            summary=json.loads((output/'summary.json').read_text())
            self.assertEqual((summary['total_photos'],summary['failed_photos'],summary['zero_face_photos']),(5,1,1))
            self.assertEqual(summary['detection']['score_threshold'],.75)
            cases=json.loads((output/'photo_collections.json').read_text())
            self.assertEqual([len(cases[k]) for k in ('solo','pair','crowd')],[1,1,1])
            self.assertEqual(sum(len(g['photos']) for g in cases['all']),5)
            states={p['state'] for g in cases['zero'] for p in g['photos']}
            self.assertEqual(states,{'failed','assigned'})
            self.assertTrue((output/'features.npz').is_file())
            page=(output/'index.html').read_text()
            self.assertIn('YuNet 0.75',page)
            self.assertIn('분석 실패 1장',page)
            self.assertNotIn('__REPORT_DATA__',page)

    def test_empty_input_and_nested_output_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp)
            with self.assertRaisesRegex(ValueError,'지원하는 사진'):
                run(folder,folder.parent/'output')
            with self.assertRaisesRegex(ValueError,'입력 사진 폴더 밖'):
                run(folder,folder/'output')


if __name__ == '__main__':
    unittest.main()
