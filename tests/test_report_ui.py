"""역할: 결과 UI의 안전한 데이터 삽입·이미지 경로·단일 HTML 구성·빈 결과 검사."""
import json
from pathlib import Path
import tempfile
import unittest
from PIL import Image
from tests.experiments.face_lab.report_ui import render_html


class ReportUiTests(unittest.TestCase):
    def test_untrusted_names_escaped_and_paths_generated(self):
        with tempfile.TemporaryDirectory() as d:
            run=Path(d);(run/'crops').mkdir();Image.new('RGB',(20,20)).save(run/'crops/p_f001.jpg')
            face=dict(face_id='p_f001',photo_id='p',relative_path='</script><script>alert(1)</script>.jpg',
                      group_id='person001',label='인물001',status='grouped',crop_path='https://invalid.example/evil.jpg',reason='',review='')
            render_html(run,[dict(photo_id='p',relative_path=face['relative_path'],status='success',detected_faces=1)],[face])
            html=(run/'index.html').read_text(encoding='utf-8')
            self.assertNotIn('</script><script>alert(1)',html)
            self.assertIn('&lt;/script&gt;',html)
            data=json.loads(html.split('<script id="report-data" type="application/json">')[1].split('</script>')[0])
            self.assertEqual(data['faces'][0]['crop'],'crops/p_f001.jpg')
            self.assertNotIn('https://invalid.example',html)
            self.assertIn('id="viewer"',html);self.assertIn('id="comparison"',html)

    def test_empty_results_and_invalid_ids(self):
        with tempfile.TemporaryDirectory() as d:
            render_html(Path(d),[],[])
            html=(Path(d)/'index.html').read_text(encoding='utf-8')
            self.assertIn('"faces": []',html)
            with self.assertRaisesRegex(ValueError,'ID'):
                render_html(Path(d),[],[dict(face_id='../escape',photo_id='p')])
