# 역할: EXIF·좌표·품질·군집·HTML 안전성과 정답 평가의 기본 회귀 검사
import csv
import json
import tempfile
import unittest
from argparse import Namespace
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image
from face_lab.grouping import group_faces
from face_lab.preprocess import decode_oriented, detection_input, to_original
from face_lab.models import verify, MODELS


class CoreTests(unittest.TestCase):
    def test_orientation_and_unicode_path(self):
        image = Image.new("RGB", (20, 10), "red")
        exif = image.getexif()
        exif[274] = 6
        stream = BytesIO()
        image.save(stream, format="JPEG", exif=exif)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "한글 공백 사진.jpg"
            path.write_bytes(stream.getvalue())
            bgr = decode_oriented(path.read_bytes())
        self.assertEqual(bgr.shape, (20,10,3))
        self.assertGreater(int(bgr[0,0,2]), 240)
        self.assertLess(int(bgr[0,0,0]), 10)

    def test_resize_coordinates(self):
        image = np.zeros((1001,2003,3), np.uint8)
        small, sx, sy = detection_input(image, 640)
        self.assertEqual(small.shape[1], 640)
        row = np.array([10,20,30,40,10,20,30,40,50,60,70,80,90,100,.95], np.float32)
        mapped = to_original(row, sx, sy)
        self.assertAlmostEqual(mapped[0], 10*2003/640, places=4)
        self.assertAlmostEqual(mapped[13], 100*1001/small.shape[0], places=4)
        self.assertEqual(mapped[14], row[14])

    def test_complete_linkage_breaks_chain(self):
        # A-B=.940 B-C=.766 A-C=.5: complete linkage must break this chain.
        angles = np.deg2rad([0,20,60])
        vectors = {str(i): np.array([np.cos(t),np.sin(t)], np.float32) for i,t in enumerate(angles)}
        faces = [dict(face_id=str(i), photo_id=str(i), quality_reason="") for i in range(3)]
        group_faces(faces, vectors, .75, 2, 0)
        grouped = [f for f in faces if f["group_id"]]
        self.assertEqual(len(grouped), 2)
        self.assertEqual(sum(f["status"] == "unknown" for f in faces), 1)

    def test_same_photo_and_quality(self):
        faces = [dict(face_id=str(i), photo_id="same", quality_reason="small_face" if i==2 else "") for i in range(3)]
        vectors = {str(i): np.array([1,0], np.float32) for i in range(3)}
        group_faces(faces, vectors, .45, 2, .03)
        self.assertEqual(faces[0]["review"], "same_photo_same_group")
        self.assertEqual(faces[2]["reason"], "small_face")
        self.assertEqual(faces[2]["status"], "unknown")

    def test_pointer_and_html_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/"bad.onnx"
            for body in (b"version https://git-lfs.github.com/spec/v1", b"<html>error</html>"):
                path.write_bytes(body)
                with self.assertRaisesRegex(ValueError, "Invalid model"):
                    verify(path, MODELS["yunet"])

    def test_ambiguous_face_rejection(self):
        angles = np.deg2rad([0,30,60])
        vectors = {str(i): np.array([np.cos(t),np.sin(t)],np.float32) for i,t in enumerate(angles)}
        faces = [dict(face_id=str(i),photo_id=str(i),quality_reason="") for i in range(3)]
        group_faces(faces,vectors,.8,2,.03)
        self.assertTrue(any(f["reason"]=="ambiguous_competing_cluster" for f in faces))
        self.assertTrue(all(f["status"]=="unknown" for f in faces))

    def test_single_cross_edge_does_not_reject_coherent_groups(self):
        # Boundary faces are close, but neither fits ALL faces of the competing
        # cluster. Strong single cross edges must not override complete linkage.
        angles=np.deg2rad([0,5,25,50,70,75])
        vectors={str(i):np.array([np.cos(t),np.sin(t)],np.float32) for i,t in enumerate(angles)}
        faces=[dict(face_id=str(i),photo_id=str(i),quality_reason="") for i in range(6)]
        group_faces(faces,vectors,.7,2,.03)
        self.assertTrue(all(f["status"]=="grouped" for f in faces))
        self.assertEqual(len({f["group_id"] for f in faces}),2)
        self.assertEqual(faces[0]["group_id"],faces[2]["group_id"])
        self.assertNotEqual(faces[2]["group_id"],faces[3]["group_id"])

    def test_grouping_diagnostics_preserve_predictions_and_group_scores(self):
        import copy
        angles=np.deg2rad([0,5,25,50,70,75])
        vectors={str(i):np.array([np.cos(t),np.sin(t)],np.float32) for i,t in enumerate(angles)}
        original=[dict(face_id=str(i),photo_id=str(i),quality_reason='') for i in range(6)]
        traced=copy.deepcopy(original);plain=copy.deepcopy(original);diagnostics={}
        group_faces(plain,vectors,.7,2,.03)
        group_faces(traced,vectors,.7,2,.03,diagnostics)
        self.assertEqual(plain,traced)
        self.assertEqual(len(diagnostics),6)
        record=diagnostics['2']
        self.assertAlmostEqual(record['own_min_similarity'],np.cos(np.deg2rad(25)),places=6)
        self.assertAlmostEqual(record['competing_min_similarity'],np.cos(np.deg2rad(50)),places=6)
        self.assertGreater(record['competitors'][0]['max_similarity'],record['competing_min_similarity'])
        self.assertFalse(record['ambiguity_rule_matched'])



if __name__ == "__main__":
    unittest.main()
