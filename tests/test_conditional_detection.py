# 역할: 조건부 검사 경계·기존 상자/랜드마크 보존·중복 제거·특징 추출 시점 검증
import unittest
from unittest.mock import patch
import numpy as np
from face_lab.detection import detect_with_policy
from face_lab.engine import FaceEngine, Settings


def row(x, score=.9):
    return np.array([x, 0, 10, 12, x+2, 3, x+8, 3, x+5, 6, x+3, 9, x+7, 9, score], np.float32)


class ConditionalDetectionTests(unittest.TestCase):
    def test_zero_one_two_three_trigger_boundary(self):
        for count in (0, 1, 2, 3):
            with self.subTest(count=count):
                primary = [row(i*30) for i in range(count)]
                with patch('face_lab.detection.multiscale_detect', side_effect=[(primary, []), ([row(200)], [])]) as detect:
                    kept, trace = detect_with_policy(None, None, Settings())
                expected = count == 0 or count >= 3
                self.assertEqual(detect.call_count, 2 if expected else 1)
                self.assertEqual(trace['conditional_triggered'], expected)
                self.assertEqual(len(kept), count + int(expected))

    def test_primary_landmarks_and_lower_score_preserved(self):
        primary = [row(0, .8), row(30), row(60)]
        higher_score_duplicate = row(0, .99)
        higher_score_duplicate[4:14] += 1
        supplemental = [higher_score_duplicate, row(90), row(91, .8)]
        with patch('face_lab.detection.multiscale_detect', side_effect=[(primary, []), (supplemental, [])]):
            kept, trace = detect_with_policy(None, None, Settings())
        self.assertEqual(len(kept), 4)
        self.assertEqual(trace['added_faces'], 1)
        for old, current in zip(primary, kept):
            np.testing.assert_array_equal(old, current)

    def test_explicit_single_or_multiscale_without_conditional(self):
        for sides in ((1280,), (960,1600,2400)):
            with patch('face_lab.detection.multiscale_detect', return_value=([], [])) as detect:
                _, trace = detect_with_policy(None, None, Settings(detection_sides=sides, conditional_detection=False))
            detect.assert_called_once()
            self.assertEqual(detect.call_args.args[2], sides)
            self.assertFalse(trace['conditional_triggered'])
            self.assertEqual(trace['trigger_reason'], 'disabled')

    def test_features_extracted_once_per_final_face(self):
        engine = object.__new__(FaceEngine)
        engine.settings = Settings()
        engine.detector = object(); engine.recognizer = object()
        rows = [row(10), row(40)]
        crop = np.zeros((112,112,3), np.uint8)
        vector = np.ones(128, np.float32) / np.sqrt(128)
        trace = dict(primary_faces=0, conditional_triggered=True, added_faces=2)
        with patch('face_lab.engine.decode_oriented', return_value=np.zeros((100,100,3),np.uint8)), \
             patch('face_lab.engine.detect_with_policy', return_value=(rows, trace)), \
             patch('face_lab.engine.extract', return_value=(crop,vector)) as extract:
            faces, vectors = engine.extract_photo(b'fixture', 'p1')
        self.assertEqual(extract.call_count, 2)
        self.assertEqual(len(faces), 2)
        self.assertEqual(len(vectors), 2)
        self.assertEqual(engine.last_detection, trace)

    def test_invalid_settings_rejected(self):
        for kwargs in ({'detection_sides': ()}, {'supplemental_detection_sides': (0,)}, {'conditional_min_faces': 0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                Settings(**kwargs)
