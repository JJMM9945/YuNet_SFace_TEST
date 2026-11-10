import unittest

from tests.experiments.face_lab.quality_diagnosis import size_band


class QualityDiagnosisTests(unittest.TestCase):
    def test_size_band_boundaries(self):
        self.assertEqual(size_band(59.9), '0–60px')
        self.assertEqual(size_band(60), '60–100px')
        self.assertEqual(size_band(100), '100–200px')
        self.assertEqual(size_band(200), '200px 이상')

    def test_invalid_boundaries(self):
        with self.assertRaises(ValueError):
            size_band(50, (100, 60))

    def test_custom_boundaries(self):
        self.assertEqual(size_band(99, (100, 300)), '0–100px')
        self.assertEqual(size_band(300, (100, 300)), '300px 이상')

