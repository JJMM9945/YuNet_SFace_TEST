"""Current fixed detector and photo-album settings."""
import unittest
from dataclasses import FrozenInstanceError
from face_lab.engine import Settings


class SelectedDefaultsTests(unittest.TestCase):
    def test_selected_multiscale_average_defaults(self):
        settings = Settings()
        self.assertEqual(settings.detection_sides, (960,1600,2400))
        self.assertEqual(settings.score_threshold, .75)
        self.assertEqual((settings.linkage,settings.similarity_threshold,
                          settings.ambiguity_margin,settings.min_group_size), ('average',.4,.01,1))

    def test_detection_score_cannot_be_changed(self):
        with self.assertRaises(TypeError):
            Settings(score_threshold=.8)
        settings = Settings()
        with self.assertRaises(FrozenInstanceError):
            settings.score_threshold = .8
