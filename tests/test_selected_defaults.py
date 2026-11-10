# 역할: 여러 크기 .75·average .40 기본값과 단일 크기 명시 동작 검사
import unittest
from tests.experiments.face_lab.__main__ import parser


class SelectedDefaultsTests(unittest.TestCase):
    def test_selected_multiscale_average_defaults(self):
        a=parser().parse_args(['analyze'])
        self.assertEqual(a.detection_sides,[960,1600,2400])
        self.assertEqual(a.score_threshold,.75)
        self.assertEqual((a.linkage,a.similarity_threshold,a.ambiguity_margin,a.min_group_size),('average',.4,.01,1))
        self.assertFalse(a.reassign_unknown)

    def test_single_scale_baseline_remains_explicitly_selectable(self):
        a=parser().parse_args(['analyze','--detection-sides','1600','--score-threshold','.8','--linkage','complete'])
        self.assertEqual(a.detection_sides,[1600]);self.assertEqual(a.score_threshold,.8);self.assertEqual(a.linkage,'complete')
