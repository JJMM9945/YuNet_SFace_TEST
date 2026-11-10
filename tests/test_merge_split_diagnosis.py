import unittest

from tests.experiments.face_lab.merge_split_diagnosis import accepted_merges, choose_development_threshold


def pair(score, same, same_photo=False):
    return dict(group_a='a', group_b='b', score=score, same_identity=same, same_photo=same_photo)


class MergeSplitDiagnosisTests(unittest.TestCase):
    def test_same_photo_veto(self):
        rows = [pair(.5, False, True)]
        self.assertEqual(accepted_merges(rows, .4, True), [])
        self.assertEqual(len(accepted_merges(rows, .4, False)), 1)

    def test_development_selection_prioritizes_zero_wrong_merges(self):
        rows = [pair(.39, True), pair(.36, False)]
        selected = choose_development_threshold(rows)
        self.assertEqual(selected['wrong_merges'], 0)
        self.assertEqual(selected['correct_merges'], 1)
        self.assertGreater(selected['threshold'], .36)

