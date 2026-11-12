# 역할: 여러 크기 입력 중복 제거·원본 좌표·캐시 서명 분리 검사
import unittest
from unittest.mock import patch
import numpy as np
from face_lab.detection import multiscale_detect


class Detector:
    def __init__(self): self.sizes = []
    def setInputSize(self, size): self.sizes.append(size)
    def detect(self, image):
        scale = image.shape[1]/200
        row = np.array([20, 20, 50, 60, 30, 35, 55, 35, 45, 50, 32, 65, 55, 65, .9], np.float32)
        row[:14] *= scale
        return None, np.array([row])


class MultiscaleTests(unittest.TestCase):
    def test_mapping_duplicate_inputs_and_global_nms(self):
        detector = Detector()
        rows, inputs = multiscale_detect(detector, np.zeros((100, 200, 3), np.uint8), [100, 200, 400, 100], .3)
        self.assertEqual(detector.sizes, [(100,50), (200,100)])
        self.assertEqual(len(rows), 1)
        np.testing.assert_allclose(rows[0][:4], [20,20,50,60])
        np.testing.assert_allclose(rows[0][4:14], [30,35,55,35,45,50,32,65,55,65])
        self.assertEqual(len(inputs), 2)



if __name__ == '__main__': unittest.main()
