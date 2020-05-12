import unittest

import numpy as np

from extra_foam.algorithms import (
    edge_detect, fourier_transform_2d
)


class TestEdgeDetect(unittest.TestCase):
    def testGeneral(self):
        img = np.ones((6, 8), dtype=np.float32)
        edge_detect(img)


class TestFourierTransform(unittest.TestCase):
    def testGeneral(self):
        img = np.ones((6, 8), dtype=np.float32)
        fourier_transform_2d(img)
