import unittest

from karaboFAI.algorithms import intersection


class TestGeometry(unittest.TestCase):
    def test_interaction(self):
        # one contains the other
        self.assertListEqual(list(intersection(100, 80, 0, 0, 50, 30, 0, 0)),
                             [50, 30, 0, 0])
        self.assertListEqual(list(intersection(50, 30, 0, 0, 100, 80, 0, 0)),
                             [50, 30, 0, 0])
        self.assertListEqual(list(intersection(10, 5, 5, 2, 50, 50, 0, 0)),
                             [10, 5, 5, 2])
        self.assertListEqual(list(intersection(50, 50, 0, 0, 10, 5, 5, 2)),
                             [10, 5, 5, 2])

        # no interaction
        self.assertListEqual(list(intersection(100, 100, 0, 0, 5, 5, -10, -10)),
                             [-5, -5, 0, 0])
        self.assertListEqual(list(intersection(5, 5, -10, -10, 100, 100, 0, 0)),
                             [-5, -5, 0, 0])

        # partially intersect
        self.assertListEqual(list(intersection(10, 10, 0, 0, 15, 15, -10, -10)),
                             [5, 5, 0, 0])
        self.assertListEqual(list(intersection(15, 15, -10, -10, 10, 10, 0, 0)),
                             [5, 5, 0, 0])

        self.assertListEqual(list(intersection(10, 10, 1, 1, 15, 15, 5, 10)),
                             [6, 1, 5, 10])
        self.assertListEqual(list(intersection(15, 15, 5, 10, 10, 10, 1, 1)),
                             [6, 1, 5, 10])

        self.assertListEqual(list(intersection(10, 20, 0, 0, 4, 24, 2, -2)),
                             [4, 20, 2, 0])
        self.assertListEqual(list(intersection(4, 24, 2, -2, 10, 20, 0, 0)),
                             [4, 20, 2, 0])
