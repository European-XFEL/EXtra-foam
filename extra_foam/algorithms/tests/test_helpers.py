import unittest

from extra_foam.algorithms import intersection


class TestGeometry(unittest.TestCase):
    def test_interaction(self):
        # one contains the other
        self.assertListEqual(intersection([0, 0, 100, 80], [0, 0, 50, 30]),
                             [0, 0, 50, 30])
        self.assertListEqual(intersection([0, 0, 50, 30], [0, 0, 100, 80]),
                             [0, 0, 50, 30])
        self.assertListEqual(intersection([5, 2, 10, 5], [0, 0, 50, 50]),
                             [5, 2, 10, 5])
        self.assertListEqual(intersection([0, 0, 50, 50], [5, 2, 10, 5]),
                             [5, 2, 10, 5])

        # no interaction
        self.assertListEqual(intersection([0, 0, 100, 100], [-10, -10, 5, 5]),
                             [0, 0, -5, -5])
        self.assertListEqual(intersection([-10, -10, 5, 5], [0, 0, 100, 100]),
                             [0, 0, -5, -5])

        # partially intersect
        self.assertListEqual(intersection([0, 0, 10, 10], [-10, -10, 15, 15]),
                             [0, 0, 5, 5])
        self.assertListEqual(intersection([-10, -10, 15, 15], [0, 0, 10, 10]),
                             [0, 0, 5, 5])

        self.assertListEqual(intersection([1, 1, 10, 10], [5, 10, 15, 15]),
                             [5, 10, 6, 1])
        self.assertListEqual(intersection([5, 10, 15, 15], [1, 1, 10, 10]),
                             [5, 10, 6, 1])

        self.assertListEqual(intersection([0, 0, 10, 20], [2, -2, 4, 24]),
                             [2, 0, 4, 20])
        self.assertListEqual(intersection([2, -2, 4, 24], [0, 0, 10, 20]),
                             [2, 0, 4, 20])
