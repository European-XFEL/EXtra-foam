import unittest

import numpy as np

from karaboFAI.algorithms import compute_spectrum


class TestXas(unittest.TestCase):
    def test_computeSpectrum(self):
        # empty data
        bin_center, absorptions, bin_count = compute_spectrum(
            [], [], [[], []], 1)
        np.testing.assert_array_equal(np.array([]), bin_center)
        self.assertListEqual([], absorptions)
        np.testing.assert_array_equal(np.array([]), bin_count)

        # test raises

        with self.assertRaisesRegex(ValueError, "> 0"):
            compute_spectrum([1], [1], [[2], [2]], 0)

        with self.assertRaisesRegex(ValueError, "different lengths"):
            compute_spectrum([1], [1, 2], [[2], [3]], 1)

        with self.assertRaisesRegex(ValueError, "different lengths"):
            compute_spectrum([1], [1], [[2, 3], [3]], 1)

        with self.assertRaisesRegex(TypeError, "Not understandable"):
            compute_spectrum(1, [1], [[2, 3], [3]], 1)

        with self.assertRaisesRegex(TypeError, "Not understandable"):
            compute_spectrum([1], [1], [3], 1)

        # test spectrum

        # assign data
        n_pts = 20
        n_bins = 5
        energy = [i for i in range(n_pts)]
        I0 = [i+1 for i in range(n_pts)]
        I1 = [np.e * (i+1) for i in range(n_pts)]
        I2 = [np.e ** 2 * (i+1) for i in range(n_pts)]

        bin_center, absorptions, bin_count = compute_spectrum(
            energy, I0, [I1, I2], n_bins)

        bin_center_gt = [1.9, 5.7, 9.5, 13.3, 17.1]
        absorptions_gt = [[-1] * 5, [-2] * 5]
        bin_count_gt = [4] * 5

        np.testing.assert_array_almost_equal(bin_center_gt, bin_center)
        np.testing.assert_array_almost_equal(absorptions_gt[0], absorptions[0])
        np.testing.assert_array_almost_equal(absorptions_gt[1], absorptions[1])
        np.testing.assert_array_equal(bin_count_gt, bin_count)
