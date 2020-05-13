import pytest

import numpy as np

from extra_foam.algorithms.spectrum import compute_spectrum_1d


class TestSpectrum:
    def testComputeSpectrum1D(self):
        # test data with different length
        with pytest.raises(ValueError):
            compute_spectrum_1d([], [1, 2], n_bins=5)

        # test empty array
        stats, edges, counts = compute_spectrum_1d([], [], n_bins=5)
        np.testing.assert_array_equal(np.full((5,), np.nan), stats)
        np.testing.assert_array_equal(np.full((5,), np.nan), edges)
        np.testing.assert_array_equal(np.full((5,), np.nan), counts)

        # test nan_to_num
        stats, centers, counts = compute_spectrum_1d([], [], n_bins=5, nan_to_num=True)
        np.testing.assert_array_equal(np.full((5,), 0), stats)
        np.testing.assert_array_equal(np.full((5,), np.nan), centers)
        np.testing.assert_array_equal(np.full((5,), 0), counts)

        # test edge_to_center
        stats, edges, counts = compute_spectrum_1d(
            [], [], n_bins=5, nan_to_num=True, edge2center=False)
        np.testing.assert_array_equal(np.full((5,), 0), stats)
        np.testing.assert_array_equal(np.full((6,), np.nan), edges)
        np.testing.assert_array_equal(np.full((5,), 0), counts)

        # test valid data
        x = np.arange(10)
        y = np.ones(10)

        stats, centers, counts = compute_spectrum_1d(x, y, n_bins=5)
        np.testing.assert_array_equal(np.ones(5), stats)
        np.testing.assert_array_almost_equal([0.9, 2.7, 4.5, 6.3, 8.1], centers)
        np.testing.assert_array_equal(2 * np.ones(5), counts)

        _, edges, _ = compute_spectrum_1d(x, y, n_bins=5, edge2center=False)
        np.testing.assert_array_almost_equal([0., 1.8, 3.6, 5.4, 7.2, 9.], edges)
