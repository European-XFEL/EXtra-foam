import pytest

import math

import numpy as np

from extra_foam.algorithms.statistics_py import (
    hist_with_stats, nanhist_with_stats, compute_statistics, _get_outer_edges,
    nanmean, nansum, quick_min_max
)


class TestStatistics:
    def testQuickMinMax(self):
        # test array size <= 1e5
        arr = np.array([[np.nan, 1, 2, 3, 4], [5, 6, 7, 8, np.nan]])
        assert quick_min_max(arr) == (1., 8.)
        assert quick_min_max(arr, q=1.0) == (1, 8)
        assert quick_min_max(arr, q=0.9) == (2, 7)
        assert quick_min_max(arr, q=0.7) == (3, 6)
        assert quick_min_max(arr, q=0.3) == (3, 6)

        with pytest.raises(ValueError):
            quick_min_max(arr, q=1.1)

        # test array size > 1e5
        arr = np.ones((1000, 1000), dtype=np.float32)
        assert quick_min_max(arr) == (1, 1)
        assert quick_min_max(arr, q=0.9) == (1, 1)

        arr[::3] = 2
        assert quick_min_max(arr) == (1., 2.)
        assert quick_min_max(arr, q=0.9) == (1, 2)

    def _assert_array_almost_equal(self, a, b):
        np.testing.assert_array_almost_equal(a, b)
        if isinstance(a, np.ndarray):
            assert a.dtype == b.dtype

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    @pytest.mark.parametrize("f_cpp, f_py", [(nanmean, np.nanmean), (nansum, np.nansum)])
    def testCppStatistics(self, f_cpp, f_py, dtype):
        a1d = np.array([np.nan, 1, 2], dtype=dtype)
        a2d = np.array([[np.nan, 1, 2], [3, 6, np.nan]], dtype=dtype)
        a3d = np.array([[[np.nan, np.nan,      2], [3, 6, np.nan]],
                        [[     1,      4, np.nan], [6, 3, np.nan]]], dtype=dtype)
        a4d = np.ones((2, 3, 4, 5), dtype=dtype)
        a4d[:, ::2, ::3, ::4] = np.nan
        a5d = np.ones((1, 2, 3, 4, 5), dtype=dtype)
        a5d[:, ::2, :2, ::3, ::4] = np.nan

        with np.warnings.catch_warnings():
            np.warnings.simplefilter("ignore", category=RuntimeWarning)

            # axis is None
            self._assert_array_almost_equal(f_py(a1d), f_cpp(a1d))
            self._assert_array_almost_equal(f_py(a2d), f_cpp(a2d))
            self._assert_array_almost_equal(f_py(a3d), f_cpp(a3d))
            self._assert_array_almost_equal(f_py(a4d), f_cpp(a4d))
            self._assert_array_almost_equal(f_py(a5d), f_cpp(a5d))

            # axis = 0
            self._assert_array_almost_equal(f_py(a3d, axis=0), f_cpp(a3d, axis=0))
            self._assert_array_almost_equal(f_py(a4d, axis=0), f_cpp(a4d, axis=0))

            # axis = (1, 2)
            self._assert_array_almost_equal(f_py(a3d, axis=(-2, -1)), f_cpp(a3d, axis=(-2, -1)))
            self._assert_array_almost_equal(f_py(a4d, axis=(-2, -1)), f_cpp(a4d, axis=(-2, -1)))

    def testNanhistWithStats(self):
        # case 1
        roi = np.array([[np.nan, 1, 2], [3, 6, np.nan]], dtype=np.float32)
        hist, bin_centers, mean, median, std = nanhist_with_stats(roi, (1, 3), 4)
        np.testing.assert_array_equal([1, 0, 1, 1], hist)
        np.testing.assert_array_equal([1.25, 1.75, 2.25, 2.75], bin_centers)
        arr_gt = [1, 2, 3]
        assert np.mean(arr_gt) == mean
        assert np.median(arr_gt) == median
        assert np.std(arr_gt) == pytest.approx(std)

        # case 2 (the actual array is empty after filtering)
        roi = np.array([[np.nan, 0, 0], [0, 0, np.nan]], dtype=np.float32)
        hist, bin_centers, mean, median, std = nanhist_with_stats(roi, (1, 3), 4)
        np.testing.assert_array_equal([0, 0, 0, 0], hist)
        assert np.isnan(mean)
        assert np.isnan(median)
        assert np.isnan(std)

        # case 3 (elements in the actual array have the same value)
        roi = np.array([[1, 0, np.nan], [1, np.nan, 0]], dtype=np.float32)
        hist, bin_centers, mean, median, std = nanhist_with_stats(roi, (1e-6, 3), 4)
        np.testing.assert_array_equal([0, 2, 0, 0], hist)
        assert 1 == mean
        assert 1 == median
        assert 0 == std

        # case 4 (3D input)
        roi = np.array([[[np.nan, 1, 2], [3, 6, np.nan]],
                        [[np.nan, 0, 1], [2, 5, np.nan]]], dtype=np.float32)
        hist, bin_centers, mean, median, std = nanhist_with_stats(roi, (1, 3), 4)
        np.testing.assert_array_equal([2, 0, 2, 1], hist)
        np.testing.assert_array_equal([1.25, 1.75, 2.25, 2.75], bin_centers)
        arr_gt = [1, 1, 2, 2, 3]
        assert np.mean(arr_gt) == pytest.approx(mean)
        assert np.median(arr_gt) == median
        assert np.std(arr_gt) == pytest.approx(std)

        # case 5 (finite outer edges cannot be found)
        roi = np.array([[-np.inf, np.nan, 3], [np.nan, 5, 6]], dtype=np.float32)
        with pytest.raises(ValueError):
            nanhist_with_stats(roi, (-np.inf, np.inf), 4)

    def testHistWithStats(self):
        data = np.array([0, 1, 2, 3, 6, 0], dtype=np.float32)  # 1D
        hist, bin_centers, mean, median, std = hist_with_stats(data, (1, 3), 4)
        np.testing.assert_array_equal([1, 0, 1, 1], hist)
        np.testing.assert_array_equal([1.25, 1.75, 2.25, 2.75], bin_centers)
        arr_gt = [1, 2, 3]
        assert np.mean(arr_gt) == mean
        assert np.median(arr_gt) == median
        assert np.std(arr_gt) == pytest.approx(std)

        # case 2 (empty input)
        data = np.array([[[]]], dtype=np.float32)  # 3D
        hist, bin_centers, mean, median, std = hist_with_stats(data, (1, 3), 2)
        np.testing.assert_array_equal([0, 0], hist)
        np.testing.assert_array_equal([1.5, 2.5], bin_centers)
        assert np.isnan(mean)
        assert np.isnan(median)
        assert np.isnan(std)

        # case 3 (elements have the same value)
        roi = np.array([[1, 1, 1], [1, 1, 1]], dtype=np.float32)  # 2D
        hist, bin_centers, mean, median, std = hist_with_stats(roi, (0, 3), 4)
        np.testing.assert_array_equal([0, 6, 0, 0], hist)
        np.testing.assert_array_equal([0.375, 1.125, 1.875, 2.625], bin_centers)
        assert 1 == mean
        assert 1 == median
        assert 0 == std

        # case 4 (finite outer edges cannot be found)
        roi = np.array([[np.inf, 2, 3], [4, 5, 6]], dtype=np.float32)
        with pytest.raises(ValueError):
            hist_with_stats(roi, (-np.inf, np.inf), 4)

    def testFindActualRange(self):
        arr = np.array([1, 2, 3, 4])
        assert (-1.5, 2.5) == _get_outer_edges(arr, (-1.5, 2.5))

        arr = np.array([1, 2, 3, 4])
        assert (1, 4) == _get_outer_edges(arr, (-math.inf, math.inf))

        arr = np.array([1, 1, 1, 1])
        assert (0.5, 1.5) == _get_outer_edges(arr, (-math.inf, math.inf))

        arr = np.array([1, 2, 3, 4])
        assert (3, 4) == _get_outer_edges(arr, (3, math.inf))

        arr = np.array([1, 2, 3, 4])
        assert (4, 5) == _get_outer_edges(arr, (4, math.inf))

        arr = np.array([1, 2, 3, 4])
        assert (5, 6) == _get_outer_edges(arr, (5, math.inf))

        arr = np.array([1, 2, 3, 4])
        assert (0, 1) == _get_outer_edges(arr, (-math.inf, 1))

        arr = np.array([1, 2, 3, 4])
        assert (-1, 0) == _get_outer_edges(arr, (-math.inf, 0))

        arr = np.array([1, 2, 3, 4])
        assert (-1, 0) == _get_outer_edges(arr, (-math.inf, 0))

        arr = np.array([1, 2, 3, -np.inf])
        assert (-np.inf, 0) == _get_outer_edges(arr, (-math.inf, 0))

        arr = np.array([1, 2, 3, np.inf])
        assert (0, np.inf) == _get_outer_edges(arr, (0, math.inf))

        arr = np.array([1, -np.inf, 3, np.inf])
        assert (-np.inf, np.inf) == _get_outer_edges(arr, (-math.inf, math.inf))

    def testComputeStatistics(self):
        with np.warnings.catch_warnings():
            np.warnings.simplefilter("ignore", category=RuntimeWarning)

            # test input contains only Nan
            data = np.empty((3, 2), dtype=np.float32)
            data.fill(np.nan)
            assert all(np.isnan(x) for x in compute_statistics(data))

        data = np.array([])
        assert all(np.isnan(x) for x in compute_statistics(data))

        data = np.array([1, 1, 2, 1, 1])
        assert (1.2, 1.0, 0.4) == compute_statistics(data)
