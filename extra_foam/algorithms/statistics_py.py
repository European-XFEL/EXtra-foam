"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import numpy as np

from .imageproc_py import mask_image_data
from .statistics import nanmean, nansum


def find_actual_range(arr, range):
    """Find the actual range for an array of data.

    This is a helper function to find the non-infinite range (lb, ub) as
    input for some other functions.

    :param numpy.ndarray arr: data.
    :param tuple range: desired range.

    :return tuple: actual range.

    Note: the input data is assume to be non-free.

    """
    v_min, v_max = range
    assert v_min < v_max

    if not np.isfinite(v_min) and not np.isfinite(v_max):
        if arr.size == 0:
            v_min, v_max = 0., 0.
        else:
            v_min, v_max = np.min(arr), np.max(arr)

        if v_min == v_max:
            # np.histogram convention
            v_min = v_min - 0.5
            v_max = v_max + 0.5
    elif not np.isfinite(v_max):
        if arr.size == 0:
            v_max = v_min + 1.0
        else:
            v_max = np.max(arr)
            if v_max <= v_min:
                # this could happen when v_max is +Inf while v_min is finite
                v_max = v_min + 1.0  # must have v_max > v_min
    elif not np.isfinite(v_min):
        if arr.size == 0:
            v_min = v_max - 1.0
        else:
            v_min = np.min(arr)
            if v_min >= v_max:
                # this could happen when v_min is -Inf while v_max is finite
                v_min = v_max - 1.0  # must have v_max > v_min

    return v_min, v_max


def compute_statistics(data):
    """Compute statistics of an array.

    :param numpy.ndarray data: input array.
    """
    if len(data) == 0:
        # suppress runtime warning
        return np.nan, np.nan, np.nan
    return np.mean(data), np.median(data), np.std(data)


def nanhist_with_stats(roi, bin_range=(-np.inf, np.inf), n_bins=10):
    """Compute nan-histogram and nan-statistics of an array.

    :param numpy.ndarray roi: image ROI.
    :param tuple bin_range: (lb, ub) of histogram.
    :param int n_bins: number of bins of histogram.
    """
    # Note: Since the nan functions in numpy is typically 5-8 slower
    # than the non-nan counterpart, it is always faster to remove nan
    # first, which results in a copy, and then calculate the statistics.

    # TODO: the following three steps can be merged into one to improve
    #       the performance.
    filtered = roi.copy()
    mask_image_data(filtered, threshold_mask=bin_range, keep_nan=True)
    filtered = filtered[~np.isnan(filtered)]

    actual_range = find_actual_range(filtered, bin_range)
    hist, bin_edges = np.histogram(filtered, range=actual_range, bins=n_bins)
    bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2.0
    mean, median, std = compute_statistics(filtered)

    return hist, bin_centers, mean, median, std


def hist_with_stats(data, bin_range=(-np.inf, np.inf), n_bins=10):
    """Compute histogram and statistics of an array.

    :param numpy.ndarray data: input data.
    :param tuple bin_range: (lb, ub) of histogram.
    :param int n_bins: number of bins of histogram.
    """
    v_min, v_max = find_actual_range(data, bin_range)

    filtered = data[(data >= v_min) & (data <= v_max)]
    hist, bin_edges = np.histogram(
        filtered, bins=n_bins, range=(v_min, v_max))
    bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2.0
    mean, median, std = compute_statistics(filtered)

    return hist, bin_centers, mean, median, std
