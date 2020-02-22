"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import numpy as np

from .sampling import slice_curve


def normalize_auc(y, x, auc_range=None):
    """Normalize a curve a given area under the curve (AUC).

    :param numpy.ndarray y: 1D array.
    :param numpy.ndarray x: 1D array.
    :param None/tuple auc_range: x range for calculating AUC.

    :return numpy.ndarray: the normalized y.

    :raise ValueError
    """
    # if y contains only 0 (np.any() is much faster to np.count_nonzero())
    if not np.any(y):
        return np.copy(y)

    # get the integration
    if auc_range is None:
        integ = np.trapz(*slice_curve(y, x))
    else:
        integ = np.trapz(*slice_curve(y, x, *auc_range))

    if integ == 0:
        raise ValueError("Normalized by 0!")

    return y / integ


def find_actual_range(arr, range):
    """Find the actual range for an array of data.

    This is a helper function to find the non-infinite range (lb, ub) as
    input for some other functions.

    :param numpy.ndarray arr: data.
    :param tuple range: desired range.

    :return tuple: actual range.
    """
    v_min, v_max = range
    assert v_min < v_max

    if not np.isfinite(v_min) and not np.isfinite(v_max):
        v_min, v_max = np.min(arr), np.max(arr)
        if v_min == v_max:
            # np.histogram convention
            v_min = v_min - 0.5
            v_max = v_max + 0.5
    elif not np.isfinite(v_max):
        v_max = np.max(arr)
        if v_max <= v_min:
            # this could happen when v_max is +Inf while v_min is finite
            v_max = v_min + 1.0  # must have v_max > v_min
    elif not np.isfinite(v_min):
        v_min = np.min(arr)
        if v_min >= v_max:
            # this could happen when v_min is -Inf while v_max is finite
            v_min = v_max - 1.0  # must have v_max > v_min

    return v_min, v_max


def compute_statistics(arr):
    """Compute statistics of an array.

    :TODO: optimize
    """
    if len(arr) == 0:
        # suppress runtime warning
        return np.nan, np.nan, np.nan
    return np.mean(arr), np.median(arr), np.std(arr)
