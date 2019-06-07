"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

Enhancement of numpy.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from concurrent.futures import ThreadPoolExecutor

import numpy as np


def nanmean_axis0_para(data, *, chunk_size=10, max_workers=4):
    """Parallel implementation of numpy.nanmean.

    :param numpy.ndarray data: array.
    :param int chunk_size: the slice size of along the second dimension
        of the input data.
    :param int max_workers: The maximum number of threads that can be
        used to execute the given calls.

    :return numpy.ndarray: averaged input data along the first axis if
        the dimension of input data is larger than 3, otherwise the
        original data.
    """
    if data.ndim < 3:
        return data

    def nanmean_imp(out, start, end):
        """Implementation of parallelized nanmean.

        :param numpy.ndarray out: result 2D array. (x, y)
        :param int start: start index
        :param int end: end index (not included)
        """
        with np.warnings.catch_warnings():
            np.warnings.filterwarnings('ignore', category=RuntimeWarning)

            out[start:end, :] = np.nanmean(data[:, start:end, :], axis=0)

    ret = np.zeros_like(data[0, ...])

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        start = 0
        max_idx = data.shape[1]
        while start < max_idx:
            executor.submit(nanmean_imp,
                            ret, start, min(start + chunk_size, max_idx))
            start += chunk_size

    return ret


def mask_by_threshold(data, a_min=-np.inf, a_max=np.inf, inplace=False):
    """Mask an array by threshold.

    :param numpy.ndarray data: array to be masked.
    :param float a_min: lower boundary of the threshold mask.
    :param float a_max: upper boundary of the threshold mask.
    :param bool inplace: True for apply the mask in-place.

    :return numpy.ndarray: masked data.
    """
    if not inplace:
        masked = data.copy()
    else:
        masked = data

    # Convert 'nan' to '-inf' and it will later be converted to the
    # lower range of mask, which is usually 0.
    # We do not convert 'nan' to 0 because: if the lower range of
    # mask is a negative value, 0 will be converted to a value
    # between 0 and 255 later.
    masked[np.isnan(masked)] = -np.inf
    # clip the array, which now will contain only numerical values
    # within the mask range
    np.clip(masked, a_min, a_max, out=masked)

    return masked
