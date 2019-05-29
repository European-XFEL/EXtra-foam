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
    """Parallel implementation of nanmean.

    :param numpy.ndarray x: 3D data array. (pulse indices, x, y)
    :param int chunk_size: the slice size of along the second dimension
        of the input data.
    :param int max_workers: The maximum number of threads that can be
        used to execute the given calls.
    :return:
    """
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
