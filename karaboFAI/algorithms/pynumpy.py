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
    def nanmean_imp(out, start, end):
        """Implementation of parallelized nanmean.

        :param numpy.ndarray out: result 2D array. (x, y)
        :param int start: start index
        :param int end: end index (not included)
        """
        with np.warnings.catch_warnings():
            np.warnings.filterwarnings('ignore', category=RuntimeWarning)

            out[start:end, :] = np.nanmean(data[:, start:end, :], axis=0)

    if data.ndim < 3:
        ret = data
    else:
        ret = np.zeros_like(data[0, ...])

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            start = 0
            max_idx = data.shape[1]
            while start < max_idx:
                executor.submit(nanmean_imp,
                                ret, start, min(start + chunk_size, max_idx))
                start += chunk_size

    return ret


def mask_image(image, *,
               threshold_mask=None,
               image_mask=None,
               inplace=False):
    """Mask an array by threshold.

    The masked pixel value will will be set to 0.

    :param numpy.ndarray image: array to be masked.
    :param tuple/None threshold_mask: (min, max) of the threshold mask.
    :param numpy.ndarray/None image_mask: image mask. It assumes the shape
        of the image_mask is the same as the image.
    :param bool inplace: True for apply the mask in-place.

    :return numpy.ndarray: masked data.
    """
    if not inplace:
        masked = image.copy()
    else:
        masked = image

    if image_mask is not None:
        masked[image_mask] = 0

    # it is reasonable to set NaN to zero after nanmean
    masked[np.isnan(masked)] = 0

    if threshold_mask is not None:
        a_min, a_max = threshold_mask
        masked[(masked > a_max) | (masked < a_min)] = 0

    return masked
