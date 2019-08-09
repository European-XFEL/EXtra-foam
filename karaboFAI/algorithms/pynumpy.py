"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

Enhancement of numpy.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import numpy as np


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
