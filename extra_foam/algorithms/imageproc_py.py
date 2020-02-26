"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>, Ebad Kamil <ebad.kamil@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import numpy as np

from .imageproc import (
    nanmeanImageArray, movingAvgImageData,
    maskImageData, maskNanImageData, maskZeroImageData,
    correctGain, correctOffset, correctGainOffset
)


def nanmean_image_data(data, *, kept=None):
    """Compute nanmean of an array of images of a tuple/list of two images.

    :param tuple/list/numpy.array data: a tuple/list of two 2D arrays, or
        a 2D or 3D numpy array.
    :param None/list kept: indices of the kept images.
    """
    if isinstance(data, (tuple, list)):
        return nanmeanImageArray(*data)

    if data.ndim == 2:
        return data.copy()

    if kept is None:
        return nanmeanImageArray(data)

    return nanmeanImageArray(data, kept)


def correct_image_data(data, *,
                       gain=None,
                       offset=None,
                       slicer=slice(None, None)):
    """Apply gain and/or offset correct to image data.

    :param numpy.array data: image data, Shape = (y, x) or (indices, y, x)
    :param None/numpy.array gain: gain constants, which has the same
        shape as the image data.
    :param None/numpy.array offset: offset constants, which has the same
        shape as the image data.
    :param slice slicer: gain and offset slicer.
    """
    if gain is not None and offset is not None:
        correctGainOffset(data, gain[slicer], offset[slicer])
    elif offset is not None:
        correctOffset(data, offset[slicer])
    elif gain is not None:
        correctGain(data, gain[slicer])


def mask_image_data(image_data, *,
                    image_mask=None, threshold_mask=None, keep_nan=False):
    """Mask image data by image mask and/or threshold mask.

    :param numpy.ndarray image_data: image to be masked.
        Shape = (y, x) or (indices, y, x)
    :param numpy.ndarray/None image_mask: image mask, which has the same
        shape as the image.
    :param tuple/None threshold_mask: (min, max) of the threshold mask.
    :param bool keep_nan: True for masking all pixels in nan and False for
        masking all pixels to zero (including the existing nan).
    """
    f = maskNanImageData if keep_nan else maskZeroImageData

    if image_mask is None and threshold_mask is None:
        f(image_data)
    elif image_mask is None:
        f(image_data, *threshold_mask)
    elif threshold_mask is None:
        f(image_data, image_mask)
    else:
        f(image_data, image_mask, *threshold_mask)


def image_with_mask(image_data, mask=None, *, threshold_mask=None):
    """Mask both image data and mask.

    :param numpy.ndarray image_data: image to be masked.
        Shape = (y, x) or (indices, y, x)
    :param numpy.ndarray/None mask: image mask, which has the same
        shape as the image. If image mask is given, it will be modified
        inplace.
    :param tuple/None threshold_mask: (min, max) of the threshold mask.
    """
    if mask is None and threshold_mask is None:
        mask = np.zeros_like(image_data, dtype=np.bool)
        maskImageData(image_data, mask)
        return mask

    if threshold_mask is None:
        maskImageData(image_data, mask)
        return mask

    if mask is None:
        mask = np.zeros_like(image_data, dtype=np.bool)
        maskImageData(image_data, mask, *threshold_mask)
        return mask

    maskImageData(image_data, mask, *threshold_mask)
    return mask
