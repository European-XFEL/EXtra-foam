"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>, Ebad Kamil <ebad.kamil@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import numpy as np
from scipy import ndimage

from .imageproc import (
    nanmeanImageArray, movingAvgImageData,
    imageDataNanMask, maskImageDataNan, maskImageDataZero,
    correctGain, correctOffset, correctDsscOffset, correctGainOffset,
    binPhotons
)
from .imageproc import dropletize as dropletize_impl


def dropletize(data, adu_count, out=None):
    if out is None:
        out = np.zeros_like(data, dtype=np.float32)

    # Mask NaNs
    nan_mask = np.isnan(data)
    data = np.nan_to_num(data)

    labelled, n_labels = ndimage.label(data)
    dropletize_impl(data, labelled, out, n_labels, adu_count)

    out[nan_mask] = np.nan

    return out

def bin_photons(data, adu_count, out=None):
    if data.ndim not in [2, 3]:
        raise ValueError("Only 2D and 3D arrays are supported")

    if out is None:
        out = np.empty_like(data)

    binPhotons(data, adu_count, out)

    return out

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
                       intradark=False,
                       detector=""):
    """Apply gain and/or offset correct to image data.

    :param numpy.array data: image data, Shape = (y, x) or (indices, y, x)
    :param None/numpy.array gain: gain constants, which has the same
        shape as the image data.
    :param None/numpy.array offset: offset constants, which has the same
        shape as the image data.
    :param bool intradark: apply interleaved intra-dark correction after
        the gain/offset correction.
    :param str detector: detector name. If given, specialized correction
        may be applied. "DSSC" - change data pixels with value 0 to 256
        before applying offset correction.
    """
    if gain is not None and offset is not None:
        correctGainOffset(data, gain, offset)
    elif offset is not None:
        if detector == "DSSC":
            correctDsscOffset(data, offset)
        else:
            correctOffset(data, offset)
    elif gain is not None:
        correctGain(data, gain)

    if intradark:
        correctOffset(data)


def mask_image_data(arr, *,
                    image_mask=None,
                    threshold_mask=None,
                    keep_nan=True,
                    out=None):
    """Mask image data by image mask and/or threshold mask.

    :param numpy.ndarray arr: image data to be masked.
        Shape = (y, x) or (indices, y, x)
    :param numpy.ndarray image_mask: image mask. If provided, it must have
        the same shape as a single image, and the type must be bool.
        Shape = (y, x)
    :param tuple/None threshold_mask: (min, max) of the threshold mask.
    :param bool keep_nan: True for masking all pixels in nan and False for
        masking all pixels to zero.
    :param numpy.ndarray out: Optional output array in which to mark the
        union of all pixels being masked. The default is None; if provided,
        it must have the same shape as the image, and the dtype must be bool.
        Only available if the image data is a 2D array. Shape = (y, x)
    """
    f = maskImageDataNan if keep_nan else maskImageDataZero

    if out is None:
        if image_mask is None and threshold_mask is None:
            f(arr)
        elif image_mask is None:
            f(arr, *threshold_mask)
        elif threshold_mask is None:
            f(arr, image_mask)
        else:
            f(arr, image_mask, *threshold_mask)
    else:
        if arr.ndim == 3:
            raise ValueError("'arr' must be 2D when 'out' is specified!")

        if out.dtype != bool:
            raise ValueError("Type of 'out' must be bool!")

        if image_mask is None:
            if threshold_mask is None:
                imageDataNanMask(arr, out)  # get the mask
                f(arr)  # mask nan (only for keep_nan = False)
            else:
                f(arr, *threshold_mask, out)
        else:
            if threshold_mask is None:
                f(arr, image_mask, out)
            else:
                f(arr, image_mask, *threshold_mask, out)
