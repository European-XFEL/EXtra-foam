"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import os.path as osp

import numpy as np
import imageio


def write_image(img, filepath):
    """Write an image to file.

    :param numpy.ndarray img: image data.
    :param str filepath: path of the image file.
    """
    if not filepath:
        raise ValueError("Please specify a file to save current image!")

    try:
        suffix = osp.splitext(filepath)[-1]
        if '.npy' == suffix:
            np.save(filepath, img)
        else:
            # We do not perform a type check here to make 'write_image' and
            # 'read_image' more generic. Users can in principle read
            # unsupported file types in the GUI although the GUI has a filter.
            # But as long as it does not crash the app and 'imageio' can
            # handle it, it is the users to responsibility for the correctness
            # of the result.
            imageio.imwrite(filepath, img)
    except Exception as e:
        raise ValueError(f"Failed to write image to {filepath}: {str(e)}")


def read_image(filepath, *, expected_shape=None):
    """Read an image from file.

    :param str filepath: path of the image file.
    :param tuple/None expected_shape: expect shape of the image.

    :return numpy.ndarray: image data.
    """
    if not filepath:
        raise ValueError("Please specify the image file!")

    try:
        suffix = osp.splitext(filepath)[-1]
        if '.npy' == suffix:
            img = np.load(filepath)
        else:
            # imread returns an Array object which is a subclass of
            # np.ndarray
            img = imageio.imread(filepath)

        if expected_shape is not None and img.shape != expected_shape:
            raise ValueError(f"Shape of image {img.shape} differs from "
                             f"expected {expected_shape}!")
        elif img.ndim != 2:
            raise ValueError("Image must be an array with 2 dimensions!")

        return img

    except Exception as e:
        raise ValueError(f"Failed to load image from {filepath}: {str(e)}")


def read_cal_constants(filepath):
    """Read calibration constant from the given file.

    The file must be a '.npy' file.

    :param str filepath: path of the constant file.

    :return numpy.ndarray: constant array.
    """
    if not filepath:
        raise ValueError("Please specify the image file!")

    try:
        c = np.load(filepath)
    except Exception as e:
        raise ValueError(f"Failed to load constants from {filepath}: {str(e)}")

    if c.ndim not in (2, 3):
        raise ValueError("Constants must be an array with 2 or 3 dimensions!")

    return c
