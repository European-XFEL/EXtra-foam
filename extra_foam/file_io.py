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


def write_image(filepath, img):
    """Write an image to file.

    :param str filepath: path of the image file.
    :param numpy.ndarray img: image data.
    """
    if not filepath:
        raise ValueError("Empty filepath!")

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
            raise ValueError("Image must be a 2D array!")

        return img

    except Exception as e:
        raise ValueError(f"Failed to load image from {filepath}: {str(e)}")


def write_numpy_array(filepath, arr):
    """Write numpy array to file.

    The file must be a '.npy' file.

    :param str filepath: path of the .npy file.
    :param numpy.ndarray arr: array data.
    """
    if not filepath:
        raise ValueError("Empty filepath!")

    try:
        np.save(filepath, arr)
    except Exception as e:
        raise ValueError(
            f"Failed to write numpy array to {filepath}: {str(e)}")


def read_numpy_array(filepath, *, dimensions=None):
    """Read numpy array from file.

    The file must be a '.npy' file.

    :param str filepath: path of the .npy file.
    :param tuple/None dimensions: expected dimensions of the numpy array.

    :return numpy.ndarray: numpy array.
    """
    if '.npy' != osp.splitext(filepath)[-1]:
        raise ValueError("Input must be a .npy file!")

    try:
        c = np.load(filepath)
    except Exception as e:
        raise ValueError(
            f"Failed to read numpy array from {filepath}: {str(e)}")

    if dimensions is not None and c.ndim not in dimensions:
        raise ValueError(f"Expect array with dimensions "
                         f"{dimensions}: actual {c.ndim}")

    return c
