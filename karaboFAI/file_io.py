"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

Input and output from files.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import os.path as osp

import numpy as np
import imageio

from .config import config


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
    :param tuple expected_shape: expect shape of the image.

    :return numpy.ndarray: image data.
    """
    if not filepath:
        raise ValueError("Please specify the image file!")

    try:
        suffix = osp.splitext(filepath)[-1]
        if '.npy' == suffix:
            ref = np.load(filepath)
        else:
            # imread returns an Array object which is a subclass of
            # np.ndarray
            ref = imageio.imread(filepath)

        if expected_shape is not None and ref.shape != expected_shape:
            raise ValueError(f"Shape of image {ref.shape} differs from "
                             f"expected {expected_shape}!")

        image_dtype = config["IMAGE_DTYPE"]
        if ref.dtype != image_dtype:
            ref = ref.astype(image_dtype)

        return ref

    except Exception as e:
        raise ValueError(f"Failed to load image from {filepath}: {str(e)}")
