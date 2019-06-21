"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

Serialization.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import struct
import numpy as np

from ..config import config


if 'IMAGE_DTYPE' in config:
    _DEFAULT_DTYPE = config['IMAGE_DTYPE']
else:
    _DEFAULT_DTYPE = np.float32


def serialize_image(img):
    """Serialize a single image.

    :param numpy.ndarray img: a 2d numpy array.
    """
    if not isinstance(img, np.ndarray):
        raise TypeError(r"Input image must be a numpy.ndarray!")

    if img.ndim != 2:
        raise ValueError(f"The shape of image must be (y, x)!")

    return struct.pack('>II', *img.shape) + img.tobytes()


def deserialize_image(data, dtype=_DEFAULT_DTYPE):
    """Deserialize a single image.

    :param bytes data: serialized image bytes.
    :param type dtype: data type of the image.

    :return: a 2d numpy array.
    """
    offset = 8
    w, h = struct.unpack('>II', data[:offset])

    img = np.frombuffer(data, dtype=dtype, offset=offset)
    img.shape = w, h
    return img


def serialize_images(imgs):
    """Serialize an array of images.

    :param numpy.ndarray imgs: a 3d numpy array with shape (indices, y, x).
    """
    if not isinstance(imgs, np.ndarray):
        raise TypeError(r"Input images must be a numpy.ndarray!")

    if imgs.ndim != 3:
        raise ValueError(f"The shape of image must be (indices, y, x)!")

    return struct.pack('>III', *imgs.shape) + imgs.tobytes()


def deserialize_images(data, dtype=_DEFAULT_DTYPE):
    """Deserialize an array of images.

    :param bytes data: serialized image bytes.
    :param type dtype: data type of the image.

    :return: a 3d numpy array with shape (indices, y, x)
    """
    offset = 12
    shape = struct.unpack('>III', data[:offset])

    img = np.frombuffer(data, dtype=dtype, offset=offset)
    img.shape = shape

    return img
