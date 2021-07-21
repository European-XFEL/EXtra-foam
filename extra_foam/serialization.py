"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import struct
import numpy as np

from .config import config


if 'SOURCE_PROC_IMAGE_DTYPE' in config:
    _DEFAULT_DTYPE = config['SOURCE_PROC_IMAGE_DTYPE']
else:
    _DEFAULT_DTYPE = np.float32


def serialize_image(img, is_mask=False):
    """Serialize a single image.

    :param numpy.ndarray img: a 2d numpy array.
    :param bool is_mask: if True, it assumes that the input is
        a boolean array.
    """
    if not isinstance(img, np.ndarray):
        raise TypeError(r"Input image must be a numpy.ndarray!")

    if img.ndim != 2:
        raise ValueError(f"The shape of image must be (y, x)!")

    if is_mask:
        if img.dtype != bool:
            img = img.astype(bool)
        return struct.pack('>II', *img.shape) + np.packbits(img).tobytes()

    return struct.pack('>II', *img.shape) + img.tobytes()


def deserialize_image(data, dtype=_DEFAULT_DTYPE, is_mask=False):
    """Deserialize a single image.

    :param bytes data: serialized image bytes.
    :param type dtype: data type of the image.
    :param bool is_mask: if True, it assumes that the input is the buffer of
        a bit array.

    :return: a 2d numpy array.
    """
    offset = 8
    w, h = struct.unpack('>II', data[:offset])

    if is_mask:
        packed_bits = np.frombuffer(data, dtype=np.uint8, offset=offset)
        img = np.unpackbits(packed_bits)[:w*h].astype(bool, copy=False)
        img.shape = w, h
        return img

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
