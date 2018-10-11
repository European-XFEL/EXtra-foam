"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

Helper functions for data processing.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import numpy as np


def sub_array_with_range(y, x, range_=None):
    if range_ is None:
        return y, x
    indices = np.where(np.logical_and(x <= range_[1], x >= range_[0]))
    return y[indices], x[indices]


def integrate_curve(y, x, range_=None):
    itgt = np.trapz(*sub_array_with_range(y, x, range_))
    return itgt if itgt else 1.0


def down_sample(x):
    """Down sample an array.

    :param numpy.ndarray x: data.

    :return numpy.ndarray: down-sampled data.
    """
    # down-sample rate. the rate is fixed at 2 due to the complexity of
    # upsampling
    rate = 2

    if not isinstance(x, np.ndarray):
        raise TypeError("Input must be a numpy.ndarray!")

    if len(x.shape) == 1:
        return x[::rate]

    if len(x.shape) == 2:
        return x[::rate, ::rate]

    if len(x.shape) == 3:
        # the first dimension is the data ID, which will not be down-sampled
        return x[:, ::rate, ::rate]

    raise ValueError("Array dimension > 3!")


def up_sample(x, shape):
    """Up sample an array.

    :param numpy.ndarray x: data.
    :param tuple shape: shape of the up-sampled data.

    :return numpy.ndarray: up-sampled data.

    :raises: ValueError, TypeError

    Examples:

    x = np.array([[0, 0, 0],
                  [0, 1, 0],
                  [0, 0, 0]])

    up_sample(x, (6, 6)) will return

        np.array([[0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0],
                  [0, 0, 1, 1, 0, 0],
                  [0, 0, 1, 1, 0, 0],
                  [0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0]])
    """
    # up-sample rate
    rate = 2

    if not isinstance(x, np.ndarray):
        raise TypeError("Input must be a numpy.ndarray!")

    if not isinstance(shape, tuple):
        raise TypeError("shape must be a tuple!")

    msg = 'Array with shape {} cannot be upsampled to another array with ' \
          'shape {}'.format(x.shape, shape)

    if len(x.shape) == 1:
        if len(shape) != len(x.shape) or np.ceil(shape[0]/2) != x.shape[0]:
            raise ValueError(msg)

        ret = np.zeros(x.shape[0]*2)
        ret[:2*x.shape[0]] = x.repeat(rate)
        return ret[:shape[0]]

    elif len(x.shape) == 2:
        if len(shape) != len(x.shape) or \
                np.ceil(shape[0] / 2) != x.shape[0] or \
                np.ceil(shape[1] / 2) != x.shape[1]:
            raise ValueError(msg)

        ret = np.zeros((x.shape[0]*2, x.shape[1]*2))
        ret[:2*x.shape[0], :2*x.shape[1]] = \
            x.repeat(rate, axis=0).repeat(rate, axis=1)
        return ret[:shape[0], :shape[1]]

    elif len(x.shape) == 3:
        # the first dimension is the data ID, which will not be up-sampled
        if len(shape) != len(x.shape) or \
                np.ceil(shape[1] / 2) != x.shape[1] or \
                np.ceil(shape[2] / 2) != x.shape[2]:
            raise ValueError(msg)

        ret = np.zeros((x.shape[0], x.shape[1]*2, x.shape[2]*2))
        ret[:, :2*x.shape[1], :2*x.shape[2]] = \
            x.repeat(rate, axis=1).repeat(rate, axis=2)
        return ret[:, :shape[1], :shape[2]]

    raise ValueError("Array dimension > 3!")
