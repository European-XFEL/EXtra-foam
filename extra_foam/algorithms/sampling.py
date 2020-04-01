"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import numpy as np


def slice_curve(y, x, x_min=None, x_max=None):
    """Slice an x-y plot based on the range of x values.

    x is assumed to be monotonically increasing.

    :param numpy.ndarray y: 1D array.
    :param numpy.ndarray x: 1D array.
    :param None/float x_min: minimum x value.
    :param None/float x_max: maximum x value.

    :return: (the sliced x and y)
    :rtype: (numpy.ndarray, numpy.ndarray)
    """
    if x_min is None:
        x_min = x.min()

    if x_max is None:
        x_max = x.max()

    indices = np.where(np.logical_and(x <= x_max, x >= x_min))
    return y[indices], x[indices]


def down_sample(x):
    """Down sample an array.

    :param numpy.ndarray x: data.

    :return numpy.ndarray: down-sampled data.
    """
    # down-sample rate. the rate is fixed at 2 due to the complexity of
    # up-sampling
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

    up_sample(x, (5, 5)) will return

        np.array([[0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 0, 1, 1, 0],
                  [0, 0, 1, 1, 0],
                  [0, 0, 0, 0, 0]])

    For other target shapes, ValueError will be raised. This implementation
    makes the down-sampling and up-sampling self-consistent.
    """
    # up-sample rate
    rate = 2

    if not isinstance(x, np.ndarray):
        raise TypeError("Input must be a numpy.ndarray!")

    if not isinstance(shape, tuple):
        raise TypeError("shape must be a tuple!")

    msg = 'Array with shape {} cannot be up-sampled to another array with ' \
          'shape {}'.format(x.shape, shape)

    if len(x.shape) == 1:
        if len(shape) != len(x.shape) or np.ceil(shape[0]/rate) != x.shape[0]:
            raise ValueError(msg)

        ret = np.zeros(x.shape[0]*rate)
        ret[:] = x.repeat(rate)
        return ret[:shape[0]]

    elif len(x.shape) == 2:
        if len(shape) != len(x.shape) or \
                np.ceil(shape[0] / rate) != x.shape[0] or \
                np.ceil(shape[1] / rate) != x.shape[1]:
            raise ValueError(msg)

        ret = np.zeros((x.shape[0]*rate, x.shape[1]*rate))
        ret[:, :] = x.repeat(rate, axis=0).repeat(rate, axis=1)
        return ret[:shape[0], :shape[1]]

    elif len(x.shape) == 3:
        # the first dimension is the data ID, which will not be up-sampled
        if len(shape) != len(x.shape) or \
                np.ceil(shape[1] / rate) != x.shape[1] or \
                np.ceil(shape[2] / rate) != x.shape[2]:
            raise ValueError(msg)

        ret = np.zeros((x.shape[0], x.shape[1]*rate, x.shape[2]*rate))
        ret[:, :, :] = x.repeat(rate, axis=1).repeat(rate, axis=2)
        return ret[:, :shape[1], :shape[2]]

    raise ValueError("Array dimension > 3!")
