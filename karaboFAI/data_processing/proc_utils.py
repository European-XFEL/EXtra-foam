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
