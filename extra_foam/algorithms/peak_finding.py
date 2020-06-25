"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from scipy.signal import find_peaks


def find_peaks_1d(a, *args, **kwargs):
    return find_peaks(a, *args, **kwargs)
