"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import numpy as np

_MAX_N_PULSES_PER_TRAIN = 2700

_MAX_INT32 = np.iinfo(np.int32).max
_MIN_INT32 = np.iinfo(np.int32).min

_IMAGE_DTYPE = np.float32
_PIXEL_DTYPE = np.float32

# TODO: improve
_MAX_N_GOTTHARD_PULSES = 120

GOTTHARD_DEVICE = {
    "MID": "MID_EXP_DES/DET/GOTTHARD_RECEIVER:daqOutput",
    "SCS": "SCS_PAM_XOX/DET/GOTTHARD_RECEIVER1:daqOutput",
}

_DEFAULT_CLIENT_PORT = 45454

_CLIENT_TIME_OUT = 0.1  # second

# initial (width, height) of a special analysis window
_GUI_SPECIAL_WINDOW_SIZE = (1680, 1080)
# interval for polling new processed data, in milliseconds
_GUI_PLOT_UPDATE_TIMER = 10
