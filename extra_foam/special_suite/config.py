"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import numpy as np


_PIXEL_DTYPE = np.float32

# TODO: improve
_MAX_N_GOTTHARD_PULSES = 120

GOTTHARD_DEVICE = {
    "MID": "MID_EXP_DES/DET/GOTTHARD_RECEIVER:daqOutput",
    "SCS": "SCS_PAM_XOX/DET/GOTTHARD_RECEIVER1:daqOutput",
}
