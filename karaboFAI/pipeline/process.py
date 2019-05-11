"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

Process management.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""

from collections import namedtuple


ProcessInfo = namedtuple("ProcessInfo", [
    "process",
    "stdout_file",
    "stderr_file",
])

