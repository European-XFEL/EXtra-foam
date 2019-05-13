"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

Exceptions.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""


class AggregatingError(Exception):
    """Raised when data aggregating fails."""
    pass


class GeometryFileError(Exception):
    """Raised when error is related to geometry file."""
    pass


class AssemblingError(Exception):
    """Raised when image assembling fails."""
    pass


class ProcessingError(Exception):
    """Raised when data processor fails."""
    pass
