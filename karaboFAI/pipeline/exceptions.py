"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

Exceptions.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""


class _ProcessingError(Exception):
    """Base Exception for non-fatal errors in pipeline.

    The error must not be fatal and the rest of the data processing pipeline
    can still resume.
    """
    pass


class AggregatingError(_ProcessingError):
    """Raised when data aggregating fails."""
    pass


class ProcessingError(_ProcessingError):
    """Raised when data processor fails."""
    pass


class _StopPipelineError(Exception):
    """Base Exception for fatal errors in pipeline.

    The error is fatal so once it is raised, the pipeline should be stopped
    since it does not make sense to continue.
    """
    pass


class AssemblingError(_StopPipelineError):
    """Raised when image assembling fails."""
    pass
