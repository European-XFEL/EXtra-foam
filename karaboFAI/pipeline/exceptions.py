"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

Exceptions.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""


class ProcessingError(Exception):
    """Base Exception for non-fatal errors in pipeline.

    The error must not be fatal and the rest of the data processing pipeline
    can still resume.
    """
    pass


class StopPipelineError(Exception):
    """Base Exception for fatal errors in pipeline.

    The error is fatal so once it is raised, the pipeline should be stopped
    since it does not make sense to continue.
    """
    pass


class ImageProcessingError(StopPipelineError):
    """Raised when ImageProcessor.process fails."""
    pass


class AssemblingError(StopPipelineError):
    """Raised when image assembling fails."""
    pass


class PumpProbeIndexError(StopPipelineError):
    """Raised when the pulse indices are invalid."""
    pass


class DropAllPulsesError(StopPipelineError):
    """Raised when no pulse is valid after pulse filtering."""
    pass
