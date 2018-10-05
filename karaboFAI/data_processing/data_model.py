"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

Data model for analysis and visualization.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from enum import IntEnum


class DataSource(IntEnum):
    CALIBRATED_FILE = 0  # calibrated data from files
    CALIBRATED = 1  # calibrated data from Karabo-bridge
    ASSEMBLED = 2  # assembled data from Karabo-bridge
    PROCESSED = 3  # processed data from the Middle-layer device


class ProcessedData:
    """A class which stores the processed data.

    Attributes:
        tid (int): train ID.
        momentum (numpy.ndarray): x-axis of azimuthal integration result.
            Shape = (momentum,)
        intensity (numpy.ndarray): y-axis of azimuthal integration result.
            Shape = (pulse_id, intensity)
        intensity_mean (numpy.ndarray): average of the y-axis of azimuthal
            integration result over pulses. Shape = (intensity,)
        image (numpy.ndarray): detector images for all the pulses.
            Shape = (pulse_id, y, x)
        image_mean (numpy.ndarray): average of the detector images over
            pulses. Shape = (y, x)
        image_mask (numpy.ndarray): an image mask which is applied to all
            the detector images, default = None. Shape = (y, x)
    """
    def __init__(self, tid, *,
                 momentum=None,
                 intensity=None,
                 intensity_mean=None,
                 image=None,
                 image_mean=None,
                 image_mask=None):
        """Initialization."""
        if not isinstance(tid, int):
            raise ValueError("Train ID must be an integer!")
        # tid is not allowed to be modified once initialized.
        self._tid = tid

        self.momentum = momentum
        self.intensity = intensity
        self.intensity_mean = intensity_mean

        self.image = image
        self.image_mean = image_mean
        self.image_mask = image_mask

    @property
    def tid(self):
        return self._tid

    def empty(self):
        """Check the goodness of the data.

        TODO: improve
        """
        if self.intensity is None or self.image is None:
            return True
        return False
