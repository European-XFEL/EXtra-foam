"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

Data model for analysis and visualization.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import numpy as np
from enum import IntEnum


class DataSource(IntEnum):
    CALIBRATED_FILE = 0  # calibrated data from files
    CALIBRATED = 1  # calibrated data from Karabo-bridge
    PROCESSED = 2  # processed data from the Middle-layer device


class ProcessedData:
    """A class which stores the processed data.

    TODO: separate the ProcessedData class for FAI and BDP?

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
        threshold_mask (tuple): (min, max) threshold of the image mask.
        image_mask (numpy.ndarray): an image mask which is applied to all
            the detector images, default = None. Shape = (y, x)
    """
    def __init__(self, tid, *,
                 momentum=None,
                 intensity=None,
                 intensity_mean=None,
                 images=None,
                 image_mean=None,
                 roi1=None,
                 roi2=None,
                 threshold_mask=None,
                 image_mask=None):
        """Initialization."""
        if not isinstance(tid, int):
            raise ValueError("Train ID must be an integer!")
        # tid is not allowed to be modified once initialized.
        self._tid = tid

        self.momentum = momentum
        self.intensity = intensity
        self.intensity_mean = intensity_mean

        self.images = images
        self.image_mean = image_mean

        self.roi1 = roi1
        self.roi2 = roi2

        # the mask information is stored in the data so that all the
        # processing and visualization can use the same mask
        self.threshold_mask = threshold_mask
        self.image_mask = image_mask

    @property
    def tid(self):
        return self._tid

    def empty(self):
        """Check the goodness of the data.

        TODO: improve
        """
        if self.images is None:
            return True
        return False


class Data4Visualization:
    """Data shared between all the windows and widgets.

    The internal data is only modified in MainGUI.updateAll()
    """
    def __init__(self):
        self.__value = ProcessedData(-1)

    def get(self):
        return self.__value

    def set(self, value):
        self.__value = value
