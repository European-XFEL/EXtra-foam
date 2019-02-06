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
    PROCESSED = 2  # processed data from the Middle-layer device


class RoiHist:
    """A class which stores historical data of ROI."""
    train_ids = []
    roi1_intensities = []
    roi2_intensities = []

    MAX_LENGTH = 100000

    @classmethod
    def clear(cls):
        cls.train_ids.clear()
        cls.roi1_intensities.clear()
        cls.roi2_intensities.clear()

    @classmethod
    def append(cls, tid, roi1, roi2):
        cls.train_ids.append(tid)
        cls.roi1_intensities.append(roi1)
        cls.roi2_intensities.append(roi2)
        if len(cls.train_ids) > cls.MAX_LENGTH:
            # expensive
            cls.train_ids.pop(0)
            cls.roi1_intensities.pop(0)
            cls.roi2_intensities.pop(0)

    @classmethod
    def full(cls):
        if len(cls.train_ids) >= cls.MAX_LENGTH:
            return True
        return False


class ProcessedData:
    """A class which stores the processed data.

    Attributes:
        tid (int): train ID.
        momentum (numpy.ndarray): x-axis of azimuthal integration result.
            Shape = (momentum,)
        intensities (numpy.ndarray): y-axis of azimuthal integration result.
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
    def __init__(self, tid):
        """Initialization."""
        if not isinstance(tid, int):
            raise ValueError("Train ID must be an integer!")
        # tid is not allowed to be modified once initialized.
        self._tid = tid

        self.momentum = None
        self.intensities = None
        self.intensity_mean = None

        self.images = None
        self.image_mean = None

        self.roi1 = None
        self.roi2 = None
        self.roi_hist = RoiHist()

        # the mask information is stored in the data so that all the
        # processing and visualization can use the same mask
        self.threshold_mask = None
        self.image_mask = None

    @property
    def tid(self):
        return self._tid

    @property
    def roi_train_ids(self):
        return self.roi_hist.train_ids

    @property
    def roi1_intensities(self):
        return self.roi_hist.roi1_intensities

    @property
    def roi2_intensities(self):
        return self.roi_hist.roi2_intensities

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
