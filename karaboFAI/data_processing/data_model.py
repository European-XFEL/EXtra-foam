"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

Data models for analysis and visualization.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from enum import IntEnum
import abc

from ..logger import logger


class DataSource(IntEnum):
    CALIBRATED_FILE = 0  # calibrated data from files
    CALIBRATED = 1  # calibrated data from Karabo-bridge
    PROCESSED = 2  # processed data from the Middle-layer device


class OpLaserMode(IntEnum):
    NORMAL = 0
    EVEN_ON = 1
    ODD_ON = 2


class AbstractData(abc.ABC):
    """Abstract data for data node in ProcessedData."""
    MAX_LENGTH = 100000

    @classmethod
    @abc.abstractmethod
    def clear(cls):
        pass

    @classmethod
    @abc.abstractmethod
    def update_hist(cls, *args, **kwargs):
        pass


class RoiData(AbstractData):
    """A class which stores ROI data."""
    MAX_LENGTH = 100000

    train_ids = []
    roi1_intensity_hist = []
    roi2_intensity_hist = []

    def __init__(self):
        super().__init__()

        self.roi1 = None
        self.roi2 = None

    @classmethod
    def clear(cls):
        cls.train_ids.clear()
        cls.roi1_intensity_hist.clear()
        cls.roi2_intensity_hist.clear()

    @classmethod
    def update_hist(cls, tid, intensity1, intensity2):
        cls.train_ids.append(tid)
        cls.roi1_intensity_hist.append(intensity1)
        cls.roi2_intensity_hist.append(intensity2)
        if len(cls.train_ids) > cls.MAX_LENGTH:
            # expensive
            cls.train_ids.pop(0)
            cls.roi1_intensity_hist.pop(0)
            cls.roi2_intensity_hist.pop(0)

        if len(cls.train_ids) >= cls.MAX_LENGTH:
            logger.DEBUG("ROI history is full!")


class LaserOnOffData(AbstractData):
    """A class which stores Laser on-off data."""
    MAX_LENGTH = 100000

    train_ids = []
    fom_hist = []

    def __init__(self):
        super().__init__()
        self.on_pulse_intensity = None
        self.off_pulse_intensity = None
        self.on_off_diff = None

    @classmethod
    def clear(cls):
        cls.on_pulse_intensity = None
        cls.off_pulse_intensity = None
        cls.on_off_diff = None
        cls.fom_hist.clear()

    @classmethod
    def update_hist(cls, tid, fom):
        cls.train_ids.append(tid)
        cls.fom_hist.append(fom)

        if len(cls.train_ids) > cls.MAX_LENGTH:
            # expensive
            cls.train_ids.pop(0)
            cls.fom_hist.pop(0)

        if len(cls.train_ids) >= cls.MAX_LENGTH:
            logger.DEBUG("Laser on-off history is full!")


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
    def __init__(self, tid, *,
                 momentum=None,
                 intensities=None,
                 intensity_mean=None,
                 images=None,
                 image_mean=None):
        """Initialization."""
        if not isinstance(tid, int):
            raise ValueError("Train ID must be an integer!")
        # tid is not allowed to be modified once initialized.
        self._tid = tid

        self.momentum = momentum
        self.intensities = intensities
        self.intensity_mean = intensity_mean

        self.images = images
        self.image_mean = image_mean

        self._roi = RoiData()

        self._laser_on_off = LaserOnOffData()

        # the mask information is stored in the data so that all the
        # processing and visualization can use the same mask
        self.threshold_mask = None
        self.image_mask = None

    @property
    def tid(self):
        return self._tid

    @property
    def roi1(self):
        return self._roi.roi1

    @roi1.setter
    def roi1(self, v):
        self._roi.roi1 = v

    @property
    def roi2(self):
        return self._roi.roi2

    @roi2.setter
    def roi2(self, v):
        self._roi.roi2 = v

    @property
    def roi_train_ids(self):
        return self._roi.train_ids

    @property
    def roi1_intensity_hist(self):
        return self._roi.roi1_intensity_hist

    @property
    def roi2_intensity_hist(self):
        return self._roi.roi2_intensity_hist

    @property
    def laser_on_intensity(self):
        return self._laser_on_off.on_pulse_intensity

    @laser_on_intensity.setter
    def laser_on_intensity(self, v):
        self._laser_on_off.on_pulse_intensity = v

    @property
    def laser_off_intensity(self):
        return self._laser_on_off.off_pulse_intensity

    @laser_off_intensity.setter
    def laser_off_intensity(self, v):
        self._laser_on_off.off_pulse_intensity = v

    @property
    def laser_delta_intensity(self):
        return self._laser_on_off.on_off_diff

    @laser_delta_intensity.setter
    def laser_delta_intensity(self, v):
        self._laser_on_off.on_off_diff = v

    @property
    def on_off_train_ids(self):
        return self._laser_on_off.train_ids

    @property
    def on_off_fom_hist(self):
        return self._laser_on_off.fom_hist

    def update_roi_hist(self, *args, **kwargs):
        self._roi.update_hist(*args, **kwargs)

    def update_on_off_hist(self, *args, **kwargs):
        self._laser_on_off.update_hist(*args, **kwargs)

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
