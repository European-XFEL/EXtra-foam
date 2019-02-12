"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

Data models for analysis and visualization.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import abc
from enum import IntEnum

from ..logger import logger


class DataSource(IntEnum):
    CALIBRATED_FILE = 0  # calibrated data from files
    CALIBRATED = 1  # calibrated data from Karabo-bridge
    PROCESSED = 2  # processed data from the Middle-layer device


class OpLaserMode(IntEnum):
    INACTIVE = 0  # not perform any relevant calculation
    NORMAL = 1  # on-/off- pulses in the same train
    EVEN_ON = 2  # on-/off- pulses have even/odd train IDs, respectively
    ODD_ON = 3  # on/-off- pulses have odd/even train IDs, respectively


class FomName(IntEnum):
    # Calculate the FOM based on the azimuthal integration of the mean
    # of the assembled image(s).
    ASSEMBLED_MEAN = 1
    # Calculate the FOM based on the difference between the azimuthal
    # integration result between the laser on/off pulse(s).
    LASER_ON_OFF = 2


class AiNormalizer(IntEnum):
    # Normalize the azimuthal integration curve by integration of the
    # curve itself.
    CURVE = 1
    # Normalize the azimuthal integration curve by the sum of the
    # integrations of the ROI(s).
    ROI = 2


class RoiValueType(IntEnum):
    INTEGRATION = 1  # monitor integration of ROI
    MEAN = 2  # monitor mean of ROI


class TrainData:
    MAX_LENGTH = 1000000

    def __init__(self, **kwargs):
        # x can be tid or a correlated param

        # We need to have a 'x' for each sub-dataset due to the
        # concurrency of data processing.
        self._x = []
        self._values = []
        self._info = kwargs

    def __get__(self, instance, instance_type):
        if instance is None:
            return self
        return self._x, self._values, self._info

    def __set__(self, instance, pair):
        x, value = pair
        self._x.append(x)
        self._values.append(value)

        # TODO: improve, e.g., cache
        if len(self._x) > self.MAX_LENGTH:
            self.__delete__(instance)

    def __delete__(self, instance):
        del self._x[0]
        del self._values[0]

    def clear(self):
        self._x.clear()
        self._values.clear()


class AbstractData(abc.ABC):

    @classmethod
    def clear(cls):
        for kls in cls.__dict__:
            if isinstance(cls.__dict__[kls], TrainData):
                # descriptor protocol will not be triggered here
                cls.__dict__[kls].clear()


class RoiData(AbstractData):
    """A class which stores ROI data."""

    # value (integration/mean/median) histories of ROI1 and ROI2
    values1 = TrainData()
    values2 = TrainData()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.roi1 = None  # (w, h, px, py)
        self.roi2 = None  # (w, h, px, py)


class LaserOnOffData(AbstractData):
    """A class which stores Laser on-off data."""

    # FOM history
    foms = TrainData()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.on_pulse = None
        self.off_pulse = None
        self.diff = None


class CorrelationData(AbstractData):
    """A class which stores Laser on-off data."""

    @classmethod
    def add_param(cls, idx, device_id, ppt):
        setattr(cls, f'param{idx}', TrainData(device_id=device_id,
                                              property=ppt))

    @classmethod
    def remove_param(cls, idx):
        name = f'param{idx}'
        if hasattr(cls, name):
            delattr(cls, name)

    @classmethod
    def get_params(cls):
        params = []
        for kls in cls.__dict__:
            if isinstance(cls.__dict__[kls], TrainData):
                params.append(kls)

        return params


class ProcessedData:
    """A class which stores the processed data.

    ProcessedData also provide interface for manipulating the other node
    dataset, e.g. RoiData, CorrelationData, LaserOnOffData.

    Attributes:
        tid (int): train ID.
        momentum (numpy.ndarray): x-axis of azimuthal integration result.
            Shape = (momentum,)
        intensities (numpy.ndarray): y-axis of azimuthal integration result.
            Shape = (pulse_id, intensity)
        intensity_mean (numpy.ndarray): average of the y-axis of azimuthal
            integration result over pulses. Shape = (intensity,)
        images (numpy.ndarray): detector images for all the pulses.
            Shape = (pulse_id, y, x)
        image_mean (numpy.ndarray): average of the detector images over
            pulses. Shape = (y, x)
        roi (RoiData): stores ROI related data.
        on_off (LaserOnOffData): stores laser on-off related data.
        correlation (CorrelationData): correlation related data.
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

        self._tid = tid  # current Train ID

        self.momentum = momentum
        self.intensities = intensities
        self.intensity_mean = intensity_mean

        self.images = images
        self.image_mean = image_mean

        self.roi = RoiData()
        self.on_off = LaserOnOffData()
        self.correlation = CorrelationData()

        # the mask information is stored in the data so that all the
        # processing and visualization can use the same mask
        self.threshold_mask = None
        self.image_mask = None

    @property
    def tid(self):
        return self._tid

    @classmethod
    def clear_roi_hist(cls):
        RoiData.clear()

    @classmethod
    def clear_onoff_hist(cls):
        LaserOnOffData.clear()

    @classmethod
    def clear_correlation_hist(cls):
        CorrelationData.clear()

    @staticmethod
    def add_correlator(idx, device_id, ppt):
        """Add a correlated parameter.

        :param int idx: index
        :param str device_id: device ID
        :param str ppt: property
        """
        if device_id and ppt:
            CorrelationData.add_param(idx, device_id, ppt)
        else:
            CorrelationData.remove_param(idx)

    @staticmethod
    def get_correlators():
        return CorrelationData.get_params()

    def empty(self):
        """Check the goodness of the data."""
        logger.debug("Deprecated! Check the specific data!")
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
