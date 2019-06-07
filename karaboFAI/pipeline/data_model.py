"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

Data models for analysis and visualization.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import copy
from threading import Lock

import numpy as np

from ..algorithms import nanmean_axis0_para, mask_by_threshold
from ..config import config


class PairData:
    """Store the history pair data.

    Each data point is pair of data: (x, y).

    For correlation plots: x can be a train ID or a motor position,
    and y is the figure of merit (FOM).
    """
    MAX_LENGTH = 3000  # scatter plot is expensive

    def __init__(self, **kwargs):
        # We need to have a 'x' for each sub-dataset due to the
        # concurrency of data processing.
        self._x = []
        self._y = []
        # for now it is used in CorrelationData to store device ID and
        # property information
        self._info = kwargs

        self._lock = Lock()

    def __get__(self, instance, instance_type):
        if instance is None:
            return self
        # Note: here we must ensure that the data is not copied
        with self._lock:
            x = np.array(self._x)
            y = np.array(self._y)
            info = copy.copy(self._info)
        return x, y, info

    def __set__(self, instance, pair):
        this_x, this_y = pair
        with self._lock:
            self._x.append(this_x)
            self._y.append(this_y)

        # This is a reasonable choice since we always wants to return a
        # reference in __get__!
        if len(self._x) > self.MAX_LENGTH:
            self.__delete__(instance)

    def __delete__(self, instance):
        with self._lock:
            del self._x[0]
            del self._y[0]

    def clear(self):
        with self._lock:
            self._x.clear()
            self._y.clear()
        # do not clear _info here!


class AccumulatedPairData(PairData):
    """Store the history accumulated pair data.

    Each data point is pair of data: (x, DataStat).

    The data is collected in a stop-and-collected way. A motor,
    for example, will stop in a location and collect data for a
    period of time. Then,  each data point in the accumulated
    pair data is the average of the data during this period.
    """
    class DataStat:
        """Statistic of data."""
        def __init__(self):
            self.count = None
            self.avg = None
            self.min = None
            self.max = None

    MAX_LENGTH = 600

    _min_count = 2
    _epsilon = 1e-9

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if 'resolution' not in kwargs:
            raise ValueError("'resolution' is required!")
        resolution = kwargs['resolution']
        if resolution <= 0:
            raise ValueError("'resolution must be positive!")
        self._resolution = resolution

        self._y_count = []
        self._y_avg = []
        self._y_min = []
        self._y_max = []
        self._y_std = []

    def __set__(self, instance, pair):
        this_x, this_y = pair
        with self._lock:
            if self._x:
                if abs(this_x - self._x[-1]) - self._resolution < self._epsilon:
                    self._y_count[-1] += 1
                    avg_prev = self._y_avg[-1]
                    self._y_avg[-1] += \
                        (this_y - self._y_avg[-1]) / self._y_count[-1]
                    self._y_std[-1] += \
                        (this_y - avg_prev)*(this_y - self._y_avg[-1])
                    # self._y_min and self._y_max does not store min and max
                    # Only Standard deviation will be plotted. Min Max functionality
                    # does not exist as of now.
                    # self._y_min stores y_avg - 0.5*std_dev
                    # self._y_max stores y_avg + 0.5*std_dev
                    self._y_min[-1] = self._y_avg[-1] - 0.5*np.sqrt(
                        self._y_std[-1]/self._y_count[-1])
                    self._y_max[-1] = self._y_avg[-1] + 0.5*np.sqrt(
                        self._y_std[-1]/self._y_count[-1])
                    self._x[-1] += (this_x - self._x[-1]) / self._y_count[-1]
                else:
                    # If the number of data at a location is less than
                    # min_count, the data at this location will be discarded.
                    if self._y_count[-1] < self._min_count:
                        del self._x[-1]
                        del self._y_count[-1]
                        del self._y_avg[-1]
                        del self._y_min[-1]
                        del self._y_max[-1]
                        del self._y_std[-1]
                    self._x.append(this_x)
                    self._y_count.append(1)
                    self._y_avg.append(this_y)
                    self._y_min.append(this_y)
                    self._y_max.append(this_y)
                    self._y_std.append(0.0)
            else:
                self._x.append(this_x)
                self._y_count.append(1)
                self._y_avg.append(this_y)
                self._y_min.append(this_y)
                self._y_max.append(this_y)
                self._y_std.append(0.0)

        if len(self._x) > self.MAX_LENGTH:
            self.__delete__(instance)

    def __get__(self, instance, instance_type):
        if instance is None:
            return self

        y = self.DataStat()
        with self._lock:
            if self._y_count and self._y_count[-1] < self._min_count:
                x = np.array(self._x[:-1])
                y.count = np.array(self._y_count[:-1])
                y.avg = np.array(self._y_avg[:-1])
                y.min = np.array(self._y_min[:-1])
                y.max = np.array(self._y_max[:-1])
            else:
                x = np.array(self._x)
                y.count = np.array(self._y_count)
                y.avg = np.array(self._y_avg)
                y.min = np.array(self._y_min)
                y.max = np.array(self._y_max)

            info = copy.copy(self._info)

        return x, y, info

    def __delete__(self, instance):
        with self._lock:
            del self._x[0]
            del self._y_count[0]
            del self._y_avg[0]
            del self._y_min[0]
            del self._y_max[0]
            del self._y_std[0]

    def clear(self):
        with self._lock:
            self._x.clear()
            self._y_count.clear()
            self._y_avg.clear()
            self._y_min.clear()
            self._y_max.clear()
            self._y_std.clear()
        # do not clear _info here!


class AbstractData:
    @classmethod
    def clear(cls):
        for attr in cls.__dict__.values():
            if isinstance(attr, PairData):
                # descriptor protocol will not be triggered here
                attr.clear()


class XgmData(AbstractData):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.source = None
        self.intensity = 0.0


class XasData(AbstractData):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # show the expected data type
        self.bin_center = np.array([])
        self.bin_count = np.array([])
        self.xgm = np.array([])
        self.absorptions = [np.array([]), np.array([])]


class BinData(AbstractData):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.analysis_type = None
        self.count_x = None
        self.count_y = None
        self.center_x = None
        self.center_y = None
        self.x = None
        self.y = None
        # 1D curve in each x/y bin
        self.data_x = None
        self.data_y = None
        self.fom = None


class CorrelationData(AbstractData):
    """Correlation data model."""

    n_params = len(config["CORRELATION_COLORS"])

    def __init__(self):
        super().__init__()
        self.fom = None
        for i in range(1, self.n_params+1):
            setattr(self, f"correlator{i}", None)

    def update_hist(self, tid):
        fom = self.fom
        for i in range(1, self.n_params+1):
            corr = getattr(self, f"correlator{i}")
            if corr is not None:
                setattr(self, f"correlation{i}", (corr, fom))


class RoiData(AbstractData):
    """A class which stores ROI data."""

    _n_rois = len(config["ROI_COLORS"])
    __initialized = False

    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls, *args, **kwargs)
        # (sum/mean) histories of ROIs
        if not cls.__initialized:
            for i in range(1, cls._n_rois+1):
                setattr(cls, f"roi{i}_hist", PairData())
            cls.__initialized = True
        return instance

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for i in range(1, self._n_rois+1):
            setattr(self, f"roi{i}", None)  # (x, y, w, h)
            setattr(self, f"roi{i}_proj_x", None)  # projection on x
            setattr(self, f"roi{i}_proj_y", None)  # projection on y
            setattr(self, f"roi{i}_fom", None)  # FOM

    def update_hist(self, tid):
        for i in range(1, self._n_rois+1):
            fom = getattr(self, f"roi{i}_fom")
            if fom is None:
                fom = 0
            setattr(self, f"roi{i}_hist", (tid, fom))


class AzimuthalIntegrationData(AbstractData):
    """Azimuthal integration data model.

    momentum (numpy.ndarray): x-axis of azimuthal integration result.
        Shape = (momentum,)
    intensities (numpy.ndarray): y-axis of azimuthal integration result.
        Shape = (pulse_index, intensity)
    intensity_mean (numpy.ndarray): average of the y-axis of azimuthal
        integration result over pulses. Shape = (intensity,)
    pulse_fom
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.momentum = None
        self.intensities = None
        self.intensity_mean = None
        self.pulse_fom = None


class PumpProbeData(AbstractData):
    """Pump-probe data model."""

    fom_hist = PairData()

    class MovingAverage:
        def __init__(self):
            self._x = None
            self._on_ma = None  # moving average of on data
            self._off_ma = None  # moving average of off data

            self._ma_window = 1
            self._ma_count = 0

            self._lock = Lock()

        def __get__(self, instance, instance_type):
            return self._x, self._on_ma, self._off_ma

        def __set__(self, instance, data):
            x, on, off = data

            with self._lock:
                # x is always None when on/off are image data
                if self._on_ma is not None and on.shape != self._on_ma.shape:
                    # reset moving average if data shape (ROI shape) changes
                    self._ma_count = 0
                    self._on_ma = None
                    self._off_ma = None

                self._x = x
                if self._ma_window > 1 and self._ma_count > 0:
                    if self._ma_count < self._ma_window:
                        self._ma_count += 1
                        denominator = self._ma_count
                    else:   # self._ma_count == self._ma_window
                        # here is an approximation
                        denominator = self._ma_window
                    self._on_ma += (on - self._on_ma) / denominator
                    self._off_ma += (off - self._off_ma) / denominator

                else:  # self._ma_window == 1
                    self._on_ma = on
                    self._off_ma = off
                    if self._ma_window > 1:
                        self._ma_count = 1  # 0 -> 1

        @property
        def moving_average_window(self):
            return self._ma_window

        @moving_average_window.setter
        def moving_average_window(self, v):
            if not isinstance(v, int) or v <= 0:
                v = 1

            if v < self._ma_window:
                # if the new window size is smaller than the current one,
                # we reset everything
                with self._lock:
                    self._ma_window = v
                    self._ma_count = 0
                    self._x = None
                    self._on_ma = None
                    self._off_ma = None

            self._ma_window = v

        @property
        def moving_average_count(self):
            return self._ma_count

        def reset(self):
            with self._lock:
                self._ma_window = 1
                self._ma_count = 0
                self._x = None
                self._on_ma = None
                self._off_ma = None

    # Moving average of the on/off data in pump-probe experiments, for
    # example: azimuthal integration / ROI / 1D projection, etc.
    data = MovingAverage()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.analysis_type = None

        self.abs_difference = None

        # the current average of on/off images
        self.on_image_mean = None
        self.off_image_mean = None

        # the current ROI of on/off images
        self.on_roi = None
        self.off_roi = None

        self.x = None
        # normalized on/off/on-off moving average
        self.norm_on_ma = None
        self.norm_off_ma = None
        self.norm_on_off_ma = None

        # FOM is defined as the difference of the FOMs between on and off
        # signals. For example, in azimuthal integration analysis, FOM is
        # the integration of the scattering curve; in ROI analysis, FOM is
        # the sum of ROI.
        self.fom = None

    def update_hist(self, tid):
        fom = self.fom
        if fom is not None:
            self.fom_hist = (tid, fom)

    @property
    def ma_window(self):
        return self.__class__.__dict__['data'].moving_average_window

    @ma_window.setter
    def ma_window(self, v):
        self.__class__.__dict__['data'].moving_average_window = v

    @property
    def ma_count(self):
        return self.__class__.__dict__['data'].moving_average_count

    @classmethod
    def clear(cls):
        super().clear()
        cls.__dict__['data'].reset()


class ImageData:
    """Image data model.

    ImageData is a container for storing self-consistent image data.
    Once constructed, the internal data are not allowed to be modified.

    Attributes:
        images (numpy.ndarray): all images in the train.
        pixel_size (float): pixel size of the detector.
        n_images (int): number of images in the train.
        background (float): a uniform background value.
        ma_window (int): moving average window size.
        ma_count (int): current moving average count.
        threshold_mask (tuple): (lower, upper) boundaries of the
            threshold mask.
        mean (numpy.ndarray): average image over the train.
        masked_mean (numpy.ndarray): average image over the train with
            threshold mask applied.
        ref (numpy.ndarray): reference image.
        masked_ref (numpy.ndarray): reference image with threshold mask
            applied.
    """

    def __init__(self, data, *,
                 background=0.0,
                 threshold_mask=(-np.inf, np.inf),
                 ma_window=1,
                 ma_count=1):
        """Initialization.

        :param numpy.ndarray data: image data in a train.
        :param float background: a uniform background value.
        :param tuple threshold_mask: threshold mask.
        :param int ma_window: moving average window size.
        :param int ma_count: current moving average count.
        """
        if not isinstance(data, np.ndarray):
            raise TypeError(r"Image data must be numpy.ndarray!")

        if data.ndim <= 1 or data.ndim > 3:
            raise ValueError(f"The shape of image data must be (y, x) or "
                             f"(n_pulses, y, x)!")

        self._pixel_size = config['PIXEL_SIZE']

        if data.dtype != np.float32:
            # dtype of the incoming data could be integer
            self._images = data.astype(np.float32)
        else:
            self._images = data

        if data.ndim == 3:
            self._n_images = self._images.shape[0]
        else:
            self._n_images = 1

        self._shape = self._images.shape[-2:]

        self._mean = nanmean_axis0_para(self._images)
        self._threshold_mask = threshold_mask
        # self._masked_mean does not share memory with self._mean
        self._masked_mean = mask_by_threshold(self._mean, *threshold_mask)

        # TODO: design how to deal with reference image
        self._ref = None
        self._masked_ref = None

        self._bkg = background

        self._ma_window = ma_window
        self._ma_count = ma_count

    @property
    def images(self):
        return self._images

    @property
    def pixel_size(self):
        return self._pixel_size

    @property
    def n_images(self):
        return self._n_images

    @property
    def shape(self):
        return self._shape

    @property
    def background(self):
        return self._bkg

    @property
    def ma_window(self):
        return self._ma_window

    @property
    def ma_count(self):
        return self._ma_count

    @property
    def threshold_mask(self):
        return self._threshold_mask

    @property
    def mean(self):
        return self._mean

    @property
    def masked_mean(self):
        return self._masked_mean

    @property
    def ref(self):
        return self._ref

    @property
    def masked_ref(self):
        return self._masked_ref

    def sliced_mean(self, indices):
        """Return average image over given indices.

        :param list indices: a list of indices. Ignored if data is
            train-resolved.

        :raise IndexError if any index is out of bounds
        """
        if not isinstance(indices, (tuple, list, np.ndarray)):
            raise TypeError("Indices must be either a tuple, a list or "
                            "a numpy.ndarray")

        if self._images.ndim == 3:
            return nanmean_axis0_para(self._images[indices])
        return self._mean

    def sliced_masked_mean(self, indices=None):
        """Return masked average image over given indices.

        :param list indices: a list of indices. Ignored if data is
            train-resolved.

        :raise IndexError if any index is out of bounds
        """
        if self._images.ndim == 3:
            return mask_by_threshold(self.sliced_mean(indices),
                                     *self._threshold_mask)
        # do not re-calculate!
        return self._masked_mean


class ProcessedData:
    """A class which stores the processed data.

    ProcessedData also provide interface for manipulating the other node
    dataset, e.g. RoiData, CorrelationData, PumpProbeData.

    Attributes:
        tid (int): train ID.
        image (ImageData): image data.
        xgm (XgmData): XGM data.
        ai (AzimuthalIntegrationData): azimuthal integration data.
        pp (PumpProbeData): pump-probe data.
        roi (RoiData): ROI data.
        correlation (CorrelationData): correlation related data.
        bin (BinData): binning data.
        xas (XasData): XAS data.
    """

    def __init__(self, tid, images, **kwargs):
        """Initialization."""
        self._tid = tid  # current Train ID

        self._image = ImageData(images, **kwargs)
        self.xgm = XgmData()

        self.ai = AzimuthalIntegrationData()
        self.pp = PumpProbeData()
        self.roi = RoiData()
        self.correlation = CorrelationData()
        self.bin = BinData()
        self.xas = XasData()

    @property
    def tid(self):
        return self._tid

    @property
    def image(self):
        return self._image

    @property
    def pulse_resolved(self):
        return self.image.images.ndim == 3

    @property
    def n_pulses(self):
        return self._image.n_images

    def update_hist(self):
        self.roi.update_hist(self._tid)
        self.pp.update_hist(self._tid)
        self.correlation.update_hist(self._tid)


class DataManagerMixin:
    """Interface for manipulating data model."""
    @staticmethod
    def add_correlation(idx, device_id, ppt, resolution=0.0):
        """Add a correlation.

        :param int idx: index (starts from 1)
        :param str device_id: device ID
        :param str ppt: property
        :param float resolution: resolution. Default = 0.0
        """
        if idx <= 0:
            raise ValueError("Correlation index must start from 1!")

        if device_id and ppt:
            corr = f'correlation{idx}'
            if resolution:
                setattr(CorrelationData, corr, AccumulatedPairData(
                    device_id=device_id, property=ppt, resolution=resolution))
            else:
                setattr(CorrelationData, corr, PairData(
                    device_id=device_id, property=ppt))
        else:
            DataManagerMixin.remove_correlation(idx)

    @staticmethod
    def get_correlations():
        correlations = []
        for kls in CorrelationData.__dict__:
            if isinstance(CorrelationData.__dict__[kls], PairData):
                correlations.append(kls)
        return correlations

    @staticmethod
    def remove_correlation(idx):
        name = f'correlation{idx}'
        if hasattr(CorrelationData, name):
            delattr(CorrelationData, name)

    @staticmethod
    def remove_correlations():
        for i in range(CorrelationData.n_params):
            DataManagerMixin.remove_correlation(i+1)

    @staticmethod
    def reset_correlation():
        CorrelationData.clear()

    @staticmethod
    def reset_roi():
        RoiData.clear()

    @staticmethod
    def reset_pp():
        PumpProbeData.clear()

    @staticmethod
    def reset_xas():
        XasData.clear()
