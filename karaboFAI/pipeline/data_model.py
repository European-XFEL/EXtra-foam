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

from ..algorithms import mask_image
from ..config import config

from karaboFAI.cpp import xt_nanmean_images, xt_moving_average


class PairData:
    """Store the history a data pair.

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
        if this_x is None or this_y is None:
            return

        with self._lock:
            self._x.append(this_x)
            self._y.append(this_y)

        # This is a reasonable choice since we always wants to return a
        # reference in __get__!
        if len(self._x) > self.MAX_LENGTH:
            with self._lock:
                del self._x[0]
                del self._y[0]

    def __delete__(self, instance):
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
        if this_x is None or this_y is None:
            return

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
            with self._lock:
                del self._x[0]
                del self._y_count[0]
                del self._y_avg[0]
                del self._y_min[0]
                del self._y_max[0]
                del self._y_std[0]

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
            self._x.clear()
            self._y_count.clear()
            self._y_avg.clear()
            self._y_min.clear()
            self._y_max.clear()
            self._y_std.clear()
            # do not clear _info here!


class MovingAverageArray:
    """Stores moving average of raw images."""
    def __init__(self):
        self._data = None  # moving average

        self._window = 1
        self._count = 0

    def __get__(self, instance, instance_type):
        if instance is None:
            return self

        return self._data

    def __set__(self, instance, data):
        if data is None:
            return

        if self._data is not None and self._window > 1 and \
                self._count <= self._window and data.shape == self._data.shape:
            if self._count < self._window:
                self._count += 1
                self._data = xt_moving_average(self._data, data, self._count)
            else:  # self._count == self._window
                # here is an approximation
                self._data = xt_moving_average(self._data, data, self._count)

        else:
            self._data = data
            self._count = 1

    def __delete__(self, instance):
        self._data = None
        self._count = 0

    @property
    def window(self):
        return self._window

    @window.setter
    def window(self, v):
        if not isinstance(v, int) or v <= 0:
            raise ValueError("Input must be integer")

        self._window = v

    @property
    def count(self):
        return self._count


class RawImageData(MovingAverageArray):
    """Stores moving average of raw images."""
    def __init__(self):
        super().__init__()

    @property
    def n_images(self):
        if self._data is None:
            return 0

        if self._data.ndim == 3:
            return self._data.shape[0]
        return 1

    @property
    def pulse_resolved(self):
        return self._data.ndim == 3


class DataItem:
    """Train-resolved data item.

    Note: Do not keep history of FOM for each DataItem since it is
          very expensive.

    Attributes:
        x (numpy.array): x coordinate of VFOM.
        vfom (numpy.array): Vector figure-of-merit.
        fom (float): Figure-of-merit.
        x_label (str): x label used in plots.
        y_label (str): y label used in plots.
    """
    def __init__(self, x_label="", y_label=""):
        self.x = None
        self.vfom = None
        self.fom = None

        self.x_label = x_label
        self.y_label = y_label


class DataItemPulse:
    """Pulse-resolved data item.

    Attributes:
        x (numpy.array): x coordinate of VFOM.
        vfom (list of numpy.array): list of vector figure-of-merits.
        fom (list of float): list of figure-of-merit.
        x_label (str): x label used in plots.
        y_label (str): y label used in plots.
    """
    def __init__(self, x_label="", y_label=""):
        self.x = None
        self.vfom = []
        self.fom = []

        self.x_label = x_label
        self.y_label = y_label


class XgmData:
    def __init__(self):
        self.intensity = DataItem()
        self.pos = DataItem()


class XasData:
    # TODO: fix me
    def __init__(self):
        super().__init__()

        # show the expected data type
        self.bin_center = np.array([])
        self.bin_count = np.array([])
        self.xgm = np.array([])
        self.absorptions = [np.array([]), np.array([])]


class _RoiAuxData:
    """_RoiAuxTrain class.

    Store ROI related auxiliary data, e.g. normalization.
    """
    def __init__(self):
        self.norm3 = None
        self.norm4 = None
        self.norm3_sub_norm4 = None
        self.norm3_add_norm4 = None


class RoiData(_RoiAuxData):
    """ROIData class.

    Attributes:
        rect1, rect2, rect3, rect4 (list): ROI coordinates in [x, y, w, h].

        on (_RoiAuxTrain): ROI related auxiliary pump data.
        off (_RoiAuxTrain):ROI related auxiliary probe data.

        roi1, roi2, roi1_sub_roi2, roi1_add_roi2 (DataItem):
            sum of ROI pixel values calculated from ROI1 and ROI2.
        proj1, proj2, proj1_sub_proj2, proj1_add_proj2 (DataItem):
            1D projection data calculated from ROI1 and ROI2.
    """

    def __init__(self):
        super().__init__()

        self.rect1 = [0, 0, -1, -1]
        self.rect2 = [0, 0, -1, -1]
        self.rect3 = [0, 0, -1, -1]
        self.rect4 = [0, 0, -1, -1]

        # for normalization: calculated from ROI3 and ROI4
        self.on = _RoiAuxData()
        self.off = _RoiAuxData()

        self.roi1 = DataItem()
        self.roi2 = DataItem()
        self.roi1_sub_roi2 = DataItem()
        self.roi1_add_roi2 = DataItem()

        # 1. pump-probe 'proj' will directly go to ProcessedData.pp;
        # 2. we may need two projection types at the same time;
        self.proj1 = DataItem(x_label='pixel', y_label='ROI1 projection')
        self.proj2 = DataItem(x_label='pixel', y_label='ROI2 projection')
        self.proj1_sub_proj2 = DataItem(
            x_label='pixel', y_label='ROI1 - ROI2 projection')
        self.proj1_add_proj2 = DataItem(
            x_label='pixel', y_label='ROI1 + ROI2 projection')


class AzimuthalIntegrationData(DataItem):
    """Train-resolved azimuthal integration data."""
    def __init__(self,
                 x_label="Momentum transfer (1/A)",
                 y_label="Scattering signal (arb.u.)"):
        super().__init__(x_label=x_label, y_label=y_label)


class PumpProbeData(DataItem):
    """Pump-probe data.

    VFOM is the difference between normalized on-VFOM and off-VFOM.
    """

    fom_hist = PairData()

    def __init__(self):
        super().__init__()

        self.analysis_type = None

        self.abs_difference = None

        # on/off images for the current train
        self.image_on = None
        self.image_off = None

        # normalized VFOM on/off
        self.vfom_on = None
        self.vfom_off = None

        self.reset = False

    def update_hist(self, tid):
        if self.reset:
            del self.fom_hist

        self.fom_hist = (tid, self.fom)


class ImageData:
    """Image data model.

    ImageData is a container for storing self-consistent image data.
    Once constructed, the internal data are not allowed to be modified.

    Attributes:
        _images (list): a list of pulse images in the train. A value of
            None only indicates that the corresponding pulse image is
            not needed (in the main process).
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

    if 'IMAGE_DTYPE' in config:
        _DEFAULT_DTYPE = config['IMAGE_DTYPE']
    else:
        _DEFAULT_DTYPE = np.float32

    def __init__(self, data, *,
                 mean=None,
                 reference=None,
                 background=0.0,
                 image_mask=None,
                 threshold_mask=(-np.inf, np.inf),
                 ma_window=1,
                 ma_count=1,
                 poi_indices=None):
        """Initialization.

        :param numpy.ndarray data: image data in a train.
        :param numpy.ndarray mean: nanmean of image data in a train. If not
            given, it will be calculated based on the image data. Only used
            for pulse-resolved detectors.
        :param numpy.ndarray reference: reference image.
        :param float background: a uniform background value.
        :param numpy.ndarray image_mask: image mask.
        :param tuple threshold_mask: threshold mask.
        :param int ma_window: moving average window size.
        :param int ma_count: current moving average count.

        Note: data, reference and image_mask must not be modified in-place.
        """
        if not isinstance(data, np.ndarray):
            raise TypeError(r"Image data must be numpy.ndarray!")

        if data.ndim <= 1 or data.ndim > 3:
            raise ValueError(f"The shape of image data must be (y, x) or "
                             f"(n_pulses, y, x)!")

        self._pixel_size = config['PIXEL_SIZE']

        if data.dtype != self._DEFAULT_DTYPE:
            # FIXME: dtype of the incoming data could be integer, but integer
            #        array does not have nanmean.
            images = data.astype(self._DEFAULT_DTYPE)
        else:
            images = data

        self._shape = images.shape[-2:]

        if data.ndim == 3:
            if mean is None:
                self._mean = xt_nanmean_images(images)
            else:
                self._mean = mean

            self._n_images = images.shape[0]
            self._pulse_resolved = True
            # (temporary solution for now) avoid sending all images around
            self._images = [None] * self._n_images
            if poi_indices is not None:
                for i in poi_indices:
                    self._images[i] = images[i]
        else:
            # Note: _image is _mean for train-resolved detectors
            self._mean = images
            self._n_images = 1
            self._pulse_resolved = False
            self._images = images

        self._threshold_mask = threshold_mask

        # if image_mask is given, we assume that the shape of the image
        # mask is the same as the image. This is guaranteed in
        # ImageProcessor.
        self._image_mask = image_mask

        # self._masked_mean does not share memory with self._mean
        self._masked_mean = self._mean.copy()
        mask_image(self._masked_mean,
                   threshold_mask=threshold_mask,
                   image_mask=image_mask,
                   inplace=True)

        self._ref = reference

        self._bkg = background

        self._ma_window = ma_window
        self._ma_count = ma_count

    @property
    def images(self):
        return self._images

    @property
    def reference(self):
        return self._ref

    @property
    def pixel_size(self):
        return self._pixel_size

    @property
    def n_images(self):
        return self._n_images

    @property
    def pulse_resolved(self):
        return self._pulse_resolved

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
    def image_mask(self):
        return self._image_mask

    @property
    def mean(self):
        return self._mean

    @property
    def masked_mean(self):
        return self._masked_mean


class BinData:
    """Binning data model."""

    # 1D binning
    vec1_hist = None
    fom1_hist = None
    count1_hist = None
    vec2_hist = None
    fom2_hist = None
    count2_hist = None

    # 2D binning
    fom12_hist = None
    count12_hist = None

    def __init__(self):
        super().__init__()

        self.mode = None

        self.reset1 = False
        self.reset2 = False

        # shared between 1D binning and 2D binning:
        # 1. For 1D binning, they both are y coordinates;
        # 2. For 2D binning, center1 is the x coordinate and center2 is the
        #    y coordinate.
        self.n_bins1 = 0
        self.n_bins2 = 0
        self.center1 = None
        self.center2 = None

        self.label1 = None
        self.label2 = None

        self.iloc1 = -1
        self.fom1 = None
        self.vec1 = None
        self.iloc2 = -1
        self.fom2 = None
        self.vec2 = None

        self.vec_x = None
        self.vec_label = None

        self.fom12 = None

    def update_hist(self):
        n1 = self.n_bins1
        n2 = self.n_bins2

        # reset and initialization
        if self.reset1:
            self.__class__.fom1_hist = np.zeros(n1, dtype=np.float32)
            self.__class__.count1_hist = np.zeros(n1, dtype=np.uint32)
            # Real initialization could take place later then valid vec
            # is received.
            self.__class__.vec1_hist = None

        if self.reset2:
            self.__class__.fom2_hist = np.zeros(n2, dtype=np.float32)
            self.__class__.count2_hist = np.zeros(n2, dtype=np.uint32)
            # Real initialization could take place later then valid vec
            # is received.
            self.__class__.vec2_hist = None

        if (self.reset1 or self.reset2) and n1 > 0 and n2 > 0:
            self.__class__.fom12_hist = np.zeros((n2, n1), dtype=np.float32)
            self.__class__.count12_hist = np.zeros((n2, n1), dtype=np.float32)

        # update history

        if 0 <= self.iloc1 < n1:
            self.__class__.count1_hist[self.iloc1] += 1
            self.__class__.fom1_hist[self.iloc1] = self.fom1

            if self.vec1 is not None:
                if self.vec1_hist is None or len(self.vec_x) != self.vec1_hist.shape[0]:
                    # initialization
                    self.__class__.vec1_hist = np.zeros(
                        (len(self.vec_x), n1), dtype=np.float32)

                self.__class__.vec1_hist[:, self.iloc1] = self.vec1

        if 0 <= self.iloc2 < n2:
            self.__class__.count2_hist[self.iloc2] += 1
            self.__class__.fom2_hist[self.iloc2] = self.fom2

            if self.vec2 is not None:
                if self.vec2_hist is None or len(self.vec_x) != self.vec2_hist.shape[0]:
                    # initialization
                    self.__class__.vec2_hist = np.zeros(
                        (len(self.vec_x), n2), dtype=np.float32)

                self.__class__.vec2_hist[:, self.iloc2] = self.vec2

        if 0 <= self.iloc1 < n1 and 0 <= self.iloc2 < n2:
            self.__class__.count12_hist[self.iloc2, self.iloc1] += 1
            self.__class__.fom12_hist[self.iloc2, self.iloc1] = self.fom12


class CorrelationData:
    """Correlation data model."""

    class CorrelationDataMeta(type):
        def __init__(cls, *args, **kwargs):
            super().__init__(*args, **kwargs)
            setattr(cls, 'hist', PairData(
                device_id="", property="", resolution=0.0))

    class CorrelationDataItem(metaclass=CorrelationDataMeta):
        def __init__(self):
            self.x = None
            self.y = None  # FOM
            self.device_id = ""
            self.property = ""
            self.resolution = 0.0

            self.reset = False

        def update_params(self, x, y, device_id, property, resolution):
            self.x = x
            self.y = y
            self.device_id = device_id
            self.property = property
            self.resolution = resolution

        def update_hist(self):
            if self.reset:
                del self.hist

            _, _, info = self.hist

            if self.device_id != info['device_id'] \
                    or self.property != info['property'] \
                    or self.resolution != info['resolution']:
                if self.resolution > 0:
                    self.__class__.hist = AccumulatedPairData(
                        device_id=self.device_id,
                        property=self.property,
                        resolution=self.resolution)
                else:
                    self.__class__.hist = PairData(
                        device_id=self.device_id,
                        property=self.property,
                        resolution=0.0)

            self.hist = (self.x, self.y)

    class Correlation1(CorrelationDataItem):
        pass

    class Correlation2(CorrelationDataItem):
        pass

    class Correlation3(CorrelationDataItem):
        pass

    class Correlation4(CorrelationDataItem):
        pass

    def __init__(self):
        self.correlation1 = self.Correlation1()
        self.correlation2 = self.Correlation2()
        self.correlation3 = self.Correlation3()
        self.correlation4 = self.Correlation4()

    def update_hist(self):
        self.correlation1.update_hist()
        self.correlation2.update_hist()
        self.correlation3.update_hist()
        self.correlation4.update_hist()


class ProcessedData:
    """A class which stores the processed data.

    ProcessedData also provide interface for manipulating the other node
    dataset, e.g. RoiData, CorrelationData, PumpProbeData.

    Attributes:
        _tid (int): train ID.
        _image (ImageData): image data.
        xgm (XgmData): XGM train-resolved data.
        ai (AzimuthalIntegrationData): azimuthal integration train-resolved data.
        pp (PumpProbeData): pump-probe train-resolved data.
        roi (RoiData): ROI train-resolved data.
        xas (XasData): XAS train-resolved data.

        correlation (CorrelationData): correlation related data.
        bin (BinData): binning data.
    """
    def __init__(self, tid, images, **kwargs):
        """Initialization."""
        self._tid = tid  # train ID

        self._image = ImageData(images, **kwargs)

        self.roi = RoiData()
        self.xgm = XgmData()
        self.ai = AzimuthalIntegrationData()
        self.pp = PumpProbeData()
        self.xas = XasData()

        self.corr = CorrelationData()
        self.bin = BinData()

    @property
    def tid(self):
        return self._tid

    @property
    def image(self):
        return self._image

    @property
    def pulse_resolved(self):
        return self.image.pulse_resolved

    @property
    def n_pulses(self):
        return self._image.n_images

    def update(self):
        self.pp.update_hist(self._tid)

        self.corr.update_hist()
        self.bin.update_hist()
