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
from ..config import AnalysisType, config

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
        x_label (str): label for x of VFOM.
        vfom_label (str): label for VFOM.
        has_vfom (bool): whether VFOM exists.
    """
    def __init__(self, *, x_label="", vfom_label="", has_vfom=True):
        self.x = None
        self.vfom = None
        self.fom = None

        self.x_label = x_label
        self.vfom_label = vfom_label

        self.has_vfom = has_vfom


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

        self.roi1 = DataItem(has_vfom=False)
        self.roi2 = DataItem(has_vfom=False)
        self.roi1_sub_roi2 = DataItem(has_vfom=False)
        self.roi1_add_roi2 = DataItem(has_vfom=False)

        # 1. pump-probe 'proj' will directly go to ProcessedData.pp;
        # 2. we may need two projection types at the same time;
        self.proj1 = DataItem(x_label='pixel', vfom_label='ROI1 projection')
        self.proj2 = DataItem(x_label='pixel', vfom_label='ROI2 projection')
        self.proj1_sub_proj2 = DataItem(
            x_label='pixel', vfom_label='ROI1 - ROI2 projection')
        self.proj1_add_proj2 = DataItem(
            x_label='pixel', vfom_label='ROI1 + ROI2 projection')


class AzimuthalIntegrationData(DataItem):
    """Train-resolved azimuthal integration data."""

    def __init__(self,
                 x_label="Momentum transfer (1/A)",
                 vfom_label="Scattering signal (arb.u.)"):
        super().__init__(x_label=x_label, vfom_label=vfom_label)


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
        pixel_size (float): pixel size of the detector.
        images (list): a list of pulse images in the train. A value of
            None only indicates that the corresponding pulse image is
            not needed (in the main process).
        ma_count (int): current moving average count.
        n_images (int): number of images in the train.
        poi_indices (list): indices of pulses of interest.
        background (float): a uniform background value.
        image_mask (numpy.ndarray): image mask with dtype=np.bool.
        threshold_mask (tuple): (lower, upper) boundaries of the
            threshold mask.
        reference (numpy.ndarray): reference image.
        mean (numpy.ndarray): average image over the train.
        masked_mean (numpy.ndarray): average image over the train with
            threshold mask applied.
    """

    def __init__(self):
        self._pixel_size = config['PIXEL_SIZE']

        self.images = None
        self.ma_count = 0

        self.poi_indices = [0, 0]

        self.background = None
        self.image_mask = None
        self.threshold_mask = None

        self.index_mask = None

        self.reference = None

        self.mean = None
        self.masked_mean = None

    @property
    def pixel_size(self):
        return self._pixel_size

    @property
    def n_images(self):
        if self.images is None:
            return 0

        if isinstance(self.images, list):
            return len(self.images)
        return 1

    @classmethod
    def from_array(cls, arr, *,
                   ma_count=1.0,
                   background=0.0,
                   image_mask=None,
                   threshold_mask=None,
                   poi_indices=None):
        """Construct a self-consistant ImageData."""
        if arr is not None:
            if not isinstance(arr, np.ndarray):
                raise TypeError(r"Image data must be numpy.ndarray!")

            if arr.ndim <= 1 or arr.ndim > 3:
                raise ValueError(f"The shape of image data must be (y, x) or "
                                 f"(n_pulses, y, x)!")

            image_dtype = config['IMAGE_DTYPE']
            if arr.dtype != image_dtype:
                # FIXME: dtype of the incoming data could be integer, but integer
                #        array does not have nanmean.
                arr = arr.astype(image_dtype)

        instance = cls()
        instance.images = arr
        instance.ma_count = ma_count

        if arr.ndim == 3:
            arr_mean = xt_nanmean_images(arr)
            image_list = [None] * len(arr)
            if poi_indices is None:
                poi_indices = [0, 0]
            for i in poi_indices:
                image_list[i] = arr[i]
            instance.images = image_list
        else:
            arr_mean = arr

        instance.poi_indices = poi_indices

        instance.mean = arr_mean
        instance.masked_mean = mask_image(arr_mean,
                                          threshold_mask=threshold_mask,
                                          image_mask=image_mask,
                                          inplace=False)
        instance.image_mask = image_mask
        instance.threshold_mask = threshold_mask
        instance.background = background

        return instance


class BinData:
    """Binning data model."""

    class Bin1dDataItem:
        def __init__(self):
            # bin center
            self.center = None
            # bin label
            self.label = None

            # FOM histogram
            self.fom_hist = None
            # FOM count histogram
            self.count_hist = None
            # VFOM heatmap
            self.vfom_heat = None

            # whether the analysis type has VFOM
            self.has_vfom = True
            # x coordinate of VFOM
            self.x = None
            # label for x
            self.x_label = ""
            # label for VFOM
            self.vfom_label = ""

            self.reset = False
            self.updated = False

    class Bin2dDataItem:
        def __init__(self):
            # bin center x
            self.center_x = None
            # bin center y
            self.center_y = None
            # bin label x
            self.x_label = ""
            # bin label y
            self.y_label = ""

            # FOM 2D heatmap
            self.fom_heat = None
            # FOM 2D count
            self.count_heat = None

            self.reset = False
            self.updated = False

    def __init__(self):
        super().__init__()

        self.mode = None

        self.bin1 = self.Bin1dDataItem()
        self.bin2 = self.Bin1dDataItem()

        # bin1 -> x, bin2 -> y
        self.bin12 = self.Bin2dDataItem()


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

            # if self.y is not None:
            # self.x must not be None
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


class StatisticsData:
    """Statistics data model.

    Attributes:
        fom_list (np.array): 1D array for pulse resolved FOMs in a train
        fom_bin_center (np.array): 1D array for bins centers
        fom_counts (np.array): 1D array for counts in each bin.
    """

    def __init__(self):
        self.fom_list = None
        self.fom_bin_center = np.array([])
        self.fom_counts = np.array([])


class ProcessedData:
    """A class which stores the processed data.

    Attributes:
        tid (int): train ID.
        image (ImageData): image data.
        xgm (XgmData): XGM train-resolved data.
        ai (AzimuthalIntegrationData): azimuthal integration train-resolved data.
        pp (PumpProbeData): pump-probe train-resolved data.
        roi (RoiData): ROI train-resolved data.
        xas (XasData): XAS train-resolved data.

        st (StatisticsData): statistics data.
        correlation (CorrelationData): correlation data.
        bin (BinData): binning data.
    """
    class PulseData:
        """Container for pulse-resolved data."""

        def __init__(self):
            self.ai = AzimuthalIntegrationData()
            self.roi = RoiData()

    def __init__(self, tid):
        """Initialization."""
        self._tid = tid  # train ID

        self.image = ImageData()

        self.roi = RoiData()
        self.xgm = XgmData()
        self.ai = AzimuthalIntegrationData()
        self.pp = PumpProbeData()
        self.xas = XasData()

        self.st = StatisticsData()
        self.corr = CorrelationData()
        self.bin = BinData()

        self.pulse = self.PulseData()

    @property
    def tid(self):
        return self._tid

    @property
    def n_pulses(self):
        return self.image.n_images

    @property
    def pulse_resolved(self):
        return config['PULSE_RESOLVED']

    def update(self):
        self.pp.update_hist(self._tid)

        self.corr.update_hist()
