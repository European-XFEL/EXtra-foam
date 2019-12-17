"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from abc import ABC, abstractmethod
import copy
from threading import Lock

import numpy as np

from ..config import config, AnalysisType, PumpProbeMode

from extra_foam.algorithms import (
    intersection, nanmeanImageArray, movingAverageImage,
    movingAverageImageArray, mask_image
)


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


class MovingAverageScalar:
    """Stores moving average of a scalar number."""

    def __init__(self, window=1):
        """Initialization.

        :param int window: moving average window size.
        """
        self._data = None  # moving average

        if not isinstance(window, int) or window < 0:
            raise ValueError("Window must be a positive integer.")

        self._window = window
        self._count = 0

    def __get__(self, instance, instance_type):
        if instance is None:
            return self

        return self._data

    def __set__(self, instance, data):
        if data is None:
            return

        if self._data is not None and self._window > 1 and \
                self._count <= self._window:
            if self._count < self._window:
                self._count += 1
                self._data += (data - self._data) / self._count
            else:  # self._count == self._window
                # here is an approximation
                self._data += (data - self._data) / self._count
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
            raise ValueError("Window must be a positive integer.")

        self._window = v

    @property
    def count(self):
        return self._count


class MovingAverageArray:
    """Stores moving average of 2D/3D (and higher dimension) array data."""

    def __init__(self, window=1):
        """Initialization.

        :param int window: moving average window size.
        """
        self._data = None  # moving average

        if not isinstance(window, int) or window < 0:
            raise ValueError("Window must be a positive integer.")

        self._window = window
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
                if data.ndim == 2:
                    movingAverageImage(self._data, data, self._count)
                elif data.ndim == 3:
                    movingAverageImageArray(self._data, data, self._count)
                else:
                    self._data += (data - self._data) / self._count
            else:  # self._count == self._window
                # here is an approximation
                if data.ndim == 2:
                    movingAverageImage(self._data, data, self._count)
                elif data.ndim == 3:
                    movingAverageImageArray(self._data, data, self._count)
                else:
                    self._data += (data - self._data) / self._count
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
            raise ValueError("Window must be a positive integer.")

        self._window = v

    @property
    def count(self):
        return self._count


class RawImageData(MovingAverageArray):
    """Stores moving average of raw images."""

    def __init__(self, window=1):
        super().__init__(window)

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
        y (numpy.array): Vector figure-of-merit.
        fom (float): Figure-of-merit.
        _x_label (str): label of x.
        _y_label (str): label of VFOM.
    """
    def __init__(self):
        self.x = None
        self.y = None
        self.fom = None

        self._x_label = ""
        self._y_label = ""

    @property
    def x_label(self):
        return self._x_label

    @property
    def y_label(self):
        return self._y_label


class AzimuthalIntegrationData(DataItem):
    """Azimuthal integration data item."""

    def __init__(self):
        super().__init__()

        self._x_label = "Momentum transfer (1/A)",
        self._y_label = "Scattering signal (arb.u.)"


class _RoiGeomBase(ABC):

    INVALID = None

    """RoiGeom base class."""
    def __init__(self):
        pass

    @abstractmethod
    def rect(self, data, copy=False):
        """Return a bounding box which includes the ROI.

        For a non-rectangular ROI, the bounding box should be a minimum
        rectangle to include the ROI with pixels not belong to the ROI
        set to None.

        :param numpy.ndarray data: a single image data or an array of image
            data.
        :param bool copy: True for copy the ROI data and False for not.
            Default = False.

        :return numpy.ndarray: a single ROI data or an array of ROI data.
            If the ROI is not activated or there is no intersection area
            between the ROI and the image, return None.
        """
        pass

    @property
    @abstractmethod
    def geometry(self):
        """Get the ROI geometry."""
        pass

    @geometry.setter
    @abstractmethod
    def geometry(self, v):
        """Set the ROI geometry.

        :param list/tuple v: a list of numbers which are used to describe
            the geometry of the ROI. For instance, (x, y, w, h) for
            a rectangular ROI and (x, y, r) for a circular ROI.
        """
        pass

    @abstractmethod
    def __repr__(self):
        pass


class RectRoiGeom(_RoiGeomBase):
    """RectRoiGeom class."""

    INVALID = [0, 0, -1, -1]

    def __init__(self):
        super().__init__()

        self._x, self._y, self._w, self._h = self.INVALID

    def rect(self, data, copy=False):
        """Overload."""
        x, y, w, h = self._x, self._y, self._w, self._h
        if x < 0 or y < 0 or w < 0 or h < 0:
            return None
        return np.array(data[..., y:y + h, x:x + w], copy=copy)

    @property
    def geometry(self):
        """Overload."""
        return self._x, self._y, self._w, self._h

    @geometry.setter
    def geometry(self, v):
        """Overload."""
        self._x, self._y, self._w, self._h = v

    def __repr__(self):
        return f"{self.__class__.__name__}(" \
               f"{self._x}, {self._y}, {self._w}, {self._h})"


class CircleRoiGeom(_RoiGeomBase):
    """CircleRoiItem class."""

    INVALID = (0, 0, -1)

    def __init__(self):
        super().__init__()

        self._x, self._y, self._r = self.INVALID

    def rect(self, data, copy=False):
        """Overload."""
        raise NotImplemented

    @property
    def geometry(self):
        """Overload."""
        return self._x, self._y, self._r

    @geometry.setter
    def geometry(self, v):
        """Overload."""
        self._x, self._y, self._r = v

    def __repr__(self):
        return f"{self.__class__.__name__}({self._x}, {self._y}, {self._r})"


class RoiData(DataItem):
    """RoiData class.

    Attributes:
        geom1, geom2, geom3, geom4 (RectRoiGeom): ROI geometry.
        norm (float): ROI normalizer.
        proj (RoiProjData): ROI projection data item
    """

    N_ROIS = len(config['ROI_COLORS'])

    class RoiProjData(DataItem):
        """ROI projection data item."""

        def __init__(self):
            super().__init__()

            self._x_label = "Pixel",
            self._y_label = "Projection"

    def __init__(self):
        super().__init__()

        self.geom1 = RectRoiGeom()
        self.geom2 = RectRoiGeom()
        self.geom3 = RectRoiGeom()
        self.geom4 = RectRoiGeom()

        self.norm = None

        # ROI projection
        self.proj = self.RoiProjData()


class PumpProbeData(DataItem):
    """Pump-probe data."""

    fom_hist = PairData()

    def __init__(self):
        super().__init__()

        self.analysis_type = AnalysisType.UNDEFINED

        self.mode = PumpProbeMode.UNDEFINED

        # on/off pulse indices
        self.indices_on = None
        self.indices_off = None

        # on/off images averaged over a train
        self.image_on = None
        self.image_off = None

        # on/off ROI normalizer averaged over a train
        self.roi_norm_on = None
        self.roi_norm_off = None

        # on/off normalized VFOM
        # Note: VFOM = normalized on-VFOM - normalized off-VFOM
        self.y_on = None
        self.y_off = None

        self.abs_difference = True

        self.reset = False

    def update_hist(self, tid):
        if self.reset:
            del self.fom_hist

        self.fom_hist = (tid, self.fom)


class ImageData:
    """Image data model.

    Attributes:
        pixel_size (float): pixel size (in meter) of the detector.
        images (list): a list of pulse images in the train. A value of
            None only indicates that the corresponding pulse image is
            not needed (in the main process).
        n_images (int): number of images in the train.
        sliced_indices (list): a list of indices which is selected by
            pulse slicer. The slicing is applied before applying any pulse
            filters to select pulses with a certain pattern. It can be used
            to reconstruct the indices of the selected images in the original
            data providing the number of pulses and the slicer are both known.
        poi_indices (list): indices of pulses of interest.
        background (float): a uniform background value.
        dark_mean (numpy.ndaray): average of all the dark images in
            the dark run. Shape = (y, x)
        n_dark_pulses (int): number of dark pulses in a dark train.
        dark_count (int): count of collected dark trains.
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

        self.sliced_indices = None
        self.poi_indices = None

        self.background = None
        self.dark_mean = None
        self.n_dark_pulses = 0
        self.dark_count = 0
        self.image_mask = None
        self.threshold_mask = None

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

        return len(self.images)

    @classmethod
    def from_array(cls, arr, *,
                   background=0.0,
                   image_mask=None,
                   threshold_mask=None,
                   sliced_indices=None,
                   poi_indices=None):
        """Construct a self-consistent ImageData."""
        if not isinstance(arr, np.ndarray):
            raise TypeError(r"Image data must be numpy.ndarray!")

        if arr.ndim <= 1 or arr.ndim > 3:
            raise ValueError(f"The shape of image data must be (y, x) or "
                             f"(n_pulses, y, x)!")

        image_dtype = config['IMAGE_DTYPE']
        if arr.dtype != image_dtype:
            arr = arr.astype(image_dtype)

        instance = cls()

        if arr.ndim == 3:
            n_images = len(arr)
            instance.images = [None] * n_images
            if poi_indices is None:
                poi_indices = [0, 0]
            for i in poi_indices:
                instance.images[i] = arr[i]

            instance.mean = nanmeanImageArray(arr)

            if sliced_indices is None:
                instance.sliced_indices = list(range(n_images))
            else:
                sliced_indices = list(set(sliced_indices))
                n_indices = len(sliced_indices)
                if n_indices != n_images:
                    raise ValueError(f"Length of sliced indices {sliced_indices} "
                                     f"!= number of images {n_images}")
                instance.sliced_indices = sliced_indices
        else:
            instance.images = [None]

            instance.mean = arr

            if sliced_indices is not None:
                raise ValueError("Train-resolved data does not support "
                                 "'sliced_indices'!")
            instance.sliced_indices = [0]  # be consistent

        instance.poi_indices = poi_indices

        instance.masked_mean = instance.mean.copy()
        mask_image(instance.masked_mean,
                   image_mask=image_mask,
                   threshold_mask=threshold_mask)
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

    def __init__(self):
        self.correlation1 = self.Correlation1()
        self.correlation2 = self.Correlation2()

    def update_hist(self):
        self.correlation1.update_hist()
        self.correlation2.update_hist()


class StatisticsData:
    """Statistics data model.

    Attributes:
        fom_hist (np.array): 1D array for pulse resolved FOMs in a train
        fom_bin_center (np.array): 1D array for bins centers
        fom_count (np.array): 1D array for counts in each bin.
        poi_fom_bin_center (list): a list of histogram bin centers for
            individual pulses in a train.
        poi_fom_count (list): a list of histogram bin counts for individual
            pulses in a train.
    """

    def __init__(self):
        self.fom_hist = None
        self.fom_bin_center = None
        self.fom_count = None

        self.poi_fom_bin_center = None
        self.poi_fom_count = None


class PulseIndexMask:
    LENGTH = config["MAX_N_PULSES_PER_TRAIN"]

    def __init__(self):
        self._indices = np.array([True] * self.LENGTH)

    def mask(self, idx):
        """Mask a given index/list of indices."""
        self._indices[idx] = False

    def n_kept(self, n):
        """Return number of kept indices.

        :param int n: total number of kept pulses.
        """
        return np.sum(self._indices[:n])

    def n_dropped(self, n):
        """Return number of dropped indices.

        :param int n: total number of dropped pulses.
        """
        return n - self.n_kept(n)

    def dropped_indices(self, n):
        """Return a list of dropped indices.

        :param int n: total number of pulses.
        """
        return np.where(~self._indices[:n])[0]

    def kept_indices(self, n):
        """Return a list of kept indices.

        :param int n: total number of pulses.
        """
        return np.where(self._indices[:n])[0]


class XasData:
    def __init__(self):
        self.delay_bin_centers = None
        self.delay_bin_counts = None
        self.a13_stats = None
        self.a23_stats = None
        self.a21_stats = None
        self.energy_bin_centers = None
        self.a21_heat = None
        self.a21_heatcount = None


class _XgmDataItem:
    """_XgmDataItem class.

    Store XGM pipeline data.
    """
    def __init__(self):
        self.intensity = None  # FEL intensity
        self.x = None  # x position
        self.y = None  # y position


class XgmData(_XgmDataItem):
    """XgmData class.

    Store XGM pipeline data.
    """
    def __init__(self):
        super().__init__()

        self.on = _XgmDataItem()
        self.off = _XgmDataItem()


class ProcessedData:
    """A class which stores the processed data.

    Attributes:
        tid (int): train ID.
        image (ImageData): image data.
        xgm (XgmData): XGM train-resolved data.
        ai (AzimuthalIntegrationData): azimuthal integration train-resolved data.
        pp (PumpProbeData): pump-probe train-resolved data.
        roi (RoiData): ROI train-resolved data.

        st (StatisticsData): statistics data.
        correlation (CorrelationData): correlation data.
        bin (BinData): binning data.
    """
    class PulseData:
        """Container for pulse-resolved data."""

        def __init__(self):
            self.ai = AzimuthalIntegrationData()
            self.roi = RoiData()
            self.xgm = XgmData()

    def __init__(self, tid):
        """Initialization."""
        self._tid = tid  # train ID

        self.pidx = PulseIndexMask()

        self.image = ImageData()

        self.xgm = XgmData()

        self.roi = RoiData()
        self.ai = AzimuthalIntegrationData()
        self.pp = PumpProbeData()

        self.st = StatisticsData()
        self.corr = CorrelationData()
        self.bin = BinData()

        self.trxas = XasData()

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
