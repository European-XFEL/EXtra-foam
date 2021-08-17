"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from abc import ABC, abstractmethod
import collections
from collections import namedtuple

import numpy as np

from ..config import config, AnalysisType, PumpProbeMode, ImageTransformType

from extra_foam.algorithms import (
    intersection, movingAvgImageData, mask_image_data, nanmean_image_data
)


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
            self._data = None
            self._count = 0
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

    def __init__(self, window=1, *, copy_first=False):
        """Initialization.

        :param int window: moving average window size.
        :param bool copy_first: True for copy the first data.
        """
        self._data = None  # moving average

        if not isinstance(window, int) or window < 0:
            raise ValueError("Window must be a positive integer.")

        self._window = window
        self._count = 0

        self._copy_first = copy_first

    def __get__(self, instance, instance_type):
        if instance is None:
            return self

        return self._data

    def __set__(self, instance, data):
        self.update(data)

    def __delete__(self, instance):
        self.clear()

    def update(self, data):
        if data is None:
            self._data = None
            self._count = 0
            return

        if self._data is not None and self._window > 1 and \
                self._count <= self._window and data.shape == self._data.shape:
            if self._count < self._window:
                self._count += 1
                if data.ndim in (2, 3):
                    movingAvgImageData(self._data, data, self._count)
                else:
                    self._data += (data - self._data) / self._count
            else:  # self._count == self._window
                # here is an approximation
                if data.ndim in (2, 3):
                    movingAvgImageData(self._data, data, self._count)
                else:
                    self._data += (data - self._data) / self._count
        else:
            self._data = data.copy() if self._copy_first else data
            self._count = 1

    def clear(self):
        self._data = None
        self._count = 0

    @property
    def data(self):
        return self._data

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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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

    Attributes:
        x (numpy.array): x coordinate of VFOM.
        y (numpy.array): Vector figure-of-merit.
        fom (float): Figure-of-merit.
    """

    __slots__ = ['x', 'y', 'fom']

    def __init__(self):
        self.x = None
        self.y = None
        self.fom = None


class AzimuthalIntegrationData(DataItem):
    """Azimuthal integration data item."""
    __slots__ = ['q_map', 'peaks', 'max_peak', 'max_peak_q', 'center_of_mass']

    def __init__(self):
        super().__init__()
        self.q_map = None
        self.peaks = None
        self.max_peak = None
        self.max_peak_q = None
        self.center_of_mass = None


class _RoiGeomBase(ABC):

    INVALID = None

    """RoiGeom base class."""
    def __init__(self):
        super().__init__()

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
        if w <= 0 or h <= 0:
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


class RoiDataPulse(DataItem):
    """Pulse-resolved ROI data.

    Attributes:
        geom1, geom2, geom3, geom4 (RectRoiGeom): ROI geometry.
        norm (float): pulse-resolved ROI normalizer.
        hist (HistogramDataPulse): pulse-resolved ROI histogram data
            item. Currently, only ROI histogram of POI pulses will
            be calculated.
    """

    N_ROIS = len(config['GUI_ROI_COLORS'])

    __slots__ = ['norm', 'hist']

    def __init__(self):
        super().__init__()

        self.norm = None
        self.hist = HistogramDataPulse()


class RoiDataTrain(DataItem):
    """Train-resolved ROI data.

    Attributes:
        geom1, geom2, geom3, geom4 (RectRoiGeom): ROI geometry.
        fom_slave (float): ROI slave FOM.
        norm (float): ROI normalizer.
        proj (RoiProjData): ROI projection data item
        hist (_HistogramDataItem): ROI histogram data item.
    """

    N_ROIS = len(config['GUI_ROI_COLORS'])

    __slots__ = ['geom1', 'geom2', 'geom3', 'geom4',
                 'fom_slave', 'norm', 'proj', 'hist']

    def __init__(self):
        super().__init__()

        self.geom1 = RectRoiGeom()
        self.geom2 = RectRoiGeom()
        self.geom3 = RectRoiGeom()
        self.geom4 = RectRoiGeom()

        self.fom_slave = None

        self.norm = None
        self.proj = DataItem()
        self.hist = _HistogramDataItem()


class PumpProbeData(AzimuthalIntegrationData):
    """Pump-probe data."""

    class OnOff:
        """on/off quantity averaged over a train."""
        __slots__ = ["mask",
                     "roi_norm",
                     "xgm_intensity",
                     "digitizer_pulse_integral"]

        def __init__(self):
            self.mask = None
            self.roi_norm = None
            self.xgm_intensity = None
            self.digitizer_pulse_integral = None

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

        # TODO: modify other attributes
        # on/off normalizer averaged over a train
        self.on = self.OnOff()
        self.off = self.OnOff()

        # on/off normalized VFOM
        # Note: VFOM = normalized on-VFOM - normalized off-VFOM
        self.y_on = None
        self.y_off = None

        self.abs_difference = True

        self.reset = False


class ImageData:
    """Image data model.

    Attributes:
        pixel_size (float): pixel size (in meter) of the detector.
        images (list): a list of pulse images in the train. A value of
            None only indicates that the corresponding pulse image is
            not needed (in the main process).
        mean (numpy.ndarray): average image over the train.
        masked_mean (numpy.ndarray): average image over the train with
            both image mask and threshold mask applied.
        n_images (int): number of images in the train.
        sliced_indices (list): a list of indices which is selected by
            pulse slicer. The slicing is applied before applying any pulse
            filters to select pulses with a certain pattern. It can be used
            to reconstruct the indices of the selected images in the original
            data providing the number of pulses and the slicer are both known.
        poi_indices (list): indices of pulses of interest.
        gain_mean (numpy.ndarray): average of all the gain data in the
            selected memory cells. Shape = (y, x)
        offset_mean (numpy.ndarray):  average of all the offset data in the
            selected memory cells. Shape = (y, x)
        n_dark_pulses (int): number of dark pulses in a dark train.
        dark_mean (numpy.ndarray): average of the dark run. Shape = (y, x)
        dark_count (int): count of collected dark trains.
        image_mask (numpy.ndarray): image mask. For pulse-resolved detectors,
            this image mask is shared by all the pulses in a train. However,
            their overall mask could still be different after applying the
            threshold mask. Shape = (y, x), dtype = np.bool
        image_mask_in_modules (numpy.ndarray): image mask in modules. Only
            used for detectors which require geometry to assemble multiple
            modules.
        threshold_mask (tuple): (lower, upper) of the threshold.
            It should be noted that a pixel with value outside of the boundary
            will be masked as Nan/0, depending on the masking policy.
        mask (numpy.ndarray): overall mask for the average image.
            Shape = (y, x), dtype = np.bool
        transform_type (ImageTransformType): image transform type.
        transformed (numpy.ndarray): transformed image.
        reference (numpy.ndarray): reference image.
    """

    __slots__ = ["_pixel_size",
                 "images", "mean", "masked_mean", "featured",
                 "sliced_indices", "poi_indices",
                 "gain_mean", "offset_mean",
                 "n_dark_pulses", "dark_mean", "dark_count",
                 "image_mask", "image_mask_in_modules", "threshold_mask", "mask",
                 "reference",
                 "transform_type", "transformed"]

    def __init__(self):
        self._pixel_size = config['PIXEL_SIZE']

        self.images = None
        self.mean = None
        self.masked_mean = None
        self.featured = None

        self.sliced_indices = None
        self.poi_indices = None

        self.gain_mean = None
        self.offset_mean = None

        self.n_dark_pulses = 0
        self.dark_mean = None
        self.dark_count = 0

        self.image_mask = None
        self.image_mask_in_modules = None
        self.threshold_mask = None
        self.mask = None

        self.reference = None

        self.transform_type = ImageTransformType.UNDEFINED
        self.transformed = None

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

        image_dtype = config['SOURCE_PROC_IMAGE_DTYPE']
        if arr.dtype != image_dtype:
            arr = arr.astype(image_dtype)

        instance = cls()

        if arr.ndim == 3:
            n_images = len(arr)
            instance.images = [None] * n_images
            if poi_indices is None:
                poi_indices = [0, 0]
            for i in poi_indices:
                instance.images[i] = arr[i].copy()

            instance.mean = nanmean_image_data(arr)

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

        if poi_indices is None:
            poi_indices = [0, 0]
        instance.poi_indices = poi_indices

        instance.masked_mean = instance.mean.copy()
        if image_mask is None:
            image_mask = np.zeros(arr.shape[-2:], dtype=bool)
        instance.image_mask = image_mask

        instance.mask = np.zeros_like(image_mask)
        mask_image_data(instance.masked_mean,
                        image_mask=image_mask,
                        threshold_mask=threshold_mask,
                        out=instance.mask)

        if arr.ndim == 3:
            for idx in poi_indices:
                mask_image_data(instance.images[idx],
                                image_mask=image_mask,
                                threshold_mask=threshold_mask)

        instance.threshold_mask = threshold_mask

        return instance


_BinDataItem = namedtuple('_BinDataItem', ['device_id', 'property',
                                           'centers', 'counts',
                                           'stats', 'x', 'heat'])


class BinData(collections.abc.Mapping):
    """Binning data model."""

    class BinDataItem:

        __slots__ = ['source', 'size', 'centers', 'counts', 'stats',
                     'x', 'heat', 'x_range']

        def __init__(self):
            self.source = ""
            self.centers = None
            # bin size, needed when there is only one center
            self.size = None
            self.counts = None
            self.stats = None
            self.x = None
            self.heat = None
            self.x_range = None

    __slots__ = ['mode', '_common', 'heat', 'heat_count']

    _N_BINS = 2

    def __init__(self):
        self.mode = None

        self._common = []
        for i in range(self._N_BINS):
            self._common.append(self.BinDataItem())

        self.heat = None
        self.heat_count = None

    def __contains__(self, idx):
        """Override."""
        return 0 <= idx < self._N_BINS

    def __getitem__(self, idx):
        """Overload."""
        return self._common.__getitem__(idx)

    def __iter__(self):
        """Overload."""
        return self._common.__iter__()

    def __len__(self):
        """overload."""
        return self._N_BINS


class CorrelationData(collections.abc.Mapping):
    """Correlation data model."""

    class CorrelationDataItem:

        __slots__ = ['x', 'y', 'x_slave', 'y_slave', 'source', 'resolution']

        def __init__(self):
            self.x = None
            self.y = None  # FOM
            self.x_slave = None
            self.y_slave = None  # FOM slave
            self.source = ""
            self.resolution = 0.0

    __slots__ = ['_common', '_pp']

    _N_CORRELATIONS = 2

    def __init__(self):
        self._common = []
        for i in range(self._N_CORRELATIONS):
            self._common.append(self.CorrelationDataItem())

        # pump-probe data
        self._pp = self.CorrelationDataItem()

    def __contains__(self, idx):
        """Override."""
        return 0 <= idx < self._N_CORRELATIONS

    def __getitem__(self, idx):
        """Overload."""
        return self._common.__getitem__(idx)

    def __iter__(self):
        """Overload."""
        return self._common.__iter__()

    def __len__(self):
        """overload."""
        return self._N_CORRELATIONS

    @property
    def pp(self):
        return self._pp


class _HistogramDataItem:

    __slots__ = ['hist', 'bin_centers', 'mean', 'median', 'std']

    def __init__(self):
        self.hist = None
        self.bin_centers = None
        self.mean = None
        self.median = None
        self.std = None


class HistogramDataTrain(_HistogramDataItem):
    pass


class HistogramDataPulse(collections.abc.MutableMapping):
    """Pulse-resolved histogram data model.

    Attributes:
        pulse_foms (np.array): pulse-resolved FOMs in a train.
    """

    __slots__ = ['pulse_foms', '_data']

    def __init__(self):
        self.pulse_foms = None

        self._data = dict()

    def __getitem__(self, key):
        """Overload."""
        return self._data[key]

    def __setitem__(self, key, v):
        """Overload."""
        max_k = config["MAX_N_PULSES_PER_TRAIN"]
        if not isinstance(key, int) or key < 0 or key >= max_k:
            raise KeyError("key must be an integer within [0, {max_k})!")

        item = _HistogramDataItem()
        item.hist, item.bin_centers, item.mean, item.median, item.std = v
        self._data[key] = item

    def __delitem__(self, key):
        """Overload."""
        del self._data[key]

    def __iter__(self):
        """Overload."""
        return iter(self._data)

    def __len__(self):
        """overload."""
        return len(self._data)


class PulseIndexMask:
    LENGTH = config["MAX_N_PULSES_PER_TRAIN"]

    def __init__(self):
        self._indices = np.array([True] * self.LENGTH, dtype=bool)

    def mask_by_index(self, a):
        """Mask by indices.

        :param int/iterable a: indices to be masked.
        """
        self._indices[a] = False

    def mask_by_array(self, a):
        """Mask by boolean array.

        :param numpy.array a: a boolean array.
        """
        self._indices[:len(a)][a] = False

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

    def reset(self):
        self._indices = np.array([True] * self.LENGTH)


class _XgmDataItem:
    """_XgmDataItem class.

    Store XGM pipeline data.
    """

    __slots__ = ['intensity', 'x', 'y']

    def __init__(self):
        self.intensity = None  # FEL intensity
        self.x = None  # x position
        self.y = None  # y position


class XgmData(_XgmDataItem):
    """XgmData class.

    Store XGM pipeline data.
    """

    __slots__ = []

    def __init__(self):
        super().__init__()


class _DigitizerDataItem:
    """_DigitizerDataItem class.

    Store Digitizer pipeline data.
    """

    __slots__ = ['pulse_integral']

    def __init__(self):
        self.pulse_integral = None


class _DigitizerChannelData(collections.abc.Mapping):
    """_DigitizerDataItem class.

    Store Digitizer pipeline data.
    """

    # 'A', 'B', 'C' and 'D' are for AdqDigitizer while
    # 'ADC' is for FastAdc. The final interface for the
    # digitizer will be determined later based on the
    # feature requests.
    _CHANNEL_NAMES = ('A', 'B', 'C', 'D', 'ADC')

    __slots__ = ['_pulse_integrals']

    def __init__(self):
        super().__init__()

        self._pulse_integrals = dict()
        for cn in self._CHANNEL_NAMES:
            self._pulse_integrals[cn] = _DigitizerDataItem()

    def __contains__(self, cn):
        """Override."""
        return self._CHANNEL_NAMES.__contains__(cn)

    def __getitem__(self, cn):
        """Overload."""
        return self._pulse_integrals.__getitem__(cn)

    def __iter__(self):
        """Overload."""
        return self._CHANNEL_NAMES.__iter__()

    def __len__(self):
        """overload."""
        return len(self._CHANNEL_NAMES)

    def items(self):
        return self._pulse_integrals.items()


class DigitizerData(_DigitizerChannelData):
    """DigitizerData class.

    Store Digitizer pipeline data.
    """

    __slots__ = ["ch_normalizer"]

    def __init__(self):
        super().__init__()
        # name of the channel as a normalizer
        self.ch_normalizer = self._CHANNEL_NAMES[0]


class BraggPeakData:
    __slots__ = ["roi", "roi_dims", "roi_intensity", "pulses",
                 "center_of_mass", "center_of_mass_stddev",
                 "pulse_intensity", "lineout_x", "lineout_y"]

    def __init__(self):
        self.roi = { }
        self.roi_dims = { }
        self.roi_intensity = { }
        self.pulses = { }
        self.center_of_mass = { }
        self.center_of_mass_stddev = { }
        self.pulse_intensity = { }
        self.lineout_x = { }
        self.lineout_y = { }

class ProcessedData:
    """A class which stores the processed data.

    Attributes:
        tid (int): train ID.
        image (ImageData): image data.
        xgm (XgmData): XGM train-resolved data.
        ai (AzimuthalIntegrationData): azimuthal integration train-resolved data.
        pp (PumpProbeData): pump-probe train-resolved data.
        roi (RoiData): ROI train-resolved data.

        hist (HistgramData): statistics data.
        correlation (CorrelationData): correlation data.
        bin (BinData): binning data.
    """
    class PulseData:
        """Container for pulse-resolved data."""

        __slots__ = ['ai', 'roi', 'xgm', 'digitizer',
                     'hist', 'bragg_peaks']

        def __init__(self):
            self.ai = AzimuthalIntegrationData()
            self.roi = RoiDataPulse()
            self.xgm = XgmData()
            self.digitizer = DigitizerData()
            self.hist = HistogramDataPulse()
            self.bragg_peaks = BraggPeakData()

    __slots__ = ['_tid', 'pidx', 'image',
                 'xgm', 'roi', 'ai', 'pp',
                 'hist', 'corr', 'bin',
                 'pulse']

    def __init__(self, tid):
        """Initialization."""
        self._tid = tid  # train ID

        self.pidx = PulseIndexMask()

        self.image = ImageData()

        self.xgm = XgmData()

        self.roi = RoiDataTrain()
        self.ai = AzimuthalIntegrationData()
        self.pp = PumpProbeData()

        self.hist = HistogramDataTrain()
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
