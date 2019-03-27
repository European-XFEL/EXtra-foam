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
import copy

import numpy as np

from cached_property import cached_property

from ..algorithms import nanmean_axis0_para
from ..logger import logger
from ..config import config, ImageMaskChange


class TrainData:
    """Store the history train data.

    Each data point is pair of data: (x, y).

    For correlation plots: x can be a train ID or a motor position,
    and y is the figure of merit (FOM).
    """
    MAX_LENGTH = 100000

    def __init__(self, **kwargs):
        # We need to have a 'x' for each sub-dataset due to the
        # concurrency of data processing.
        self._x = []
        self._y = []
        # for now it is used in CorrelationData to store device ID and
        # property information
        self._info = kwargs

    def __get__(self, instance, instance_type):
        if instance is None:
            return self
        # Note: here we must ensure that the data is not copied
        return self._x, self._y, copy.copy(self._info)

    def __set__(self, instance, pair):
        this_x, this_y = pair
        self._x.append(this_x)
        self._y.append(this_y)

        # This is a reasonable choice since we always wants to return a
        # reference in __get__!
        if len(self._x) > self.MAX_LENGTH:
            self.__delete__(instance)

    def __delete__(self, instance):
        del self._x[0]
        del self._y[0]

    def clear(self):
        self._x.clear()
        self._y.clear()
        # do not clear _info here!


class AccumulatedTrainData(TrainData):
    """Store the history accumulated train data.

    Each data point is pair of data: (x, DataStat).

    The data is collected in a stop-and-collected way. A motor,
    for example, will stop in a location and collect data for a
    period of time. Then,  each data point in the accumulated
    train data is the average of the data during this period.
    """
    class DataStat:
        """Statistic of data."""
        def __init__(self):
            self.count = []
            self.avg = []
            self.min = []
            self.max = []

        def add(self, v):
            self.count[-1] += 1
            self.avg[-1] += (v - self.avg[-1]) / self.count[-1]
            if v < self.min[-1]:
                self.min[-1] = v
            elif v > self.max[-1]:
                self.max[-1] = v

        def pop_last(self):
            del self.count[-1]
            del self.avg[-1]
            del self.min[-1]
            del self.max[-1]

        def pop_first(self):
            del self.count[0]
            del self.avg[0]
            del self.min[0]
            del self.max[0]

        def append(self, v):
            self.count.append(1)
            self.avg.append(v)
            self.min.append(v)
            self.max.append(v)

        def clear(self):
            self.count.clear()
            self.avg.clear()
            self.min.clear()
            self.max.clear()

    MAX_LENGTH = 5000

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

        self._y = self.DataStat()

    def __set__(self, instance, pair):
        this_x, this_y = pair
        if self._x:
            if abs(this_x - self._x[-1]) - self._resolution < self._epsilon:
                self._y.add(this_y)  # self._y.count will be updated
                self._x[-1] += (this_x - self._x[-1]) / self._y.count[-1]
            else:
                # If the number of data at a location is less than _min_count,
                # the data at this location will be discarded.
                if self._y.count[-1] < self._min_count:
                    del self._x[-1]
                    self._y.pop_last()
                self._x.append(this_x)
                self._y.append(this_y)
        else:
            self._x.append(this_x)
            self._y.append(this_y)

        if len(self._y.count) > self.MAX_LENGTH:
            self.__delete__(instance)

    def __get__(self, instance, instance_type):
        if instance is None:
            return self

        # Note: the cost of copy is acceptable as long as 'MAX_LENGTH'
        #       is 5000.
        x = copy.copy(self._x)
        y = copy.deepcopy(self._y)
        if y.count and y.count[-1] < self._min_count:
            del x[-1]
            y.pop_last()
        return x, y, copy.copy(self._info)

    def __delete__(self, instance):
        del self._x[0]
        self._y.pop_first()


class AbstractData(abc.ABC):
    @classmethod
    def clear(cls):
        for attr in cls.__dict__.values():
            if isinstance(attr, TrainData):
                # descriptor protocol will not be triggered here
                attr.clear()


class RoiData(AbstractData):
    """A class which stores ROI data."""

    # (sum/mean) histories of ROI1 and ROI2
    roi1_hist = TrainData()
    roi2_hist = TrainData()

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
    def add_param(cls, idx, device_id, ppt, resolution=0.0):
        param = f'param{idx}'
        if resolution:
            setattr(cls, param, AccumulatedTrainData(
                device_id=device_id, property=ppt, resolution=resolution))
        else:
            setattr(cls, param, TrainData(device_id=device_id, property=ppt))

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

    @classmethod
    def remove_params(cls):
        params = []
        for kls in cls.__dict__:
            if isinstance(cls.__dict__[kls], TrainData):
                params.append(kls)

        for param in params:
            delattr(cls, param)


class ImageData:
    """A class that manages the detector images.

    Operation flow:

    remove background -> cropping -> calculate mean image -> apply mask

    Attributes:
        pixel_size (float): detector pixel size.
        _images (numpy.ndarray): detector images for all the pulses in
            a train. shape = (pulse_id, y, x) for pulse-resolved
            detectors and shape = (y, x) for train-resolved detectors.
        _threshold_mask (tuple): (min, max) threshold of the pixel value.
        _image_mask (numpy.ndarray): an image mask, default = None.
            Shape = (y, x)
        _crop_area (tuple): (x, y, w, h) of the cropped image.
    """
    class RawImageData:
        def __init__(self):
            self._images = None  # moving average (original data)
            self._ma_window = 1
            self._ma_count = 0

            self._bkg = 0.0  # current background value

        def _invalid_image_cached(self):
            try:
                del self.__dict__['images']
            except KeyError:
                pass

        @property
        def background(self):
            return self._bkg

        @background.setter
        def background(self, v):
            if self._bkg != v:
                self._bkg = v
                self._invalid_image_cached()

        @cached_property
        def images(self):
            if self._images is None:
                return None
            # return a new constructed array
            return self._images - self._bkg

        def set(self, imgs):
            """Set new image train data."""
            if self._images is None:
                self._images = imgs
                self._ma_count = 1
            else:
                if imgs.shape != self._images.shape:
                    logger.error(f"The shape {imgs.shape} of the new image is "
                                 f"different from the current one "
                                 f"{self._images.shape}!")
                    return

                elif self._ma_window > 1:
                    if self._ma_count < self._ma_window:
                        self._ma_count += 1
                        self._images += (imgs - self._images) / self._ma_count
                    else:  # self._ma_count == self._ma_window
                        # here is an approximation
                        self._images += (imgs - self._images) / self._ma_window

                else:  # self._ma_window == 1
                    self._images = imgs
                    self._ma_count = 1

            self._invalid_image_cached()

        @property
        def moving_average_window(self):
            return self._ma_window

        @moving_average_window.setter
        def moving_average_window(self, v):
            if not isinstance(v, int) or v < 0:
                v = 1

            if v < self._ma_window:
                # if the new window size is smaller than the current one,
                # we reset the original image sum and count
                self._ma_window = v
                self._ma_count = 0
                self._images = None
                self._invalid_image_cached()

            self._ma_window = v

        @property
        def moving_average_count(self):
            return self._ma_count

    class ImageRefData:
        def __init__(self):
            self._image = None

        def __get__(self, instance, instance_type):
            if instance is None:
                return self
            return self._image

        def __set__(self, instance, value):
            self._image = value

    class ImageMaskData:
        def __init__(self, shape):
            """Initialization.

            :param shape: shape of the mask.
            """
            self._assembled = np.zeros(shape, dtype=bool)

        def get(self):
            """Return the assembled mask."""
            return self._assembled

        def set(self, mask):
            """Set the current mask."""
            self._assembled[:] = mask

        def add(self, x, y, w, h, flag):
            """Update an area in the mask.

            :param bool flag: True for masking the new area and False for
                unmasking.
            """
            self._assembled[y:y + h, x:x + w] = flag

        def clear(self):
            """Unmask all."""
            self._assembled[:] = False

    class ThresholdMaskData:
        def __init__(self, lb=None, ub=None):
            self._lower = lb
            self._upper = ub

        def get(self):
            lower = -np.inf if self._lower is None else self._lower
            upper = np.inf if self._upper is None else self._upper
            return lower, upper

        def set(self, lb, ub):
            self._lower = lb
            self._upper = ub

    class CropAreaData:
        def __init__(self):
            self._rect = None

        def get(self):
            return self._rect

        def set(self, x, y, w, h):
            self._rect = (x, y, w, h)

        def clear(self):
            self._rect = None

    __raw = RawImageData()
    __ref = ImageRefData()
    __threshold_mask = None
    __image_mask = None
    __crop_area = CropAreaData()

    pixel_size = None

    def __init__(self, images):
        """Initialization."""
        if self.pixel_size is None:
            self.__class__.pixel_size = config["PIXEL_SIZE"]

        if not isinstance(images, np.ndarray):
            raise TypeError(r"Images must be numpy.ndarray!")

        if images.ndim <= 1 or images.ndim > 3:
            raise ValueError(
                f"The shape of images must be (y, x) or (n_pulses, y, x)!")

        if self.__image_mask is None:
            self.__class__.__image_mask = self.ImageMaskData(images.shape[-2:])
        if self.__threshold_mask is None:
            self.__class__.__threshold_mask = self.ThresholdMaskData(
                *config['MASK_RANGE'])

        # update moving average
        self._set_images(images)

        # Instance attributes should be "frozen" after created. Otherwise,
        # the data used in different plot widgets could be different.
        self._images = self.__raw.images
        self._image_ref = self.__ref
        self._threshold_mask = self.__threshold_mask.get()
        self._image_mask = np.copy(self.__image_mask.get())
        self._crop_area = self.__crop_area.get()

        # cache these two properties
        self.ma_window
        self.ma_count

        self._registered_ops = set()

    @property
    def n_images(self):
        if self._images.ndim == 3:
            return self._images.shape[0]
        return 1

    @property
    def shape(self):
        return self._images.shape[-2:]

    @property
    def background(self):
        return self.__raw.background

    @property
    def threshold_mask(self):
        return self._threshold_mask

    @cached_property
    def ma_window(self):
        # Updating ma_window could set __raw._images to None. Since there
        # is no cache being deleted. '_images' in this instance will not
        # be set to None. Note: '_images' is not allowed to be None.
        return self.__raw.moving_average_window

    @cached_property
    def ma_count(self):
        # Updating ma_window could reset ma_count. Therefore, 'ma_count'
        # should both be a cached property
        return self.__raw.moving_average_count

    def pos(self, x, y):
        """current image -> original image."""
        if self._crop_area is None:
            return x, y
        x0, y0, _, _, = self._crop_area
        return x + x0, y + y0

    def pos_inv(self, x, y):
        """original image -> current image."""
        if self._crop_area is None:
            return x, y
        x0, y0, _, _, = self._crop_area
        return x - x0, y - y0

    def _set_images(self, imgs):
        self.__raw.set(imgs)

    def set_ma_window(self, v):
        self.__raw.moving_average_window = v

    def set_background(self, v):
        self.__raw.background = v
        self._registered_ops.add("background")

    def set_crop_area(self, flag, x, y, w, h):
        if flag:
            self.__crop_area.set(x, y, w, h)
        else:
            self.__crop_area.clear()

        self._registered_ops.add("crop")

    def set_image_mask(self, flag, x, y, w, h):
        if flag == ImageMaskChange.MASK:
            self.__image_mask.add(x, y, w, h, True)
        elif flag == ImageMaskChange.UNMASK:
            self.__image_mask.add(x, y, w, h, False)
        elif flag == ImageMaskChange.CLEAR:
            self.__image_mask.clear()
        elif flag == ImageMaskChange.REPLACE:
            self.__image_mask.set(x)

        self._registered_ops.add("image_mask")

    def set_threshold_mask(self, lb, ub):
        self.__threshold_mask.set(lb, ub)
        self._registered_ops.add("threshold_mask")

    def set_reference(self):
        if self._crop_area is None:
            self.__ref = self.masked_mean
        else:
            # recalculate the uncropped image
            self.__ref = self._masked_mean_imp(
                np.copy(self._mean_imp(self._images)))

        self._registered_ops.add("reference")

    @cached_property
    def image_mask(self):
        if self._crop_area is not None:
            x, y, w, h = self._crop_area
            return self._image_mask[y:y+h, x:x+w]

        return self._image_mask

    @cached_property
    def images(self):
        """Return the cropped, background-subtracted images.

        Warning: it shares the memory space with self._images
        """
        if self._crop_area is None:
            return self._images

        x, y, w, h = self._crop_area
        return self._images[..., y:y+h, x:x+w]

    @cached_property
    def ref(self):
        if self._image_ref is None:
            return None

        if self._crop_area is None:
            return self._image_ref

        x, y, w, h = self._crop_area
        return self._image_ref[..., y:y+h, x:x+w]

    @cached_property
    def mean(self):
        """Return the average of images over pulses in a train.

        The image is cropped and background-subtracted.

        :return numpy.ndarray: a single image, shape = (y, x)
        """
        return self._mean_imp(self.images)

    def _mean_imp(self, imgs):
        if imgs.ndim == 3:
            # pulse resolved
            return nanmean_axis0_para(imgs, max_workers=8, chunk_size=20)
        # train resolved
        return imgs

    @cached_property
    def masked_mean(self):
        """Return the masked average image.

        The image is cropped and background-subtracted before applying
        the mask.
        """
        # keep both mean image and masked mean image so that we can
        # recalculate the masked image
        return self._masked_mean_imp(np.copy(self.mean))

    def _masked_mean_imp(self, mean_image):
        # Convert 'nan' to '-inf' and it will later be converted to the
        # lower range of mask, which is usually 0.
        # We do not convert 'nan' to 0 because: if the lower range of
        # mask is a negative value, 0 will be converted to a value
        # between 0 and 255 later.
        mean_image[np.isnan(mean_image)] = -np.inf
        # clip the array, which now will contain only numerical values
        # within the mask range
        np.clip(mean_image, *self._threshold_mask, out=mean_image)

        return mean_image

    def update(self):
        invalid_caches = set()
        if "background" in self._registered_ops:
            self._images = self.__raw.images
            invalid_caches.update({"images", "mean", "masked_mean"})
        if "crop" in self._registered_ops:
            self._crop_area = self.__crop_area.get()
            invalid_caches.update(
                {"images", "ref", "mean", "masked_mean", "image_mask"})
        if "image_mask" in self._registered_ops:
            self._image_mask = np.copy(self.__image_mask.get())
            invalid_caches.add("image_mask")
        if "threshold_mask" in self._registered_ops:
            self._threshold_mask = self.__threshold_mask.get()
            invalid_caches.add("masked_mean")
        if "reference" in self._registered_ops:
            self._image_ref = self.__ref
            invalid_caches.add("ref")

        for cache in invalid_caches:
            try:
                del self.__dict__[cache]
            except KeyError:
                pass

        self._registered_ops.clear()

    @classmethod
    def reset(cls):
        """Reset all the class attributes.

        Used in unittest only.
        """
        cls.__raw = cls.RawImageData()
        cls.__ref = cls.ImageRefData()
        cls.__threshold_mask = None
        cls.__image_mask = None
        cls.__crop_area = cls.CropAreaData()


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
        roi (RoiData): stores ROI related data.
        on_off (LaserOnOffData): stores laser on-off related data.
        correlation (CorrelationData): correlation related data.
    """

    def __init__(self, tid, images=None, **kwargs):
        """Initialization."""
        self._tid = tid  # current Train ID

        if images is None:
            self._image_data = None
        else:
            self._image_data = ImageData(images, **kwargs)

        self.momentum = None
        self.intensities = None
        self.intensity_mean = None

        self.sample_degradation_foms = None

        self.roi = RoiData()
        self.on_off = LaserOnOffData()
        self.correlation = CorrelationData()

    @property
    def tid(self):
        return self._tid

    @property
    def image(self):
        return self._image_data

    @property
    def n_pulses(self):
        if self._image_data is None:
            return 0

        return self._image_data.n_images

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
    def add_correlator(idx, device_id, ppt, resolution=0.0):
        """Add a correlated parameter.

        :param int idx: index
        :param str device_id: device ID
        :param str ppt: property
        :param float resolution: resolution. Default = 0.0
        """
        if device_id and ppt:
            if resolution:
                CorrelationData.add_param(idx, device_id, ppt, resolution)
            else:
                CorrelationData.add_param(idx, device_id, ppt)
        else:
            CorrelationData.remove_param(idx)

    @staticmethod
    def get_correlators():
        return CorrelationData.get_params()

    @staticmethod
    def remove_correlators():
        CorrelationData.remove_params()

    @staticmethod
    def update_correlator_resolution(idx, resolution):
        CorrelationData.update_resolution(idx, resolution)

    def empty(self):
        """Check the goodness of the data."""
        logger.debug("Deprecated! use self.n_pulses!")
        if self.image is None:
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
