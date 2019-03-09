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

import numpy as np

from cached_property import cached_property

from ..algorithms import nanmean_axis0_para
from ..logger import logger
from ..config import ImageMaskChange


class TrainData:
    """Store the history train data.

    Each data point is pair of data: (x, value), where x can be a
    train ID for time series analysis or a correlated data for
    correlation analysis.
    """
    MAX_LENGTH = 1000000

    def __init__(self, **kwargs):
        # We need to have a 'x' for each sub-dataset due to the
        # concurrency of data processing.
        self._x = []
        self._values = []
        # for now it is used in CorrelationData to store device ID and
        # property information
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
        # do not clear _info here!


class AbstractData(abc.ABC):
    @classmethod
    def clear(cls):
        for attr in cls.__dict__.values():
            if isinstance(attr, TrainData):
                # descriptor protocol will not be triggered here
                attr.clear()


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


class ImageMaskData:
    def __init__(self, shape):
        """Initialization.

        :param shape: shape of the mask.
        """
        self._assembled = np.zeros(shape, dtype=bool)

    @property
    def mask(self):
        """Return the assembled mask.

        The owner of the ImageMaskData takes the responsibility of
        invalidating the cache when appropriate.
        """
        return self._assembled

    def replace(self, mask):
        """Replace the current mask."""
        self._assembled[:] = mask

    def update(self, rect, masking):
        """Update an area in the mask.

        :param tuple rect: (x, y, w, h) of the rectangle.
        """
        x, y, w, h = rect
        self._assembled[y:y+h, x:x+w] = masking

    def clear(self):
        """Remove all the masking areas."""
        self._assembled[:] = False


class ImageData:
    """A class that manages the detector images.

    Operation flow:

    remove background -> cropping -> calculate mean image -> apply mask

    Attributes:
        _images (numpy.ndarray): detector images for all the pulses in
            a train. shape = (pulse_id, y, x) for pulse-resolved
            detectors and shape = (y, x) for train-resolved detectors.
        _bkg (float): background level of the detector image.
        _ma_window (int): moving average window size
        _ma_count (int): current moving average count

        _threshold_mask (tuple): (min, max) threshold of the pixel value.
        _image_mask (numpy.ndarray): an image mask, default = None.
            Shape = (y, x)
        _crop_area (tuple): (w, h, x, y) of the cropped image.
        pixel_size (float): detector pixel size.
        _poni (tuple): (Cx, Cy), where Cx is the coordinate of the point
            of normal incidence along the detector's second dimension,
            in pixels, and Cy is the coordinate of the point of normal
            incidence along the detector's first dimension, in pixels.
            default = (0, 0)
    """
    _images = None
    _bkg = 0
    _ma_window = 1
    _ma_count = 0

    __image_mask_data = None

    def __init__(self, images, *,
                 threshold_mask=None,
                 background=0.0,
                 crop_area=None,
                 pixel_size=None,
                 poni=None):
        """Initialization."""
        if not isinstance(images, np.ndarray):
            raise TypeError(r"Images must be numpy.ndarray!")

        if images.ndim <= 1 or images.ndim > 3:
            raise ValueError(
                f"The shape of images must be (y, x) or (n_pulses, y, x)!")

        self._compute_moving_average(images, background)

        if self.__image_mask_data is None:
            self.__class__.__image_mask_data = ImageMaskData(self.shape)

        # An ImageData instance should have a mask which will not change
        # during the life time of the instance.
        self._image_mask = np.copy(self.__image_mask_data.mask)

        self._crop_area = crop_area

        # the mask information is stored in the data so that all the
        # processing and visualization can use the same mask
        self._threshold_mask = threshold_mask

        self.pixel_size = pixel_size

        self._poni = (0, 0) if poni is None else poni

    @classmethod
    def _compute_moving_average(cls, imgs, bkg):
        """Compute the new moving average of the original images."""
        # automatically reset if source changes, e.g. number of pulses
        # per train changes for pulse-resolved data
        if cls._images is not None and imgs.shape != cls._images.shape:
            cls._ma_count = 0

        if cls._ma_count == 0:
            cls._images = None

        if cls._ma_window > 1 and cls._images is not None:
            if cls._ma_count < cls._ma_window:
                cls._ma_count += 1
                cls._images += (imgs - cls._bkg - cls._images) / cls._ma_count
            elif cls._ma_count == cls._ma_window:
                # here is an approximation
                cls._images += (imgs - cls._bkg - cls._images) / cls._ma_window
            else:
                # should never reach here
                raise ValueError

            cls._images -= bkg - cls._bkg
        else:
            # For the single image case, we do not care about the old
            # background
            cls._images = imgs - bkg
            cls._ma_count = 1

        cls._bkg = bkg

    @property
    def n_images(self):
        if self._images.ndim == 3:
            return self._images.shape[0]
        return 1

    @property
    def shape(self):
        return self._images.shape[-2:]

    def pos(self, x, y):
        """Return the position in the original image."""
        if self._crop_area is None:
            return x, y
        x0, y0, _, _, = self._crop_area
        return x + x0, y + y0

    def update_image_mask(self, tp, x, y, w, h):
        if tp == ImageMaskChange.MASK:
            self.__image_mask_data.update((x, y, w, h), True)
        elif tp == ImageMaskChange.UNMASK:
            self.__image_mask_data.update((x, y, w, h), False)
        elif tp == ImageMaskChange.CLEAR:
            self.__image_mask_data.clear()

    def replace_image_mask(self, mask):
        self.__image_mask_data.replace(mask)

    @property
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
    def mean(self):
        """Return the average of images over pulses in a train.

        The image is cropped and background-subtracted.

        :return numpy.ndarray: a single image, shape = (y, x)
        """
        if self._images.ndim == 3:
            # pulse resolved
            return nanmean_axis0_para(self.images,
                                      max_workers=8, chunk_size=20)
        # train resolved
        return self.images

    @cached_property
    def masked_mean(self):
        """Return the masked average image.

        The image is cropped and background-subtracted before applying
        the mask.
        """
        # keep both mean image and masked mean image so that we can
        # recalculate the masked image
        mean_image = np.copy(self.mean)

        # Convert 'nan' to '-inf' and it will later be converted to the
        # lower range of mask, which is usually 0.
        # We do not convert 'nan' to 0 because: if the lower range of
        # mask is a negative value, 0 will be converted to a value
        # between 0 and 255 later.
        mean_image[np.isnan(mean_image)] = -np.inf
        # clip the array, which now will contain only numerical values
        # within the mask range
        if self._threshold_mask is not None:
            np.clip(mean_image, *self._threshold_mask, out=mean_image)
        return mean_image

    @property
    def threshold_mask(self):
        return self._threshold_mask

    @threshold_mask.setter
    def threshold_mask(self, v):
        if self._threshold_mask == v:
            return
        self._threshold_mask = v
        try:
            del self.__dict__['masked_mean']
        except KeyError:
            pass

    @property
    def crop_area(self):
        return self._crop_area

    @crop_area.setter
    def crop_area(self, v):
        if self._crop_area == v:
            return

        self._crop_area = v
        self._reset_all_caches()

    @property
    def background(self):
        return self._bkg

    @background.setter
    def background(self, v):
        if self._bkg == v:
            return

        self.__class__._images -= v - self._bkg
        self.__class__._bkg = v
        self._reset_all_caches()

    @property
    def poni(self):
        """Return the PONI in the original image."""
        poni1 = self._poni[0]
        poni2 = self._poni[1]
        if self._crop_area is not None:
            x, y, _, _ = self._crop_area
            poni1 -= y
            poni2 -= x

        return poni1, poni2

    @poni.setter
    def poni(self, v):
        self._poni = v

    @classmethod
    def set_moving_average_window(cls, v):
        if not isinstance(v, int) or v < 0:
            v = 1

        if v < cls._ma_window:
            # if the new window size is smaller than the current one,
            # we reset the original image sum and count
            cls._ma_count = 0  # a flag which indicates reset

        cls._ma_window = v

    @property
    def moving_average_count(self):
        return self.__class__._ma_count

    def _reset_all_caches(self):
        for key in ('masked_mean', 'mean', 'images'):
            try:
                del self.__dict__[key]
            except KeyError:
                pass

    @classmethod
    def reset(cls):
        """Reset all the class attributes.

        Used in unittest only.
        """
        cls._images = None  # moving average of original images
        cls._bkg = 0  # background
        cls._ma_window = 1
        cls._ma_count = 0
        cls.__image_mask_data = None


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
        if not isinstance(tid, int):
            raise ValueError("Train ID must be an integer!")

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
    def set_moving_average_window(cls, v):
        ImageData.set_moving_average_window(v)

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
