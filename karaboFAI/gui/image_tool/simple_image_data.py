"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from ...pipeline.data_model import ImageData
from ...utils import cached_property
from ...algorithms import mask_image


class _SimpleImageData:
    """SimpleImageData which is used by ImageToolWindow.

    In ImageToolWindow, some properties of the image can be changed, for
    instance, background, threshold mask, etc.

    Attributes:
        pixel_size (float): pixel size of the detector.
        threshold_mask (tuple): (lower, upper) boundaries of the
            threshold mask.
        background (float): a uniform background value.
        masked (numpy.ndarray): image with threshold mask.
    """

    def __init__(self, image_data):
        """Initialization.

        Construct a _SimpleImageData instance from an ImageData instance.

        :param ImageData image_data: an ImageData instance.
        """
        if not isinstance(image_data, ImageData):
            raise TypeError("Input must be an ImageData instance.")

        self._pixel_size = image_data.pixel_size

        # This is only used for reset the image in the ImageTool, which
        # does not occur very often. Therefore, the copy is used to avoid
        # data sharing.
        # Note: image_data.mean does not contain any NaN
        self._image = image_data.mean.copy()

        # Note:: we do not copy 'masked_mean' since it also includes image_mask

        # image mask is plotted on top of the image in ImageTool

        self._bkg = image_data.background
        self._threshold_mask = image_data.threshold_mask

    @property
    def pixel_size(self):
        return self._pixel_size

    @property
    def background(self):
        return self._bkg

    @background.setter
    def background(self, v):
        if v == self._bkg:
            return
        self._image -= v - self._bkg  # in-place operation
        self._bkg = v

        # invalidate cache
        try:
            del self.__dict__['masked']
        except KeyError:
            pass

    @property
    def threshold_mask(self):
        return self._threshold_mask

    @threshold_mask.setter
    def threshold_mask(self, mask):
        if mask == self._threshold_mask:
            return

        self._threshold_mask = mask

        # invalid cache
        del self.__dict__['masked']

    @cached_property
    def masked(self):
        return mask_image(self._image,
                          threshold_mask=self._threshold_mask,
                          inplace=False)

    @classmethod
    def from_array(cls, arr):
        """Instantiate from an array."""
        return cls(ImageData.from_array(arr))
