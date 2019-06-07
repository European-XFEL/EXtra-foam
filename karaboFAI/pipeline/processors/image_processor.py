"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

PumpProbeProcessors.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import numpy as np

from .base_processor import LeafProcessor, CompositeProcessor, SharedProperty
from ..data_model import ProcessedData
from ..exceptions import ProcessingError
from ...metadata import Metadata as mt
from ...utils import profiler


class _RawImageData:
    """Stores moving average of raw images.

    Be careful: the internal image share the memory with the first
    data!!!
    """
    def __init__(self, images=None):
        self._images = None  # moving average (original data)
        self._ma_window = 1
        self._ma_count = 0

        if images is not None:
            self.images = images

    @property
    def n_images(self):
        if self._images is None:
            return 0

        if self._images.ndim == 3:
            return self._images.shape[0]
        return 1

    @property
    def images(self):
        return self._images

    @images.setter
    def images(self, data):
        """Set new image data."""
        if not isinstance(data, np.ndarray):
            raise TypeError(r"Image data must be numpy.ndarray!")

        if data.ndim <= 1 or data.ndim > 3:
            raise ValueError(
                f"The shape of images must be (y, x) or (n_pulses, y, x)!")

        if self._images is not None and self._ma_window > 1:
            if data.shape != self._images.shape:
                # Note: this can happen, for example, when the quadrant
                #       positions of the LPD detectors changes.
                self._images = data
                self._ma_count = 1
                return

            if self._ma_count < self._ma_window:
                self._ma_count += 1
                self._images += (data - self._images) / self._ma_count
            else:  # self._ma_count == self._ma_window
                # here is an approximation
                self._images += (data - self._images) / self._ma_window

        else:  # self._images is None or self._ma_window == 1
            self._images = data
            self._ma_count = 1

    @property
    def ma_window(self):
        return self._ma_window

    @ma_window.setter
    def ma_window(self, v):
        if not isinstance(v, int) or v <= 0:
            v = 1

        if v < self._ma_window:
            # if the new window size is smaller than the current one,
            # we reset the original image sum and count
            self._ma_window = v
            self._ma_count = 0
            self._images = None

        self._ma_window = v

    @property
    def ma_count(self):
        return self._ma_count

    def clear(self):
        self._images = None
        self._ma_window = 1
        self._ma_count = 0


class ImageProcessor(CompositeProcessor):
    """ImageProcessor.

    Attributes:
        background (float): a uniform background value.
        threshold_mask (tuple): threshold mask.
    """
    background = SharedProperty()
    threshold_mask = SharedProperty()

    def __init__(self):
        super().__init__()

        self._data = _RawImageData()

    def update(self):
        """Override."""
        cfg = self._meta.get_all(mt.IMAGE_PROC)
        if cfg is None:
            return

        self.background = float(cfg['background'])
        self.threshold_mask = self.str2tuple(cfg['threshold_mask'],
                                             handler=float)
        self._data.ma_window = int(cfg['ma_window'])

    @profiler("Image Processor")
    def process(self, data):
        assembled = data['assembled']
        del data['assembled']  # remove the temporary item

        self._data.images = assembled

        data['processed'] = ProcessedData(data['tid'], self._data.images,
                                          background=self.background,
                                          threshold_mask=self.threshold_mask,
                                          ma_window=self._data.ma_window,
                                          ma_count=self._data.ma_count)
