"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from .base_processor import _BaseProcessor
from ..data_model import MovingAverageArray
from ...database import Metadata as mt
from ...utils import profiler
from ...config import ImageTransformType

from extra_foam.algorithms import (
    edge_detect, fourier_transform_2d
)


class _FourierTransform:
    __slots__ = ['logrithmic']

    def __init__(self):
        self.logrithmic = True


class _EdgeDetection:
    __slots__ = ['kernel_size', 'sigma', 'threshold']

    def __init__(self):
        self.kernel_size = None
        self.sigma = None
        self.threshold = None


class ImageTransformProcessor(_BaseProcessor):

    _masked_ma = MovingAverageArray()

    def __init__(self):
        super().__init__()

        self._ma_window = 1
        del self._masked_ma
        self.__class__._masked_ma.window = 1

        self._transform_type = ImageTransformType.UNDEFINED

        self._fft = _FourierTransform()
        self._ed = _EdgeDetection()

        self._reset = False

    def update(self):
        """Override."""
        cfg = self._meta.hget_all(mt.IMAGE_TRANSFORM_PROC)

        v = int(cfg['ma_window'])
        if self._ma_window != v:
            self._ma_window = v
            self.__class__._masked_ma.window = v

        transform_type = ImageTransformType(int(cfg["transform_type"]))
        if self._transform_type != transform_type:
            self._transform_type = transform_type
            self._reset = True

        if transform_type == ImageTransformType.FOURIER_TRANSFORM:
            fft = self._fft
            fft.logrithmic = cfg["fft:logrithmic"] == 'True'
        elif transform_type == ImageTransformType.EDGE_DETECTION:
            ed = self._ed
            ed.kernel_size = int(cfg["ed:kernel_size"])
            ed.sigma = float(cfg["ed:sigma"])
            ed.threshold = self.str2tuple(cfg["ed:threshold"])

    @profiler("Image transform processor")
    def process(self, data):
        transform_type = self._transform_type
        if transform_type == ImageTransformType.UNDEFINED:
            return

        processed = data['processed']
        transformed = processed.image.transformed

        self._masked_ma = processed.image.masked_mean
        masked_ma = self._masked_ma

        transformed.original = masked_ma
        if transform_type == ImageTransformType.CONCENTRIC_RINGS:
            transformed.transformed = masked_ma
        elif transform_type == ImageTransformType.FOURIER_TRANSFORM:
            fft = self._fft
            transformed.transformed = fourier_transform_2d(
                masked_ma, logrithmic=fft.logrithmic)
        elif transform_type == ImageTransformType.EDGE_DETECTION:
            ed = self._ed
            transformed.transformed = edge_detect(
                masked_ma,
                kernel_size=ed.kernel_size,
                sigma=ed.sigma,
                threshold=ed.threshold)
