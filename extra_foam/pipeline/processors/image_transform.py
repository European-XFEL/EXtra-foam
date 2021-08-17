"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""

from collections import defaultdict

import numpy as np
from scipy.ndimage import center_of_mass

from .base_processor import _BaseProcessor
from ..data_model import MovingAverageArray
from ...database import Metadata as mt
from ...utils import profiler
from ...config import ImageTransformType

from extra_foam.algorithms import (
    edge_detect, fourier_transform_2d, mask_image_data, SimplePairSequence
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

class _BraggPeakAnalysis:
    __slots__ = ["rois", "window"]

    def __init__(self):
        self.rois = { }
        self.window = 1

class ImageTransformProcessor(_BaseProcessor):

    def __init__(self):
        super().__init__()

        self._transform_type = ImageTransformType.UNDEFINED

        self._fft = _FourierTransform()
        self._ed = _EdgeDetection()
        self._bp = _BraggPeakAnalysis()

        self._pulse_peaks = defaultdict(lambda: MovingAverageArray(window=self._bp.window))
        self._center_of_mass_history = defaultdict(lambda: SimplePairSequence(max_len=self._bp.window))

    def update(self):
        """Override."""
        cfg = self._meta.hget_all(mt.IMAGE_TRANSFORM_PROC)

        transform_type = ImageTransformType(int(cfg["transform_type"]))
        if self._transform_type != transform_type:
            self._transform_type = transform_type

        if transform_type == ImageTransformType.FOURIER_TRANSFORM:
            fft = self._fft
            fft.logrithmic = cfg["fft:logrithmic"] == 'True'
        elif transform_type == ImageTransformType.EDGE_DETECTION:
            ed = self._ed
            ed.kernel_size = int(cfg["ed:kernel_size"])
            ed.sigma = float(cfg["ed:sigma"])
            ed.threshold = self.str2tuple(cfg["ed:threshold"])
        elif transform_type == ImageTransformType.BRAGG_PEAK_ANALYSIS:
            # Update config
            rois = { }
            for key, value in cfg.items():
                if key.startswith("bp:roi_"):
                    rois[key] = self.str2tuple(value, handler=int)
            self._bp.rois = rois

            window = int(cfg["bp:window_size"])
            if window != self._bp.window:
                self._bp.window = window

                for peak_array in self._pulse_peaks.values():
                    peak_array.window = self._bp.window
                for label in self._center_of_mass_history.keys():
                    self._center_of_mass_history[label] = SimplePairSequence(max_len=self._bp.window)

    @profiler("Image transform processor")
    def process(self, data):
        processed = data['processed']
        image = processed.image

        transform_type = self._transform_type

        masked_mean = image.masked_mean
        image.transform_type = transform_type

        if transform_type == ImageTransformType.FOURIER_TRANSFORM:
            fft = self._fft
            image.transformed = fourier_transform_2d(
                masked_mean, logrithmic=fft.logrithmic)
        elif transform_type == ImageTransformType.EDGE_DETECTION:
            ed = self._ed
            image.transformed = edge_detect(
                masked_mean,
                kernel_size=ed.kernel_size,
                sigma=ed.sigma,
                threshold=ed.threshold)

        if len(self._bp.rois) > 0:
            bragg_peak_data = processed.pulse.bragg_peaks
            pulses = data["assembled"]["data"]

            # If the detector is pulse-resolved, process only the selected
            # pulses.
            if pulses.ndim == 3:
                pulses = pulses[image.sliced_indices]

            # Add an extra, empty dimension to train-resolved detectors to make
            # the rest of the code work.
            elif pulses.ndim == 2:
                pulses = np.expand_dims(pulses, axis=0)


            for label, dims in self._bp.rois.items():
                data_label = label.split("_")[1]
                x, y, width, height = dims
                max_x = x + width
                max_y = y + height

                # Ensure that x and y never go negative
                x = max(0, x)
                y = max(0, y)

                pulses_roi = pulses[:, y:max_y, x:max_x].copy()
                mask_roi = image.image_mask[y:max_y, x:max_x]

                # Apply mask to ROIs. This is done over the ROIs instead of the
                # entire images for performance.
                for pulse in range(pulses.shape[0]):
                    mask_image_data(pulses_roi[pulse],
                                    image_mask=mask_roi,
                                    threshold_mask=image.threshold_mask)

                pulses_mean = masked_mean[y:max_y, x:max_x]

                bragg_peak_data.roi[data_label] = pulses_mean
                bragg_peak_data.roi_dims[data_label] = dims
                bragg_peak_data.roi_intensity[data_label] = np.nansum(pulses_mean)
                bragg_peak_data.pulses[data_label] = pulses_roi
                bragg_peak_data.lineout_x[data_label] = np.nanmean(pulses_roi, axis=-1)
                bragg_peak_data.lineout_y[data_label] = np.nanmean(pulses_roi, axis=-2)

                # Scipy's center_of_mass() doesn't handle NaNs or negatives very
                # well, so we need to work around that. First we shift the data
                # so everything is positive.
                shifted_pulses_mean = pulses_roi + np.abs(np.nanmin(pulses_roi))
                # And then we create a mask to only include positive values
                # (ignoring NaNs). This is actually passed as center_of_mass()'s
                # 'label' argument, but we can't use ndimage.label() because
                # that may find multiple features.
                coms = []
                for pulse in range(pulses_roi.shape[0]):
                    mask = (shifted_pulses_mean[pulse] > 0).astype(int)
                    coms.append(center_of_mass(shifted_pulses_mean[pulse], labels=mask))

                # Update the center of mass and its standard deviation
                # self._center_of_mass_history[data_label].append(com)
                bragg_peak_data.center_of_mass[data_label] = coms
                # com_y_hist, com_x_hist = self._center_of_mass_history[data_label].data()
                # bragg_peak_data.center_of_mass_stddev[data_label] = (np.std(com_y_hist), np.std(com_x_hist))

                # Update the pulse intensity
                image_dims = (pulses_roi.ndim - 2, pulses_roi.ndim - 1)
                bragg_peak_data.pulse_intensity[data_label] = np.nansum(pulses_roi, axis=image_dims)
