"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from enum import Enum
from dataclasses import dataclass, field
from collections import defaultdict

import numpy as np

from PyQt5.QtCore import pyqtSignal

from ..geometries import JungFrauGeometryFast
from ..algorithms import movingAvgImageData

from .special_analysis_base import profiler, QThreadWorker


defaultdict_ndarray = lambda: field(default_factory=lambda: defaultdict(lambda: np.array([])))
defaultdict_none = lambda: defaultdict(lambda: None)
@dataclass
class XesData:
    pumped: defaultdict = defaultdict_ndarray()
    pumped_train_avg: np.ndarray = None
    unpumped: defaultdict = defaultdict_ndarray()
    unpumped_train_avg: np.ndarray = None
    difference: defaultdict = field(default_factory=defaultdict_none)
    train_count: int = 1


class DisplayOption(Enum):
    PUMPED = "Pumped trains (avg)"
    UNPUMPED = "Unpumped trains (avg)"
    DIFFERENCE = "Difference (avg(pumped) - avg(unpumped))"


class XesTimingProcessor(QThreadWorker):
    """XES timing processor.

    Attributes:
        _output_channel (str): output channel name.
        _ppt (str): property name.
    """
    new_train_data_sgn = pyqtSignal(dict)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._even_trains_pumped = True

        self._refresh_plots = False
        self._detector = None
        self._delay_device = None
        self._target_delay_device = None

        self.reset()

    def reset(self):
        self._xes_curves = defaultdict(XesData)

    @property
    def xes_delay_data(self):
        return self._xes_curves

    def onEvenTrainsPumpedChanged(self, value: int):
        self._even_trains_pumped = bool(value)

        # If this setting has changed, then we need to swap all the
        # pumped/unpumped data.
        for data in self._xes_curves.values():
            data.pumped, data.unpumped = data.unpumped, data.pumped
            data.pumped_train_avg, data.unpumped_train_avg = data.unpumped_train_avg, data.pumped_train_avg
            for diff in data.difference.values():
                if diff is not None:
                    diff *= -1

        self._refresh_plots = True

    def setDetectorDevice(self, device, prop):
        self._detector = (device, prop)

    def setDelayDevice(self, device, prop):
        self._delay_device = (device, prop)

    def setTargetDelayDevice(self, device, prop):
        self._target_delay_device = (device, prop)

    def onImgDisplayChanged(self, value: str):
        self._display_option = DisplayOption(value)

    def onRoiGeometryChange(self, roi_params):
        idx, activated, _, x, y, w, h = roi_params

        # If a ROI has been moved/resized we need to update all of the existing
        # XES data. The data for the current delay will be updated in process(),
        # but the data for the old delays need to be updated here.
        if activated and self._rois_geom_st[idx] is not None:
            for delay_data in self._xes_curves.values():
                self.updateXES(delay_data, True, idx, (x, y, w, h))
                self.updateXES(delay_data, False, idx, (x, y, w, h))

        super().onRoiGeometryChange(roi_params)
        self._refresh_plots = True

    @profiler("XES timing processor")
    def process(self, data):
        """Override."""
        data, meta = data["raw"], data["meta"]
        tid = self.getTrainId(meta)
        self.new_train_data_sgn.emit(data)

        if any(attr is None for attr in [self._detector, self._delay_device, self._target_delay_device]):
            return None

        # Get the actual and target delays, and round them to 3 decimal places
        delay = self.getPropertyData(data, *self._delay_device)
        delay = np.around(delay, 3)
        target_delay = self.getPropertyData(data, *self._target_delay_device)
        target_delay = np.around(target_delay, 3)

        # We ignore trains where the delay motor hasn't reached its target
        if delay != target_delay:
            return None

        img = self.squeezeToImage(tid, self.getPropertyData(data, *self._detector))

        # Set all negative values to 0 and mask ASIC edges
        img[img < 0] = 0
        JungFrauGeometryFast.mask_module_py(img)

        delay_data = self._xes_curves[target_delay]
        is_pumped = (tid % 2 == 0 and self._even_trains_pumped) or \
                    (tid % 2 != 0 and not self._even_trains_pumped)
        img_avg = delay_data.pumped_train_avg if is_pumped else delay_data.unpumped_train_avg

        # Add to running average
        if img_avg is None:
            if is_pumped:
                delay_data.pumped_train_avg = img
            else:
                delay_data.unpumped_train_avg = img
        else:
            delay_data.train_count += 1
            movingAvgImageData(img_avg, img, delay_data.train_count)

        if img_avg is not None:
            activated_rois = [(idx, geom) for idx, geom in self._rois_geom_st.items()
                              if geom is not None]
            for roi_idx, roi_geom in activated_rois:
                self.updateXES(delay_data, is_pumped, roi_idx, roi_geom)

        # Select some data to return for display
        match self._display_option:
            case DisplayOption.PUMPED:
                img_avg = delay_data.pumped_train_avg
            case DisplayOption.UNPUMPED:
                img_avg = delay_data.unpumped_train_avg
            case DisplayOption.DIFFERENCE:
                # Only after the first pumped and unpumped trains will we have
                # the data to display the difference between them.
                if delay_data.pumped_train_avg is None or delay_data.unpumped_train_avg is None:
                    return None
                else:
                    img_avg = delay_data.pumped_train_avg - delay_data.unpumped_train_avg
            case x:
                raise RuntimeError(f"Unsupported display type: {x}")

        # Check if the plots need to refresh their data for this train
        refresh_plots = self._refresh_plots
        if refresh_plots:
            self._refresh_plots = False

        self.log.info(f"Train {tid} processed")
        return {
            "img_avg": img_avg,
            "delay": target_delay,
            "xes": self._xes_curves,
            "refresh_plots": refresh_plots
        }

    def updateXES(self, delay_data, update_pumped, roi_idx, roi_geom):
        img_avg = delay_data.pumped_train_avg if update_pumped else delay_data.unpumped_train_avg

        # Extract the ROI
        x, y, w, h = roi_geom
        roi = img_avg[y:y + h, x:x + w]

        # Compute normalized XES curve
        xes = np.nansum(roi, axis=0) / np.nansum(roi)

        # Update curves for this delay and ROI
        pumped = delay_data.pumped
        unpumped = delay_data.unpumped
        if update_pumped:
            pumped[roi_idx] = xes
        else:
            unpumped[roi_idx] = xes

        # Check that both the current pumped/unpumped array are the same
        # length. These are only updated every train, so when the width
        # of a ROI (and thus length of the [un]pumped arrays) is changed
        # it will take two trains (a pumped and unpumped one) before
        # these fields are updated.
        if len(pumped[roi_idx]) == len(unpumped[roi_idx]):
            delay_data.difference[roi_idx] = pumped[roi_idx] - unpumped[roi_idx]
