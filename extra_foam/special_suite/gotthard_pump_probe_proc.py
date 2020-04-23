"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import numpy as np

from extra_foam.pipeline.data_model import MovingAverageArray
from extra_foam.utils import profiler

from .special_analysis_base import ProcessingError, QThreadWorker
from .config import _PIXEL_DTYPE


class GotthardPumpProbeProcessor(QThreadWorker):
    """Gotthard pump-probe analysis processor.

    Attributes:
        _output_channel (str): output channel name.
        _poi_index (int): index of the pulse of interest.
        _on_slicer (slice): a slicer used to slice on-pulses in a train.
        _off_slicer (slice): a slicer used to slice off-pulses in a train.
        _dark_slicer (slice): a slicer used to slice dark pulses in a train.
        _raw_ma (numpy.ndarray): moving average of the raw data.
            Shape=(pulses, pixels)
        _vfom_ma (numpy.ndarray): moving average of the vector figure-of-merit
            data. Shape=(pulses, pixels)
    """

    _raw_ma = MovingAverageArray()
    _vfom_ma = MovingAverageArray()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._output_channel = ""
        self._ppt = "data.adc"

        self._poi_index = 0

        self._on_slicer = slice(None, None)
        self._off_slicer = slice(None, None)
        self._dark_slicer = slice(None, None)

        del self._raw_ma
        del self._vfom_ma

    def onOutputChannelChanged(self, ch: str):
        self._output_channel = ch

    def onMaWindowChanged(self, value: str):
        self.__class__._raw_ma.window = int(value)
        self.__class__._vfom_ma.window = int(value)

    def onOnSlicerChanged(self, value: list):
        self._on_slicer = slice(*value)

    def onOffSlicerChanged(self, value: list):
        self._off_slicer = slice(*value)

    def onPoiIndexChanged(self, value: int):
        self._poi_index = value

    def onDarkSlicerChanged(self, value: list):
        self._dark_slicer = slice(*value)

    def sources(self):
        """Override."""
        return [
            (self._output_channel, self._ppt),
        ]

    @profiler("Gotthard Processor (pump-probe)")
    def process(self, data):
        """Override."""
        data, meta = data["raw"], data["meta"]
        tid = self.getTrainId(meta)

        if not self._output_channel or not self._ppt:
            return
        raw = self.getPropertyData(data, self._output_channel, self._ppt)

        # check data shape
        if raw.ndim != 2:
            raise ProcessingError(f"Gotthard data must be a 2D array: "
                                  f"actual {raw.ndim}D")

        raw = raw.astype(_PIXEL_DTYPE)

        # ------------
        # process data
        # ------------

        # Note: we do not check whether on/off/dark share a same pulse index

        # update the moving average of raw data
        self._raw_ma = raw

        # calculate the dark for the current train
        dark_mean = np.mean(raw[self._dark_slicer], axis=0)

        # calculate figure-of-merit for the current train
        on = raw[self._on_slicer] - dark_mean
        off = raw[self._off_slicer] - dark_mean

        if len(on) != len(off):
            raise ProcessingError(f"Number of on and off pulses are different: "
                                  f"{len(on)} and {len(off)}")

        # check POI index
        if self._poi_index >= len(on):
            raise ProcessingError(f"POI index {self._poi_index} out of "
                                  f"boundary [{0} - {len(on) - 1}]")

        # TODO: switch among several VFOM definitions
        vfom = on - off
        vfom_mean = np.mean(vfom, axis=0)

        self._vfom_ma = vfom
        vfom_ma = self._vfom_ma
        vfom_ma_mean = np.mean(vfom_ma, axis=0)

        self.log.info(f"Train {tid} processed")

        return {
            # raw spectrum
            "spectrum": raw,
            # index of pulse of interest
            "poi_index": self._poi_index,
            # VFOM for the current train
            "vfom": vfom,
            # Moving averaged of vfom
            "vfom_ma": vfom_ma,
            # averaged VFOM over pulses for the current train
            "vfom_mean": vfom_mean,
            # moving average of vfom_mean
            "vfom_ma_mean": vfom_ma_mean,
        }

    def reset(self):
        """Override."""
        del self._raw_ma
        del self._vfom_ma
