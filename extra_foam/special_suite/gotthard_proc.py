"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import numpy as np

from extra_foam.algorithms import hist_with_stats
from extra_foam.pipeline.data_model import MovingAverageArray

from .config import _MAX_INT32, _PIXEL_DTYPE
from .special_analysis_base import ProcessingError, profiler, QThreadWorker

_DEFAULT_N_BINS = 10
_DEFAULT_BIN_RANGE = "-inf, inf"


class GotthardProcessor(QThreadWorker):
    """Gotthard analysis processor.

    Attributes:
        _output_channel (str): output channel name.
        _pulse_slicer (slice): a slicer used to slice pulses in a train.
        _poi_index (int): index of the pulse of interest after slicing.
        _scale (float): scale of the x axis. If 0, it means no scale will
            be applied and the unit of x-axis is pixel. While a positive
            value means converting pixel to eV by multiplying this value
            for the x axis.
        _offset (float): offset of the x axis when the value of scale is
            not zero.
        _bin_range (tuple): range of the ADU histogram.
        _n_bins (int): number of bins of the ADU histogram.
        _hist_over_ma (bool): True for calculating the histogram over the
            moving averaged data. Otherwise, it is calculated over the
            current train.
        _raw_ma (numpy.ndarray): moving average of the raw data.
            Shape=(pulses, pixels)
        _dark_ma (numpy.ndarray): moving average of the dark data.
            Shape=(pulses, pixels)
        _dark_mean_ma (numpy.ndarray): average of pulses in a train of the
            moving average of the dark data. It is used for dark subtraction.
            Shape=(pixels,)
    """

    _raw_ma = MovingAverageArray()
    _dark_ma = MovingAverageArray(_MAX_INT32)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._output_channel = ""
        self._ppt = "data.adc"

        self._pulse_slicer = slice(None, None)
        self._poi_index = 0

        self._scale = 0
        self._offset = 0

        self._bin_range = self.str2range(_DEFAULT_BIN_RANGE)
        self._n_bins = _DEFAULT_N_BINS
        self._hist_over_ma = False

        del self._raw_ma

        del self._dark_ma
        self._dark_mean_ma = None

    def onOutputChannelChanged(self, ch: str):
        self._output_channel = ch

    def onMaWindowChanged(self, value: str):
        self.__class__._raw_ma.window = int(value)

    def onScaleChanged(self, value: str):
        self._scale = float(value)

    def onOffsetChanged(self, value: str):
        self._offset = float(value)

    def onBinRangeChanged(self, value: tuple):
        self._bin_range = value

    def onNoBinsChanged(self, value: str):
        self._n_bins = int(value)

    def onHistOverMaChanged(self, state: bool):
        self._hist_over_ma = state

    def onPulseSlicerChanged(self, value: list):
        self._pulse_slicer = slice(*value)
        dark_ma = self._dark_ma
        if dark_ma is not None:
            self._dark_mean_ma = np.mean(dark_ma[self._pulse_slicer], axis=0)

    def onPoiIndexChanged(self, value: int):
        self._poi_index = value

    def onLoadDarkRun(self, dirpath):
        """Override."""
        run = self._loadRunDirectoryST(dirpath)
        if run is not None:
            try:
                arr = run.get_array(self._output_channel, self._ppt)
                shape = arr.shape
                if arr.ndim != 3:
                    self.log.error(f"Data must be a 3D array! "
                                   f"Actual shape: {shape}")
                    return

                self.log.info(f"Found dark data with shape {shape}")
                # FIXME: performance
                self._dark_ma = np.mean(
                    arr.values, axis=0, dtype=_PIXEL_DTYPE)
                self._dark_mean_ma = np.mean(
                    self._dark_ma[self._pulse_slicer],
                    axis=0, dtype=_PIXEL_DTYPE)
            except Exception as e:
                self.log.error(f"Unexpect exception when getting data array: "
                               f"{repr(e)}")

    def onRemoveDark(self):
        """Override."""
        del self._dark_ma
        self._dark_mean_ma = None

    def sources(self):
        """Override."""
        return [
            (self._output_channel, self._ppt),
        ]

    @profiler("Gotthard Processor")
    def process(self, data):
        """Override."""
        data, meta = data["raw"], data["meta"]
        tid = self.getTrainId(meta)

        raw = self.getPropertyData(data, self._output_channel, self._ppt)

        # check data shape
        if raw.ndim != 2:
            raise ProcessingError(f"Gotthard data must be a 2D array: "
                                  f"actual {raw.ndim}D")

        raw = raw.astype(_PIXEL_DTYPE)

        # check POI index
        max_idx = raw[self._pulse_slicer].shape[0]
        if self._poi_index >= max_idx:
            raise ProcessingError(f"POI index {self._poi_index} out of "
                                  f"boundary [{0} - {max_idx - 1}]")

        # ------------
        # process data
        # ------------

        if self.recordingDark():
            # update the moving average of dark data
            self._dark_ma = raw

            self._dark_mean_ma = np.mean(
                self._dark_ma[self._pulse_slicer], axis=0)

            # During dark recording, no offset correcttion is applied and
            # only dark data and its statistics are displayed.
            spectrum = raw[self._pulse_slicer]
            spectrum_ma = self._dark_ma[self._pulse_slicer]
        else:
            # update the moving average of raw data
            self._raw_ma = raw

            if self.subtractDark() and self._dark_mean_ma is not None:
                spectrum = raw[self._pulse_slicer] - self._dark_mean_ma
                spectrum_ma = self._raw_ma[self._pulse_slicer] - self._dark_mean_ma
            else:
                spectrum = raw[self._pulse_slicer]
                spectrum_ma = self._raw_ma[self._pulse_slicer]

        spectrum_mean = np.mean(spectrum, axis=0)
        spectrum_ma_mean = np.mean(spectrum_ma, axis=0)

        if self._scale == 0:
            x = None
        else:
            x = np.arange(len(spectrum_mean)) * self._scale - self._offset

        self.log.info(f"Train {tid} processed")

        return {
            # x axis of the spectrum
            "x": x,
            # spectrum for the current train
            "spectrum": spectrum,
            # moving average of spectrum
            "spectrum_ma": spectrum_ma,
            # average of the spectrum for the current train over pulses
            "spectrum_mean": spectrum_mean,
            # moving average of spectrum_mean
            "spectrum_ma_mean": spectrum_ma_mean,
            # index of pulse of interest
            "poi_index": self._poi_index,
            # hist, bin_centers, mean, median, std
            "hist": hist_with_stats(
                self.getRoiData(spectrum_ma) if self._hist_over_ma else
                self.getRoiData(spectrum),
                self._bin_range, self._n_bins)
        }

    def reset(self):
        """Override."""
        del self._raw_ma
        del self._dark_ma
