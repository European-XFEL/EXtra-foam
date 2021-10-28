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

from .config import _IMAGE_DTYPE, _MAX_INT32
from .special_analysis_base import profiler, QThreadWorker

_DEFAULT_N_BINS = 10
_DEFAULT_BIN_RANGE = "-inf, inf"


class CamViewProcessor(QThreadWorker):
    """Camera view processor.

    Attributes:
        _output_channel (str): output channel name.
        _ppt (str): property name.
        _raw_ma (numpy.ndarray): moving average of the raw image data.
            Shape=(y, x)
        _dark_ma (numpy.ndarray): moving average of the dark data.
            Shape=(pulses, pixels)
        _bin_range (tuple): range of the ROI histogram.
        _n_bins (int): number of bins of the ROI histogram.
    """

    _raw_ma = MovingAverageArray()
    _dark_ma = MovingAverageArray(_MAX_INT32)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._output_channel = ''
        self._ppt = ''

        self.__class__._raw_ma.window = 1

        self._bin_range = self.str2range(_DEFAULT_BIN_RANGE)
        self._n_bins = _DEFAULT_N_BINS

        del self._dark_ma

    def onOutputChannelChanged(self, value: str):
        self._output_channel = value

    def onPropertyChanged(self, value: str):
        self._ppt = value

    def onMaWindowChanged(self, value: str):
        self.__class__._raw_ma.window = int(value)

    def onRemoveDark(self):
        """Override."""
        del self._dark_ma

    def onBinRangeChanged(self, value: tuple):
        self._bin_range = value

    def onNoBinsChanged(self, value: str):
        self._n_bins = int(value)

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
                self._dark_ma = np.mean(arr.values, axis=0, dtype=_IMAGE_DTYPE)
            except Exception as e:
                self.log.error(f"Unexpect exception when getting data array: "
                               f"{repr(e)}")

    @profiler("Camera view processor")
    def process(self, data):
        """Override."""
        data, meta = data["raw"], data["meta"]

        tid = self.getTrainId(meta)

        img = self.squeezeToImage(
            tid, self.getPropertyData(data, self._output_channel, self._ppt))
        if img is None:
            return

        if self.recordingDark():
            self._dark_ma = img
            displayed = self._dark_ma
        else:
            self._raw_ma = img
            displayed = self._raw_ma
            if self.subtractDark() and self._dark_ma is not None:
                # caveat: cannot subtract inplace
                displayed = displayed - self._dark_ma

        self.log.info(f"Train {tid} processed")

        return {
            "displayed": displayed,
            "roi_hist": hist_with_stats(self.getRoiData(displayed),
                                        self._bin_range, self._n_bins),
        }

    def reset(self):
        """Override."""
        del self._raw_ma
        del self._dark_ma
