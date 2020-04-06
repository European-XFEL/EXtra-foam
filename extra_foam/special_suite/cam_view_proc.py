"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import numpy as np

from .special_analysis_base import ProcessingError, QThreadWorker
from ..pipeline.data_model import MovingAverageArray
from ..utils import profiler
from ..config import config, _MAX_INT32


_IMAGE_DTYPE = config['SOURCE_PROC_IMAGE_DTYPE']


class CamViewProcessor(QThreadWorker):
    """Camera view processor.

    Attributes:
        _output_channel (str): output channel name.
        _ppt (str): property name.
        _raw_ma (numpy.ndarray): moving average of the raw image data.
            Shape=(y, x)
        _dark_ma (numpy.ndarray): moving average of the dark data.
            Shape=(pulses, pixels)
    """

    _raw_ma = MovingAverageArray()
    _dark_ma = MovingAverageArray(_MAX_INT32)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._output_channel = ''
        self._ppt = ''

        self._setMaWindow(1)

        del self._dark_ma

    def _setMaWindow(self, v):
        self.__class__._raw_ma.window = v

    def onOutputChannelChanged(self, value: str):
        self._output_channel = value

    def onPropertyChanged(self, value: str):
        self._ppt = value

    def onMaWindowChanged(self, value: str):
        self._setMaWindow(int(value))

    def onRemoveDark(self):
        """Override."""
        del self._dark_ma

    def onLoadDarkRun(self, dirpath):
        """Override."""
        run = self._loadRunDirectory(dirpath)
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

    @profiler("Module scan Processor")
    def process(self, data):
        """Override."""
        data, _ = data

        data = data[self._output_channel]
        tid = data['metadata']["timestamp.tid"]

        img = data[self._ppt].astype(_IMAGE_DTYPE)

        if img.ndim != 2:
            raise ProcessingError(f"Image data must be a 2D array: "
                                  f"actual {img.ndim}D")

        if self._recording_dark:
            self._dark_ma = img
            displayed = self._dark_ma
        else:
            self._raw_ma = img
            displayed = self._raw_ma
            if self._subtract_dark and self._dark_ma is not None:
                # caveat: cannot subtract inplace
                displayed = displayed - self._dark_ma

        self.log.info(f"Train {tid} processed")

        return {
            "displayed": displayed,
        }
