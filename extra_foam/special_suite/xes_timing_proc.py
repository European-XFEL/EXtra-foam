"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from extra_foam.algorithms import SimplePairSequence

from .special_analysis_base import profiler, QThreadWorker


_MAX_WINDOW = 180000  # 60 s * 10 train/s * 300 pulses/train


class XesTimingProcessor(QThreadWorker):
    """XES timing processor.

    Attributes:
        _output_channel (str): output channel name.
        _ppt (str): property name.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._output_channel = ''
        self._ppt = "data.image.data"  # 'data.adc'

        self._delay_device = ''
        self._delay_ppt = 'integerProperty'

        self._delay_scan = SimplePairSequence(max_len=_MAX_WINDOW)

    def onOutputChannelChanged(self, value: str):
        self._output_channel = value

    def onDelayDeviceChanged(self, value: str):
        self._delay_device = value

    def sources(self):
        """Override."""
        return [
            (self._output_channel, self._ppt, 1),
            (self._delay_device, self._delay_ppt, 0),
        ]

    @profiler("XES timing processor")
    def process(self, data):
        """Override."""
        data, meta = data["raw"], data["meta"]

        tid = self.getTrainId(meta)

        img = self.squeezeToImage(
            tid, self.getPropertyData(data, self._output_channel, self._ppt))
        if img is None:
            return

        delay = self.getPropertyData(
            data, self._delay_device, self._delay_ppt)

        self._delay_scan.append((tid, delay))

        self.log.info(f"Train {tid} processed")

        return {
            "displayed": img,
            "delay_scan": self._delay_scan.data(),
        }
