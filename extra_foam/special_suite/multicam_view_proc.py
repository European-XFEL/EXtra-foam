"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from .special_analysis_base import QThreadWorker
from ..utils import profiler


class MultiCamViewProcessor(QThreadWorker):
    """Multi-camera view processor.

    Attributes:
        _output_channels (list): list of output channel names.
        _properties (list): list of properties.
    """
    _N_CAMERAS = 4

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._output_channels = [''] * self._N_CAMERAS
        self._properties = [''] * self._N_CAMERAS

    def onOutputChannelChanged(self, idx: int, value: str):
        self._output_channels[idx] = value

    def onPropertyChanged(self, idx: int, value: str):
        self._properties[idx] = value

    @profiler("Multi-camera views Processor")
    def process(self, data):
        """Override."""
        data, meta = data

        tid = self._get_tid(meta)

        channels = {i: None for i in range(self._N_CAMERAS)}
        images = {i: None for i in range(self._N_CAMERAS)}
        for i, (ch, ppt) in enumerate(zip(self._output_channels,
                                          self._properties)):
            img = self._fetch_property_data(tid, data, ch, ppt)

            images[i] = self._squeeze_camera_image(tid, img)
            channels[i] = ch

        self.log.info(f"Train {tid} processed")

        return {
            "channels": channels,
            "images": images
        }
