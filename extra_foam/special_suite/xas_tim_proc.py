"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from .special_analysis_base import QThreadWorker
from ..utils import profiler


class XasTimProcessor(QThreadWorker):
    """XAS-TIM processor.

    Attributes:
    """

    _XGM_PROPERTY = ""
    _DIGITIZER_PROPERTY = ""
    _MONO_PROPERTY = ""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._xgm_output_channel = ""
        self._digitizer_output_channel = ""
        self._mono_device_id = ""

    @profiler("XAS-TIM Processor")
    def process(self, data):
        """Override."""
        data, _ = data

        tid = data['metadata']["timestamp.tid"]

        self.log.info(f"Train {tid} processed")

        return {
        }
