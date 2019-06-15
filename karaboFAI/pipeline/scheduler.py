"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

Pipeline scheduler.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from .worker import ProcessWorker
from .pipe import MpInQueue, MpOutQueue
from .processors import (
    AzimuthalIntegrationProcessor, BinProcessor, CorrelationProcessor,
    PumpProbeProcessor, XgmProcessor, RoiProcessor, XasProcessor
)


class Scheduler(ProcessWorker):
    """Pipeline scheduler."""
    def __init__(self):
        """Initialization."""
        super().__init__('scheduler')

        self._inputs = [MpInQueue(f"{self._name}:input")]
        self._output = MpOutQueue(f"{self._name}:output", gui=True)

        self._xgm_proc = XgmProcessor()

        self._pp_proc = PumpProbeProcessor()
        self._correlation_proc = CorrelationProcessor()
        self._bin_proc = BinProcessor()

        self._roi_proc = RoiProcessor()
        self._ai_proc = AzimuthalIntegrationProcessor()
        self._xas_proc = XasProcessor()

        self._tasks = [
            self._xgm_proc, self._pp_proc, self._roi_proc, self._ai_proc,
            self._correlation_proc, self._bin_proc, self._xas_proc
        ]
