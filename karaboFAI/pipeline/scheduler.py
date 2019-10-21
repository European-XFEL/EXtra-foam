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
    AzimuthalIntegrationProcessorTrain, BinProcessor, CorrelationProcessor,
    PumpProbeProcessor, RoiProcessorTrain, StatisticsProcessor,
)


class Scheduler(ProcessWorker):
    """Pipeline scheduler."""
    def __init__(self):
        """Initialization."""
        super().__init__('scheduler')

        self._inputs = [MpInQueue(f"{self._name}:input")]
        self._output = MpOutQueue(f"{self._name}:output", gui=True)

        self._pp_proc = PumpProbeProcessor()

        self._roi_proc_train = RoiProcessorTrain()
        self._ai_proc_train = AzimuthalIntegrationProcessorTrain()

        self._statistics = StatisticsProcessor()
        self._correlation_proc = CorrelationProcessor()
        self._bin_proc = BinProcessor()

        self._tasks = [
            self._pp_proc,
            self._roi_proc_train,
            self._ai_proc_train,
            self._statistics,
            self._correlation_proc,
            self._bin_proc,
        ]
