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
from .pipe import KaraboBridge, MpOutQueue
from .processors import (
    AzimuthalIntegrationProcessorPulse, DataReductionProcessor,
    ImageAssemblerFactory, ImageProcessorPulse, ImageProcessorTrain,
    RoiProcessorPulse
)
from ..config import config


class ImageWorker(ProcessWorker):
    """Pipeline scheduler."""
    def __init__(self):
        """Initialization."""
        super().__init__('image_worker')

        self._inputs = [KaraboBridge(f"{self._name}:input")]
        self._output = MpOutQueue(f"{self._name}:output")

        self._assembler = ImageAssemblerFactory.create(config['DETECTOR'])
        self._image_proc_pulse = ImageProcessorPulse()
        self._roi_proc = RoiProcessorPulse()
        self._ai_proc = AzimuthalIntegrationProcessorPulse()
        self._data_reduction_proc = DataReductionProcessor()
        self._image_proc_train = ImageProcessorTrain()

        self._tasks = [
            self._assembler,
            self._image_proc_pulse,
            self._roi_proc,
            self._ai_proc,
            self._data_reduction_proc,
            self._image_proc_train,
        ]
