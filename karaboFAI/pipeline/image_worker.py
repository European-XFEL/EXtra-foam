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
    AzimuthalIntegrationProcessorPulse,
    PostPulseFilter, ImageAssemblerFactory,
    ImageProcessorPulse, ImageProcessorTrain, RoiProcessorPulse,
    XgmExtractor, XgmPulseFilter
)
from ..config import config


class ImageWorker(ProcessWorker):
    """Pipeline scheduler."""
    def __init__(self):
        """Initialization."""
        super().__init__('image_worker')

        self._inputs = [KaraboBridge(f"{self._name}:input")]
        self._output = MpOutQueue(f"{self._name}:output")

        self._xgm_extractor = XgmExtractor()
        self._xgm_pulse_filter = XgmPulseFilter()
        self._assembler = ImageAssemblerFactory.create(config['DETECTOR'])
        self._image_proc_pulse = ImageProcessorPulse()
        self._roi_proc_pulse = RoiProcessorPulse()
        self._ai_proc_pulse = AzimuthalIntegrationProcessorPulse()
        self._post_pulse_filter = PostPulseFilter()
        self._image_proc_train = ImageProcessorTrain()

        self._tasks = [
            self._xgm_extractor,
            self._xgm_pulse_filter,
            self._assembler,
            self._image_proc_pulse,
            self._roi_proc_pulse,
            self._ai_proc_pulse,
            self._post_pulse_filter,
            self._image_proc_train,
        ]
