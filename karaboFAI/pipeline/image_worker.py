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
from .processors import ImageAssemblerFactory, ImageProcessor
from .data_model import ProcessedData


class ImageWorker(ProcessWorker):
    """Pipeline scheduler."""
    def __init__(self, detector):
        """Initialization."""
        super().__init__('image_worker')

        self._inputs = [KaraboBridge(f"{self._name}:input")]
        self._output = MpOutQueue(f"{self._name}:output")

        # processor pipeline flow:
        # ImageAssembler ->
        #
        # ImageProcessor ->
        self._assembler = ImageAssemblerFactory.create(detector)
        self._processor = ImageProcessor()

        self._tasks = [
            self._assembler, self._processor
        ]

    def _preprocess(self, data):
        """Override."""
        raw, meta = data

        # get the train ID of the first metadata
        # Note: this is better than meta[src_name] because:
        #       1. For streaming AGIPD/LPD data from files, 'src_name' does
        #          not work;
        #       2. Prepare for the scenario where a 2D detector is not
        #          mandatory.
        tid = next(iter(meta.values()))["timestamp.tid"]
        processed = ProcessedData(tid)
        processed.raw = raw
        return processed
