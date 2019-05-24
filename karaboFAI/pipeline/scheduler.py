"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

Pipeline scheduler.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import sys, traceback
from queue import Empty, Full

from .image_assembler import ImageAssemblerFactory
from .data_aggregator import DataAggregator
from .worker import ProcessWorker
from .processors import (
    AzimuthalIntegrationProcessor, _BaseProcessor, CorrelationProcessor,
    ImageProcessor, PumpProbeProcessor, RoiProcessor, XasProcessor
)
from .exceptions import (
    AggregatingError, AssemblingError, ProcessingError)
from ..config import config, DataSource
from ..helpers import profiler


class Scheduler(ProcessWorker):
    """Pipeline scheduler."""
    def __init__(self, detector, name='scheduler'):
        """Initialization."""
        super().__init__(name)

        self._tasks = []

        self._image_assembler = ImageAssemblerFactory.create(detector)
        self._image_proc = ImageProcessor()
        self._data_aggregator = DataAggregator()

        # processor pipeline flow:
        # ImageProcessor ->
        #
        # PumpProbeProcessor ->
        #
        # RoiProcessor, AzimuthalIntegrationProcessor ->
        #
        # CorrelationProcessor, XasProcessor

        self._pp_proc = PumpProbeProcessor()
        self._roi_proc = RoiProcessor()
        self._ai_proc = AzimuthalIntegrationProcessor()
        self._correlation_proc = CorrelationProcessor()
        self._xas_proc = XasProcessor()

        self._tasks = [
            self._pp_proc, self._roi_proc, self._ai_proc,
            self._correlation_proc, self._xas_proc
        ]

    def _run_once(self):
        """Run the data processor."""
        timeout = self._timeout

        try:
            data = self._input.get(timeout=timeout)
        except Empty:
            return

        self.update()

        self._data_aggregator.update()
        self._image_assembler.update()

        self._image_proc.update()
        self._ai_proc.update()
        self._pp_proc.update()
        self._roi_proc.update()
        self._xas_proc.update()
        self._correlation_proc.update()

        processed_data = self._process_core(data)
        if processed_data is None:
            return

        if self._source_type == DataSource.BRIDGE:
            # always keep the latest data in the queue
            try:
                self._output.put(processed_data, timeout=timeout)
            except Full:
                self.pop_output()
                print("Data dropped by the scheduler")
        elif self._source_type == DataSource.FILE:
            # wait until data in the queue has been processed
            while not self.closing:
                try:
                    self._output.put(processed_data, timeout=timeout)
                    break
                except Full:
                    continue
        else:
            raise ProcessingError(
                f"Unknown source type {self._source_type}!")

    @profiler("Process Data (total)")
    def _process_core(self, data):
        """Process data received from the bridge."""
        raw, meta = data

        # get the train ID of the first metadata
        # Note: this is better than meta[src_name] because:
        #       1. For streaming AGIPD/LPD data from files, 'src_name' does
        #          not work;
        #       2. Prepare for the scenario where a 2D detector is not
        #          mandatory.
        tid = next(iter(meta.values()))["timestamp.tid"]

        try:
            assembled = self._image_assembler.assemble(raw)
        except AssemblingError as e:
            print(f"Train ID: {tid}: " + repr(e))
            return
        except Exception as e:
            print(f"Unexpected Exception: Train ID: {tid}: " + repr(e))
            return

        try:
            processed = self._image_proc.process_image(tid, assembled)
        except Exception as e:
            print(f"Unexpected Exception: Train ID: {tid}: " + repr(e))
            return

        try:
            self._data_aggregator.aggregate(processed, raw)
        except AggregatingError as e:
            print(f"Train ID: {tid}: " + repr(e))
        except Exception as e:
            print(f"Unexpected Exception: Train ID: {tid}: " + repr(e))

        for task in self._tasks:
            try:
                task.run_once(processed, raw)
            except ProcessingError as e:
                exc_type, exc_value, exc_traceback = sys.exc_info()
                print(repr(traceback.format_tb(exc_traceback)))
                print(f"Train ID: {tid}: " + repr(e))
            except Exception as e:
                exc_type, exc_value, exc_traceback = sys.exc_info()
                print(repr(traceback.format_tb(exc_traceback)))
                print(f"Unexpected Exception: Train ID: {tid}: " + repr(e))

        return processed
