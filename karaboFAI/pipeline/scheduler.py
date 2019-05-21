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
from .data_model import ProcessedData
from .worker import ProcessWorker
from .processors import (
    AzimuthalIntegrationProcessor, _BaseProcessor, CorrelationProcessor,
    PumpProbeProcessor, RoiProcessor, XasProcessor
)
from .exceptions import (
    AggregatingError, AssemblingError, ProcessingError)
from ..config import config, DataSource
from ..helpers import profiler


class Scheduler(ProcessWorker):
    """Pipeline scheduler."""
    def __init__(self, name="scheduler"):
        """Initialization."""
        super().__init__(name)

        self._tasks = []

        self._image_assembler = ImageAssemblerFactory.create(
            config['DETECTOR'])
        self._data_aggregator = DataAggregator()

        # processor pipeline flow:
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

    def __setattr__(self, key, value):
        if isinstance(value, _BaseProcessor):
            self._tasks.append(value)
        super().__setattr__(key, value)

    def run(self):
        """Run the data processor."""
        self.empty_output()  # remove old data

        timeout = config['TIMEOUT']
        print("Scheduler processes started!")

        while not self._shutdown_event.is_set():
            try:
                data = self._input.get(timeout=timeout)
            except Empty:
                continue

            self.update()

            self._data_aggregator.update()
            self._image_assembler.update()

            self._ai_proc.update()
            self._pp_proc.update()
            self._roi_proc.update()
            self._xas_proc.update()
            self._correlation_proc.update()

            processed_data = self._process(data)
            print("Data was processed")
            if processed_data is None:
                continue
            if self._source_type == DataSource.BRIDGE:
                # always keep the latest data in the queue
                try:
                    self._output.put(processed_data, timeout=timeout)
                except Full:
                    self.pop_output()
                    print("Data dropped by the scheduler")
            elif self._source_type == DataSource.FILE:
                # wait until data in the queue has been processed
                while not self._shutdown_event.is_set():
                    try:
                        self._output.put(processed_data, timeout=timeout)
                        break
                    except Full:
                        continue
            else:
                raise ProcessingError(
                    f"Unknown source type {self._source_type}!")

        print("Scheduler shutdown cleanly!")

    @profiler("Process Data (total)")
    def _process(self, data):
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
            return None
        except Exception as e:
            print(f"Unexpected Exception: Train ID: {tid}: " + repr(e))

        try:
            processed = ProcessedData(tid, assembled)
        except Exception as e:
            print(f"Unexpected Exception: Train ID: {tid}: " + repr(e))

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
