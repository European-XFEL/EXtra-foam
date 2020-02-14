"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import multiprocessing as mp
from queue import Empty, Full
import sys
import traceback
import time

from .exceptions import StopPipelineError, ProcessingError
from .pipe import KaraboBridge, MpInQueue, MpOutQueue
from .processors import (
    DigitizerProcessor,
    AzimuthalIntegProcessorPulse, AzimuthalIntegProcessorTrain,
    BinningProcessor,
    Broker,
    CorrelationProcessor,
    ImageAssemblerFactory,
    ImageProcessor,
    CtrlDataProcessor,
    PostPulseFilter,
    PumpProbeProcessor,
    ImageRoiPulse, ImageRoiTrain,
    HistogramProcessor,
    XgmProcessor,
    TrXasProcessor,
)
from ..config import config, DataSource
from ..ipc import RedisConnection
from ..ipc import process_logger as logger


class ProcessWorker(mp.Process):
    """Base worker class for heavy online data analysis."""

    _db = RedisConnection()

    def __init__(self, name):
        super().__init__()

        self._name = name

        self._input = None  # pipe-in
        self._output = None  # pipe-out

        self._tasks = []

        self._pause_ev = mp.Event()
        self._close_ev = mp.Event()

        self._timeout = config["PIPELINE_TIMEOUT"]

        # the time when the previous data processing was finished
        self._prev_processed_time = None

    @property
    def name(self):
        return self._name

    @property
    def input(self):
        return self._input

    @property
    def output(self):
        return self._output

    def run(self):
        """Override."""
        # start input and output pipes
        self._input.start(self._close_ev)
        self._output.start(self._close_ev)

        while not self.closing:
            try:
                self._process_input_output()
            except Exception as e:
                exc_type, exc_value, exc_traceback = sys.exc_info()
                logger.debug(repr(traceback.format_tb(exc_traceback)) +
                             repr(e))
                logger.error(repr(e))

    def _process_input_output(self):
        timeout = self._timeout

        if not self.running:
            self._pause_ev.wait()

            # tell input and output channels to update
            self._input.update()
            self._output.update()

        try:
            # get the data from pipe-in
            data = self._input.get(timeout=timeout)
            det = data['catalog'].main_detector
            src_type = data['meta'][det]['source_type']
        except Empty:
            return

        try:
            self._run_tasks(data)
        except StopPipelineError:
            self._prev_processed_time = time.time()
            return

        if self._prev_processed_time is not None:
            fps = 1.0 / (time.time() - self._prev_processed_time)
            logger.debug(f"FPS of {self._name}: {fps:>4.1f} Hz")
        self._prev_processed_time = time.time()

        if src_type == DataSource.BRIDGE:
            # always keep the latest data in the queue
            try:
                self._output.put_pop(data, timeout=timeout)
            except Empty:
                return
        elif src_type == DataSource.FILE:
            # wait until data in the queue has been processed
            while not self.closing:
                try:
                    self._output.put(data, timeout=timeout)
                    break
                except Full:
                    return
        else:
            raise ProcessingError(f"Unknown source type {src_type}!")

    def _run_tasks(self, data):
        """Run all tasks for once:

        :param dict data: a dictionary which is passed around processors.
        """
        for task in self._tasks:
            try:
                task.run_once(data)
            except StopPipelineError as e:
                exc_type, exc_value, exc_traceback = sys.exc_info()
                logger.debug(repr(traceback.format_tb(exc_traceback))
                             + repr(e))
                logger.error(repr(e))
                raise
            except ProcessingError as e:
                exc_type, exc_value, exc_traceback = sys.exc_info()
                logger.debug(repr(traceback.format_tb(exc_traceback))
                             + repr(e))
                logger.error(repr(e))
            except Exception as e:
                exc_type, exc_value, exc_traceback = sys.exc_info()
                logger.debug(f"Unexpected Exception!: " +
                             repr(traceback.format_tb(exc_traceback)) +
                             repr(e))
                logger.error(repr(e))

    def close(self):
        self._close_ev.set()
        self._pause_ev.set()

    @property
    def closing(self):
        return self._close_ev.is_set()

    @property
    def running(self):
        return self._pause_ev.is_set()

    def wait(self):
        self._pause_ev.wait()

    def resume(self):
        """Resume the worker.

        Note: this method is called by the main process.
        """
        self._pause_ev.set()

    def pause(self):
        """Pause the worker.

        Note: this method is called by the main process.
        """
        self._pause_ev.clear()


class PulseWorker(ProcessWorker):
    """Pipeline worker for pulse-resolved data."""
    def __init__(self):
        """Initialization."""
        super().__init__('pulse worker')

        self._input = KaraboBridge()
        self._output = MpOutQueue()

        self._broker = Broker()
        self._ctrl_data_proc = CtrlDataProcessor()
        self._xgm_proc = XgmProcessor()
        self._digitizer_proc = DigitizerProcessor()
        self._assembler = ImageAssemblerFactory.create(config['DETECTOR'])
        self._image_proc = ImageProcessor()
        self._image_roi = ImageRoiPulse()
        self._ai_proc = AzimuthalIntegProcessorPulse()
        self._post_pulse_filter = PostPulseFilter()
        self._pp_proc = PumpProbeProcessor()

        self._tasks = [
            self._broker,
            self._xgm_proc,
            self._digitizer_proc,
            self._ctrl_data_proc,
            self._assembler,
            self._image_proc,
            self._image_roi,
            self._ai_proc,
            self._post_pulse_filter,
            self._pp_proc,
        ]


class TrainWorker(ProcessWorker):
    """Pipeline worker for train-resolved data."""
    def __init__(self):
        """Initialization."""
        super().__init__('train worker')

        self._input = MpInQueue()
        self._output = MpOutQueue(drop=True, final=True)

        self._image_roi = ImageRoiTrain()
        self._ai_proc = AzimuthalIntegProcessorTrain()

        self._histogram = HistogramProcessor()
        self._correlation_proc = CorrelationProcessor()
        self._binning_proc = BinningProcessor()

        self._tr_xas = TrXasProcessor()

        self._tasks = [
            self._image_roi,
            self._ai_proc,
            self._histogram,
            self._correlation_proc,
            self._binning_proc,
            self._tr_xas,
        ]
