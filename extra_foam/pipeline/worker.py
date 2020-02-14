"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import multiprocessing as mp
from threading import Condition
from queue import Empty, Full
import sys
import traceback
import time

from .exceptions import StopPipelineError, ProcessingError
from .pipe import KaraboBridge, MpInQueue, MpOutQueue, _ProcessControlMixin
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
from ..config import config
from ..ipc import RedisConnection
from ..ipc import process_logger as logger
from ..processes import register_foam_process


class ProcessWorker(mp.Process, _ProcessControlMixin):
    """Base worker class for heavy online data analysis."""

    _db = RedisConnection()

    def __init__(self, name, pause_ev, close_ev):
        super().__init__()

        self._name = name
        register_foam_process(name, self)

        self._input = None  # pipe-in
        self._output = None  # pipe-out

        self._tasks = []

        self._pause_ev = pause_ev
        self._close_ev = close_ev

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
        self._input.start()
        self._output.start()

        while not self.closing:
            if not self.running:
                self.wait()

            try:
                self._process_input_output()
            except Exception as e:
                exc_type, exc_value, exc_traceback = sys.exc_info()
                logger.debug(repr(traceback.format_tb(exc_traceback)) +
                             repr(e))
                logger.error(repr(e))

    def _process_input_output(self):
        timeout = self._timeout

        try:
            # get the data from pipe-in
            data = self._input.get(timeout=timeout)
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

        try:
            # always keep the latest data in the queue
            self._output.put_pop(data, timeout=timeout)
        except Empty:
            return

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


class PulseWorker(ProcessWorker):
    """Pipeline worker for pulse-resolved data."""
    def __init__(self, pause_ev, close_ev):
        """Initialization."""
        super().__init__('pulse worker', pause_ev, close_ev)

        self._input = KaraboBridge(pause_ev, close_ev)
        self._output = MpOutQueue(pause_ev, close_ev)

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
    def __init__(self, pause_ev, close_ev):
        """Initialization."""
        super().__init__('train worker', pause_ev, close_ev)

        self._input = MpInQueue(pause_ev, close_ev)
        self._output = MpOutQueue(pause_ev, close_ev, final=True)

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
