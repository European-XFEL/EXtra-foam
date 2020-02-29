"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import multiprocessing as mp
from threading import Event
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
from ..config import config, PipelineSlowPolicy
from ..ipc import RedisConnection
from ..ipc import process_logger as logger
from ..processes import register_foam_process
from ..database import MonProxy


class ProcessWorker(mp.Process):
    """Base worker class for heavy online data analysis."""

    _db = RedisConnection()

    def __init__(self, name, pause_ev, close_ev):
        super().__init__()

        self._name = name
        register_foam_process(name, self)

        self._slow_policy = config["PIPELINE_SLOW_POLICY"]

        self._input = None  # pipe-in
        self._output = None  # pipe-out

        self._tasks = []

        self._pause_ev = pause_ev
        self._close_ev = close_ev
        self._input_update_ev = Event()
        self._output_update_ev = Event()

        # the time when the previous data processing was finished
        self._prev_processed_time = None

        self._mon = MonProxy()

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

        data_out = None
        while not self.closing:
            if not self.running:
                data_out = None

                self.wait()
                self.notify_update()

            if data_out is None:
                try:
                    # get the data from pipe-in
                    data_out = self._input.get()

                    try:
                        self._run_tasks(data_out)
                    except StopPipelineError:
                        tid = data_out["processed"].tid
                        self._mon.add_tid_with_timestamp(tid, dropped=True)
                        logger.info(f"Train {tid} dropped!")
                        data_out = None

                except Empty:
                    pass

            if data_out is not None:
                # TODO: still put the data but signal the data has been dropped.
                if self._slow_policy == PipelineSlowPolicy.WAIT:
                    try:
                        self._output.put(data_out)
                        data_out = None
                    except Full:
                        pass
                else:
                    # always keep the latest data in the cache
                    self._output.put_pop(data_out)
                    data_out = None

            time.sleep(0.001)

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

    @property
    def closing(self):
        return self._close_ev.is_set()

    @property
    def running(self):
        return self._pause_ev.is_set()

    def wait(self):
        self._pause_ev.wait()

    def notify_update(self):
        self._input_update_ev.set()
        self._output_update_ev.set()


class PulseWorker(ProcessWorker):
    """Pipeline worker for pulse-resolved data."""
    def __init__(self, pause_ev, close_ev):
        """Initialization."""
        super().__init__('pulse worker', pause_ev, close_ev)

        self._input = KaraboBridge(self._input_update_ev, pause_ev, close_ev)
        self._output = MpOutQueue(self._output_update_ev, pause_ev, close_ev)

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

        self._input = MpInQueue(self._input_update_ev, pause_ev, close_ev)
        self._output = MpOutQueue(self._output_update_ev, pause_ev, close_ev,
                                  final=True)

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
