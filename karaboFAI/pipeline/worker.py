"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

Abstract process worker class.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import multiprocessing as mp
from queue import Empty, Full
import sys
import traceback
import time

from .exceptions import StopPipelineError, ProcessingError, ProcessingError
from ..metadata import MetaProxy
from ..metadata import Metadata as mt
from ..config import config, DataSource
from ..ipc import ProcessWorkerLogger, RedisConnection


class ProcessWorker(mp.Process):
    """Base worker class for heavy online data analysis."""

    _db = RedisConnection()

    def __init__(self, name):
        super().__init__()

        self._name = name

        self.log = ProcessWorkerLogger()

        self._inputs = []  # pipe-ins
        self._output = None  # pipe-out

        self._tasks = []

        self._pause_ev = mp.Event()
        self._close_ev = mp.Event()

        self._meta = MetaProxy()

        self._timeout = config["TIMEOUT"]

        # the time when the previous data processing was finished
        self._prev_processed_time = None

    @property
    def name(self):
        return self._name

    @property
    def output(self):
        return self._output

    def connectInputToOutput(self, output):
        if isinstance(output, list):
            if len(self._inputs) != len(output):
                raise ValueError

            for inp, oup in zip(self._inputs, output):
                inp.connect(oup)

        for inp in self._inputs:
            inp.connect(output)

    def run(self):
        """Override."""
        timeout = self._timeout
        src_type = None

        # start input and output pipes
        for inp in self._inputs:
            inp.run_in_thread(self._close_ev)
        self._output.run_in_thread(self._close_ev)

        while not self.closing:
            try:
                if not self.running:
                    self._pause_ev.wait()

                    # update source type, which determines the behavior of
                    # pipeline
                    src_type = DataSource(
                        int(self._meta.get(mt.DATA_SOURCE, 'source_type')))

                    # tell input and output channels to update
                    for inp in self._inputs:
                        inp.update()
                    self._output.update()

                for inp in self._inputs:
                    try:
                        # get the data from pipe-in
                        data = inp.get(timeout=timeout)
                    except Empty:
                        continue

                    try:
                        self._run_tasks(data)
                    except StopPipelineError:
                        self._prev_processed_time = time.time()
                        break

                    if self._prev_processed_time is not None:
                        fps = 1.0 / (time.time() - self._prev_processed_time)
                        self.log.debug(f"FPS of {self._name}: {fps:>4.1f} Hz")
                    self._prev_processed_time = time.time()

                    if src_type == DataSource.BRIDGE:
                        # always keep the latest data in the queue
                        try:
                            self._output.put_pop(data, timeout=timeout)
                        except Empty:
                            continue
                    elif src_type == DataSource.FILE:
                        # wait until data in the queue has been processed
                        while not self.closing:
                            try:
                                self._output.put(data, timeout=timeout)
                                break
                            except Full:
                                continue
                    else:
                        raise ProcessingError(
                            f"Unknown source type {src_type}!")
            except Exception as e:
                self.log.error(repr(e))

    def _run_tasks(self, data):
        """Run all tasks for once:

        :param dict data: a dictionary which is passed around processors.
        """
        tid = data['tid']

        for task in self._tasks:
            # TODO: improve
            try:
                task.run_once(data)
            except StopPipelineError as e:
                exc_type, exc_value, exc_traceback = sys.exc_info()
                self.log.debug(repr(traceback.format_tb(exc_traceback))
                               + repr(e))
                self.log.error(repr(e))
                raise
            except ProcessingError as e:
                exc_type, exc_value, exc_traceback = sys.exc_info()
                self.log.debug(repr(traceback.format_tb(exc_traceback))
                               + repr(e))
                self.log.error(repr(e))
            except Exception as e:
                exc_type, exc_value, exc_traceback = sys.exc_info()
                self.log.debug(f"Unexpected Exception! Train ID: {tid}: " +
                               repr(traceback.format_tb(exc_traceback)) +
                               repr(e))
                self.log.error(repr(e))

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
