"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

Abstract process worker class.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import os
import signal
from collections import namedtuple
from weakref import WeakValueDictionary
import multiprocessing as mp
from threading import Thread
from queue import Empty

from ..metadata import Metadata as mt, MetaProxy
from ..config import config, DataSource, redis_connection
from ..logger import logger


ProcessInfo = namedtuple("ProcessInfo", [
    "process",
    "stdout_file",
    "stderr_file",
])


_workers = dict()


class ProcessWorker(mp.Process):
    """Base worker class for heavy online data analysis."""

    def __init__(self, name):
        super().__init__()

        _workers[name] = ProcessInfo(
            self, stdout_file=None, stderr_file=None,)

        self._name = name
        self._source_type = None

        # each Manager instance will start a process
        self._input = mp.Queue(maxsize=config["MAX_QUEUE_SIZE"])
        self._output = mp.Queue(maxsize=config["MAX_QUEUE_SIZE"])

        self._pause_event = mp.Event()
        self._shutdown_event = mp.Event()

        self._meta = MetaProxy()

        self._timeout = config["TIMEOUT"]

    @property
    def output(self):
        return self._output

    def connect_input(self, worker):
        if not isinstance(worker, ProcessWorker):
            raise TypeError("QThreadWorker is only allowed to connect "
                            "QThreadWorker instance.")

        self._input = worker.output

    def run(self):
        """Override."""
        th = Thread(target=self._monitor_queue)
        th.daemon = True
        th.start()

        print(f"{self._name} process started")
        while not self.closing:
            try:
                self._run_once()
            except Exception as e:
                print(repr(e))

    def _run_once(self):
        raise NotImplementedError

    def _monitor_queue(self):
        self.empty_output()  # remove old data

        self._shutdown_event.wait()

        self._input.cancel_join_thread()
        self._output.cancel_join_thread()

    def shutdown(self):
        self._shutdown_event.set()
        self._pause_event.set()

    @property
    def closing(self):
        return self._shutdown_event.is_set()

    def activate(self):
        self._pause_event.set()

    def pause(self):
        self._pause_event.clear()

    @property
    def running(self):
        return self._pause_event.is_set()

    def wait(self):
        self._pause_event.wait()

    def empty_output(self):
        """Empty the output queue."""
        while not self._output.empty():
            try:
                self._output.get_nowait()
            except Empty:
                break

    def pop_output(self):
        """Remove and return an item from the output queue"""
        try:
            return self._output.get_nowait()
        except Empty:
            pass

    def update(self):
        self._source_type = DataSource(
            int(self._meta.get(mt.DATA_SOURCE, 'source_type')))


class ProcessProxy:
    def shutdown_all(self):
        for name, info in _workers.items():
            proc = info.process
            logger.info(f"Shutting down {name}...")
            proc.shutdown()
            if proc.is_alive():
                proc.join(timeout=1.0)

            if proc.is_alive():
                logger.info(f"Terminating {name} after 'join' failed...")
                proc.terminate()
                proc.join()

            logger.info(f"{name} is down!")
