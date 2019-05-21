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
from queue import Empty

from ..metadata import Metadata as mt
from ..config import config, DataSource, redis_connection


class ProcessWorker(mp.Process):
    """Base worker class for heavy online data analysis."""
    def __init__(self, name):
        super().__init__()

        self._name = name
        self._source_type = None

        self._input = mp.Manager().Queue(maxsize=config["MAX_QUEUE_SIZE"])
        self._output = mp.Manager().Queue(maxsize=config["MAX_QUEUE_SIZE"])

        self._shutdown_event = mp.Event()

        self._db = redis_connection()

    @property
    def output(self):
        return self._output

    def connect_input(self, worker):
        if not isinstance(worker, ProcessWorker):
            raise TypeError("QThreadWorker is only allowed to connect "
                            "QThreadWorker instance.")

        self._input = worker.output

    def shutdown(self):
        self._shutdown_event.set()

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
            int(self._db.hget(mt.DATA_SOURCE, 'source_type')))
