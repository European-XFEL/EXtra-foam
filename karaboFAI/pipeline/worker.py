"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

Abstract process worker class.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from collections import namedtuple
import atexit
import psutil
from psutil import NoSuchProcess

import multiprocessing as mp
from threading import Thread
from queue import Empty

from ..metadata import Metadata as mt, MetaProxy
from ..config import config, DataSource
from ..logger import logger


class ProcessWorker(mp.Process):
    """Base worker class for heavy online data analysis."""

    def __init__(self, name):
        super().__init__()

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


ProcessInfo = namedtuple("ProcessInfo", [
    "name",
    "process",
])


# key: process_type
# value: a list of processes
_fai_processes = {
    'redis': [],
    'pipeline': [],
}


def register_fai_process(process_info):
    global _fai_processes

    proc = process_info.process
    name = process_info.name
    if name == 'redis':
        _fai_processes['redis'].append(proc)
    else:
        _fai_processes['pipeline'].append(proc)


def find_process_type_by_pid(pid):
    for key, procs in _fai_processes.items():
        for proc in procs:
            if proc.pid == pid:
                return key


def shutdown_all():
    """Shutdown all child processes."""
    def on_terminate(proc):
        process_type = find_process_type_by_pid(proc.pid)
        if process_type is None:
            logger.warning(f"Unknown process {proc} terminated with exit code "
                           f"{proc.returncode}")
        else:
            logger.warning(f"'{process_type}' process {proc} terminated with "
                           f"exit code {proc.returncode}")

    logger.info("Clean up all child processes ...")

    procs = psutil.Process().children()
    for p in procs:
        p.terminate()

    gone, alive = psutil.wait_procs(procs, timeout=1, callback=on_terminate)

    for p in alive:
        p.kill()


atexit.register(shutdown_all)


class ProcessManager:
    @staticmethod
    def shutdown_redis():
        global _fai_processes

        logger.info(f"Shutting down Redis server ...")
        if not _fai_processes['redis']:
            return

        for proc in _fai_processes['redis']:
            try:
                proc.terminate()
            except NoSuchProcess:
                continue

            proc.wait(0.5)
            if proc.poll() is None:
                proc.kill()

    @staticmethod
    def shutdown_pipeline():
        global _fai_processes

        logger.info(f"Shutting down Karabo bridge client ...")

        for proc in _fai_processes['pipeline']:
            proc.shutdown()
            if proc.is_alive():
                proc.join(timeout=0.5)

            if proc.is_alive():
                proc.terminate()
                proc.join(0.5)
