"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

Abstract process worker class as well as process management.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import atexit
from collections import namedtuple
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


ProcessInfoList = namedtuple("ProcessInfoList", [
    "name",
    "fai_name",
    "fai_type",
    "pid",
    "status"
])


# key: process_type
# value: a dictionary, process name in karaboFAI:Process instance
FaiProcesses = namedtuple("FaiProcesses",
                          ['redis', 'pipeline'])

_fai_processes = FaiProcesses({}, {})


def register_fai_process(process_info):
    """Register a new process."""
    global _fai_processes

    proc = process_info.process
    name = process_info.name
    if name == 'redis':
        _fai_processes.redis[name] = proc
    else:
        assert isinstance(proc, ProcessWorker)
        _fai_processes.pipeline[name] = proc


def list_fai_processes():
    """List the current FAI processes."""

    def get_proc_info(fai_name, fai_type, proc):
        with proc.oneshot():
            return ProcessInfoList(
                proc.name(),
                fai_name,
                fai_type,
                proc.pid,
                proc.status(),
            )

    info_list = []

    for name, p in _fai_processes.redis.items():
        info_list.append(get_proc_info(name, 'redis', p))

    for name, p in _fai_processes.pipeline.items():
        info_list.append(
            get_proc_info(name, 'pipeline', psutil.Process(p.pid)))

    return info_list


def find_process_type_by_pid(pid):
    """Find the name of a process in _fai_processes by pid."""
    for procs in _fai_processes:
        for name, p in procs.items():
            if p.pid == pid:
                return name


def shutdown_all():
    """Shutdown all child processes."""
    def on_terminate(proc):
        name = find_process_type_by_pid(proc.pid)
        if name is None:
            logger.warning(f"Unknown process {proc} terminated with exit code "
                           f"{proc.returncode}")
        else:
            logger.warning(f"'{name}' process {proc} terminated with "
                           f"exit code {proc.returncode}")

    logger.info("Clean up all child processes ...")
    timeout = config["PROCESS_CLEANUP_TIMEOUT"]

    procs = psutil.Process().children()
    for p in procs:
        try:
            p.terminate()
        except NoSuchProcess:
            pass

    gone, alive = psutil.wait_procs(
        procs, timeout=timeout, callback=on_terminate)

    for p in alive:
        try:
            p.kill()
        except NoSuchProcess:
            pass

    gone, alive = psutil.wait_procs(
        alive, timeout=timeout, callback=on_terminate)

    if alive:
        for p in alive:
            print(f"process {p} survived SIGKILL, please clean it manually")


atexit.register(shutdown_all)


class ProcessManager:
    @staticmethod
    def shutdown_redis():
        logger.info(f"Shutting down Redis server ...")
        if not _fai_processes.redis:
            return

        for _, proc in _fai_processes.redis.items():
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

        logger.info(f"Shutting down pipeline processors ...")

        for _, proc in _fai_processes.pipeline.items():
            if proc.is_alive():
                proc.shutdown()
                proc.join(timeout=0.5)

            if proc.is_alive():
                proc.terminate()
                proc.join(0.5)
