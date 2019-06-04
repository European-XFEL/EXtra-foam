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
import multiprocessing as mp

import psutil
from psutil import NoSuchProcess

from ..metadata import MetaProxy
from ..config import config
from ..ipc import ProcessWorkerLogger, RedisConnection
from ..logger import logger


class ProcessWorker(mp.Process):
    """Base worker class for heavy online data analysis."""

    _db = RedisConnection()

    def __init__(self, name):
        super().__init__()

        self._name = name
        self.source_type = None

        self.log = ProcessWorkerLogger()

        self._inputs = []  # pipe-ins
        self._output = None  # pipe-out

        self._pause_ev = mp.Event()
        self._close_ev = mp.Event()

        self._meta = MetaProxy()

        self._timeout = config["TIMEOUT"]

    @property
    def output(self):
        return self._output

    def run(self):
        """Override."""
        for inp in self._inputs:
            inp.run_in_thread(self._close_ev)
        self._output.run_in_thread(self._close_ev)

        while not self.closing:
            try:
                if not self.running:
                    self._pause_ev.wait()

                    # update metadata
                    self.update()

                    # tell input and output channels to update
                    for inp in self._inputs:
                        inp.update()
                    self._output.update()

                self._run_once()
            except Exception as e:
                self.log.error(repr(e))

    def _run_once(self):
        raise NotImplementedError

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

    def update(self):
        """Update metadata."""
        pass


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


def close_all_child_processes():
    def on_terminate(proc):
        name = find_process_type_by_pid(proc.pid)
        if name is None:
            logger.warning(f"Unknown process {proc} terminated with exit code "
                           f"{proc.returncode}")
        else:
            logger.warning(f"'{name}' process {proc} terminated with "
                           f"exit code {proc.returncode}")

    logger.info("Clean up child processes ...")
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


atexit.register(close_all_child_processes)


class ProcessManager:
    @staticmethod
    def shutdown_redis():
        logger.info(f"Shutting down Redis server ...")
        if not _fai_processes.redis:
            return

        for _, proc in _fai_processes.redis.items():
            try:
                proc.terminate()
                proc.wait(0.5)
            except NoSuchProcess:
                continue

        for _, proc in _fai_processes.redis.items():
            if proc.poll() is None:
                proc.kill()
                proc.wait(0.5)

    @staticmethod
    def shutdown_pipeline():
        logger.info(f"Shutting down pipeline processors ...")

        for _, proc in _fai_processes.pipeline.items():
            if proc.is_alive():
                proc.close()
                proc.join(timeout=0.5)

        for _, proc in _fai_processes.pipeline.items():
            if proc.is_alive():
                proc.terminate()
                proc.join(0.5)
