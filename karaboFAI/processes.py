"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

Process management.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import atexit
from collections import namedtuple

import psutil
from psutil import NoSuchProcess

from .config import config
from .logger import logger
from .pipeline.worker import ProcessWorker


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


def _find_process_type_by_pid(pid):
    """Find the name of a process in _fai_processes by pid."""
    for procs in _fai_processes:
        for name, p in procs.items():
            if p.pid == pid:
                return name


def _close_all_child_processes():
    def on_terminate(proc):
        name = _find_process_type_by_pid(proc.pid)
        if name is None:
            logger.warning(f"Unknown {proc} terminated with exit code "
                           f"{proc.returncode}")
        else:
            logger.warning(f"'{name}' {proc} terminated with "
                           f"exit code {proc.returncode}")

    logger.debug("Clean up child processes ...")
    timeout = config["PROCESS_CLEANUP_TIMEOUT"]

    procs = psutil.Process().children()
    for p in procs:
        try:
            p.terminate()
        except NoSuchProcess:
            pass

    _, alive = psutil.wait_procs(
        procs, timeout=timeout, callback=on_terminate)

    for p in alive:
        try:
            p.kill()
        except NoSuchProcess:
            pass

    _, alive = psutil.wait_procs(
        alive, timeout=timeout, callback=on_terminate)

    if alive:
        for p in alive:
            print(f"{p} survived SIGKILL, please clean it manually")


atexit.register(_close_all_child_processes)


def wait_until_redis_shutdown(timeout=5):
    """Wait until the Redis server is down."""
    redis_proc = None
    for proc in psutil.process_iter(attrs=['cmdline']):
        cmdline = proc.info['cmdline']
        if cmdline and 'karaboFAI/thirdparty/bin/redis-server' in cmdline[0]:
            redis_proc = proc

    if redis_proc is not None:
        redis_proc.kill()
        _, alive = psutil.wait_procs([redis_proc], timeout=timeout)

        if alive:
            logger.warning("Redis-server process {} is still alive!")


def shutdown_redis():
    logger.info(f"Shutting down Redis server ...")
    if not _fai_processes.redis:
        return

    for _, proc in _fai_processes.redis.items():
        try:
            proc.terminate()
            proc.wait(timeout=0.5)
        except NoSuchProcess:
            continue

    for _, proc in _fai_processes.redis.items():
        if proc.poll() is None:
            proc.kill()

    for _, proc in _fai_processes.redis.items():
        if proc.poll() is None:
            proc.wait(timeout=0.5)


def shutdown_pipeline():
    logger.info(f"Shutting down pipeline processors ...")

    for _, proc in _fai_processes.pipeline.items():
        if proc.is_alive():
            proc.close()
            proc.join(timeout=0.5)

    for _, proc in _fai_processes.pipeline.items():
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=0.5)


def shutdown_all():
    shutdown_pipeline()
    shutdown_redis()
