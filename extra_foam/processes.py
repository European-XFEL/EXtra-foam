"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

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


ProcessInfo = namedtuple("ProcessInfo", [
    "name",
    "process",
])


ProcessInfoList = namedtuple("ProcessInfoList", [
    "name",
    "foam_name",
    "foam_type",
    "pid",
    "status"
])


# key: process_type
# value: a dictionary, process name in extra-foam:Process instance
FoamProcesses = namedtuple("FoamProcesses", ['redis', 'pipeline'])

_foam_processes = FoamProcesses({}, {})


def register_foam_process(process_info):
    """Register a new process."""
    proc = process_info.process
    name = process_info.name
    if name.lower() == 'redis':
        _foam_processes.redis[name] = proc
    else:
        _foam_processes.pipeline[name] = proc


def list_foam_processes():
    """List the current Foam processes."""

    def get_proc_info(foam_name, foam_type, proc):
        with proc.oneshot():
            return ProcessInfoList(
                proc.name(),
                foam_name,
                foam_type,
                proc.pid,
                proc.status(),
            )

    info_list = []
    children = psutil.Process().children()

    for name, p in _foam_processes.redis.items():
        info_list.append(get_proc_info(name, 'redis', p))
        children.remove(p)

    for name, p in _foam_processes.pipeline.items():
        p_psutil = psutil.Process(p.pid)
        info_list.append(get_proc_info(name, 'pipeline', p_psutil))
        children.remove(p_psutil)

    for child in children:
        info_list.append(get_proc_info(child.name(), 'other', child))

    return info_list


def _find_process_type_by_pid(pid):
    """Find the name of a process in _foam_processes by pid."""
    for procs in _foam_processes:
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
        if cmdline and 'extra_foam/thirdparty/bin/redis-server' in cmdline[0]:
            redis_proc = proc

    if redis_proc is not None:
        redis_proc.kill()
        _, alive = psutil.wait_procs([redis_proc], timeout=timeout)

        if alive:
            logger.warning("Redis-server process {} is still alive!")


def shutdown_redis():
    logger.info(f"Shutting down Redis server ...")
    if not _foam_processes.redis:
        return

    for _, proc in _foam_processes.redis.items():
        try:
            proc.terminate()
            proc.wait(timeout=0.5)
        except NoSuchProcess:
            continue

    for _, proc in _foam_processes.redis.items():
        if proc.poll() is None:
            proc.kill()

    for _, proc in _foam_processes.redis.items():
        if proc.poll() is None:
            proc.wait(timeout=0.5)


def shutdown_pipeline():
    logger.info(f"Shutting down pipeline processors ...")

    for _, proc in _foam_processes.pipeline.items():
        if proc.is_alive():
            proc.close()
            proc.join(timeout=0.5)

    for _, proc in _foam_processes.pipeline.items():
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=0.5)


def shutdown_all():
    shutdown_pipeline()
    shutdown_redis()
