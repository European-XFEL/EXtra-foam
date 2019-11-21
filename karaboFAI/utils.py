"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import os
import psutil
import multiprocessing as mp
import functools
import subprocess
from threading import RLock
import time

from .logger import logger


# profiler will only print out information if the execution of the given
# function takes more than the threshold value.
PROFILER_THREASHOLD = 10.0  # in ms


def profiler(info, *, process_time=False):
    def wrap(f):
        @functools.wraps(f)
        def timed_f(*args, **kwargs):
            if process_time:
                timer = time.process_time
            else:
                timer = time.perf_counter

            t0 = timer()
            result = f(*args, **kwargs)
            dt_ms = 1000 * (timer() - t0)
            if dt_ms > PROFILER_THREASHOLD:
                logger.debug(f"Process time spent on {info}: {dt_ms:.3f} ms")
            return result
        return timed_f
    return wrap


_NOT_FOUND = object()


class cached_property:
    """cached_property since Python3.8"""
    def __init__(self, func):
        self.func = func
        self.attrname = None
        self.__doc__ = func.__doc__
        self.lock = RLock()

    def __set_name__(self, owner, name):
        if self.attrname is None:
            self.attrname = name
        elif name != self.attrname:
            raise TypeError(
                f"Cannot assign the same cached_property to two different "
                f"names ({self.attrname!r} and {name!r})."
            )

    def __get__(self, instance, owner):
        if instance is None:
            return self
        if self.attrname is None:
            raise TypeError(
                "Cannot use cached_property instance without calling "
                "__set_name__ on it.")
        try:
            cache = instance.__dict__
        except AttributeError:
            # not all objects have __dict__ (e.g. class defines slots)
            msg = (
                f"No '__dict__' attribute on {type(instance).__name__!r} "
                f"instance to cache {self.attrname!r} property."
            )
            raise TypeError(msg) from None
        val = cache.get(self.attrname, _NOT_FOUND)
        if val is _NOT_FOUND:
            with self.lock:
                # check if another thread filled cache while we awaited lock
                val = cache.get(self.attrname, _NOT_FOUND)
                if val is _NOT_FOUND:
                    val = self.func(instance)
                    try:
                        cache[self.attrname] = val
                    except TypeError:
                        msg = (
                            f"The '__dict__' attribute on "
                            f"{type(instance).__name__!r} instance does not "
                            f"support item assignment for caching "
                            f"{self.attrname!r} property."
                        )
                        raise TypeError(msg) from None
        return val


def _get_system_cpu_info():
    """Get the system cpu information."""
    class CpuInfo:
        def __init__(self, n_cpus=None):
            self.n_cpus = n_cpus

        def __repr__(self):
            return f"[CPU] count: {self.n_cpus}"

    return CpuInfo(mp.cpu_count())


def _get_system_memory_info():
    """Get the system memory information."""
    class MemoryInfo:
        def __init__(self, total_memory=None, used_memory=None):
            self.total_memory = total_memory
            self.used_memory = used_memory

        def __repr__(self):
            return f"[Memory] " \
                   f"total: {self.total_memory / 1024**3:.1f} GB, " \
                   f"used: {self.used_memory / 1024**3:.1f} GB"

    mem = psutil.virtual_memory()
    return MemoryInfo(mem.total, mem.used)


def _get_system_gpu_info():
    """Get the system GPU information."""
    class GpuInfo:
        def __init__(self,
                     gpu_name=None,
                     total_memory=None,
                     used_memory=None):
            self.name = gpu_name
            self.total_memory = total_memory
            self.used_memory = used_memory

        def __repr__(self):
            if self.name is None:
                return f"[GPU] Not found"
            return f"[GPU] " \
                   f"name: {self.name}, " \
                   f"total: {self.total_memory / 1024**3:.1f} GB, " \
                   f"used: {self.used_memory / 1024**3:.1f} GB"

    command = ["nvidia-smi",
               "--query-gpu=name,memory.total,memory.used",
               "--format=csv,noheader,nounits"]

    try:
        p = psutil.Popen(command, stdout=subprocess.PIPE)
        stdout, _ = p.communicate()

        output = stdout.decode('UTF-8')
        info = []
        for line in output.split(os.linesep):
            if line:
                splitted = line.split(',')
                if len(splitted) != 3:
                    logger.error(
                        f"Received unexpected query result for GPU: {line}")
                    info.append(GpuInfo())
                else:
                    name = splitted[0]
                    total = int(splitted[1]) * 1024**2  # MB -> byte
                    used = int(splitted[2]) * 1024**2  # MB -> byte
                    info.append(GpuInfo(name, total, used))

        if len(info) == 1:
            return info[0]
        return info
    except FileNotFoundError as e:
        # raised when 'nvidia-smi' does not exist
        logger.info(repr(e))
        return GpuInfo()
    except Exception as e:
        # We don't want to prevent karaboFAI from starting simply because
        # failing to get the GPU information.
        logger.info(
            f"Unexpected error when querying GPU information: {repr(e)}")
        return GpuInfo()


def check_system_resource():
    """Check the resource of the current system"""
    cpu_info = _get_system_cpu_info()
    gpu_info = _get_system_gpu_info()
    memory_info = _get_system_memory_info()

    return cpu_info, gpu_info, memory_info


class _MetaSingleton(type):
    """Meta class and bookkeeper for Singletons."""
    _instances = dict()

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


def query_yes_no(question):
    """Ask a yes/no question and return the answer.

    :param str question: the question string.

    :return bool: True for yes and False for no.
    """
    ans = input(f"{question} (y/n)").lower()
    while True:
        if ans not in ['y', 'yes', 'n', 'no']:
            ans = input('please enter yes (y) or no (n): ')
            continue

        if ans == 'y' or ans == 'yes':
            return True
        if ans == 'n' or ans == 'no':
            return False
