"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

Helper functions.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import psutil
import multiprocessing as mp
import functools
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


def get_system_memory():
    """Get the total system memory."""
    return psutil.virtual_memory().total


def check_system_resource():
    """Check the resource of the current system"""
    n_cpus = mp.cpu_count()

    n_gpus = 0

    total_memory = get_system_memory()

    return n_cpus, n_gpus, total_memory


class _MetaSingleton(type):
    """Meta class and bookkeeper for Singletons."""
    _instances = dict()

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]
