"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import re
import os
import psutil
import socket
import multiprocessing as mp
import functools
import subprocess
from dataclasses import dataclass
from collections import namedtuple
from threading import RLock, Thread
import time

import numpy as np
import xarray as xr

from metropc.core import View
from metropc.client import ViewOutput
from metropc.viewdef import ViewDecorator

from .logger import logger


# profiler will only print out information if the execution of the given
# function takes more than the threshold value.
PROFILER_THREASHOLD = 1.0  # in ms


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



class BlockTimer:
    """A context manager for measuring the execution time of a code block

    For example::

        >>> with BlockTimer("foo"):
        ...     time.sleep(1)
        ...
        Execution of foo: 1.001s
    """
    def __init__(self, label="block", enabled=True):
        """Create the timer object

        :param str label: A name to identify the block being timed.
        :param bool enabled: Whether or not to enable this timer.
        """
        self._label = label
        self._enabled = enabled

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *_):
        duration = time.perf_counter() - self.start
        if self._enabled:
            logger.info(f"Execution of {self._label}: {duration:.4f}s")


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
        logger.debug(repr(e))
        return GpuInfo()
    except Exception as e:
        # We don't want to prevent the app from starting simply because
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


def run_in_thread(daemon=False):
    """Run a function/method in a thread."""
    def wrap(f):
        @functools.wraps(f)
        def threaded_f(*args, **kwargs):
            t = Thread(target=f, daemon=daemon, args=args, kwargs=kwargs)
            t.start()
            return t
        return threaded_f
    return wrap


def get_available_port(default_port):
    """Find an available port to bind to."""
    port = default_port
    with socket.socket() as s:
        while True:
            try:
                s.bind(("127.0.0.1", port))
            except OSError:
                port += 1
            else:
                break

    return port


#: A helper object to store information for a single series.
#:
#: :param float/np.ndarray data: The series data for the current train.
#: :param str name: The name of the series, to be displayed in a legend (optional).
#: :param float/np.ndarray error: The error of :code:`data`, shown as ± :code:`error` (optional).
#: :param float/np.ndarray error_beam_width: The
Series = namedtuple("Series", ["data", "name", "error", "error_beam_width"], defaults=[None, None, None])


def rich_output(x, xlabel="x", ylabel="y", title=None, max_points=None, **kwargs):
    """
    A helper function to wrap scalars and arrays in :code:`DataArray`'s, with
    metadata for plotting.

    There are two ways of using this function:

    1. If a single value is passed, e.g. :code:`rich_output(scalar1)` or
       :code:`rich_output(vector1)`, that value will be treated as data for the Y
       axis and the X axis coordinates will automatically be generated.
    2. If other Y axis data is passed with keyword arguments, then the first
       (positional) argument will be treated as the X axis coordinates and the Y
       axis data will be plotted against it. For example,
       :code:`rich_output(scalar1, y1=y1_scalar, y2=y2_scalar)` or
       :code:`rich_output(vector1, y1=y1_vector)`.

    Examples:

    .. code-block:: python3

       from extra_foam.utils import rich_output, Series as S

       # Single series
       rich_output(42)

       # Multiple series
       rich_output(42, y1=2.81, y2=3.14)

       # Multiple series with all the metadata
       rich_output(42, y1=S(2.81, name="e", error=0.1), y2=S(3.14, "Pi", 0.2),
                   title="Foo",
                   xlabel="Bar",
                   ylabel="Baz",
                   max_points=100)

    :param float/np.ndarray x: Only required argument, treated as either an X or
                               Y coordinate depending on whether any other
                               series are passed.
    :param str xlabel:         Label for the X axis (optional).
    :param str ylabel:         Label for the Y axis (optional).
    :param str title:          Plot title. (optional)
    :param int max_points:     Maximum number of points to display on the plot (optional).
    :param dict kwargs:        Extra keyword arguments, these can be used for
                               plotting multiple series. The rule is that any
                               keyword argument :code:`y{digits}` is treated as a
                               series. The value can either be a scalar/ndarray
                               (in which case the series label will be the
                               keyword argument name), or a :code:`Series` object. For
                               example, :code:`y1=42` will create a series with the
                               name 'y1', and :code:`y42=Series(1, name='foo')` will
                               create a series named 'foo' (optional).
    :return:                   A :code:`DataArray` containing the data and
                               metadata for plotting.
    """
    xr_attrs = {
        "xlabel": xlabel,
        "ylabel": ylabel,
        "y_series_labels": [],
        "series_errors": { },
        "series_errors_beam_widths": { }
    }

    # Copy optional arguments without a default value
    if title is not None:
        xr_attrs["title"] = title
    if max_points is not None:
        xr_attrs["max_points"] = max_points

    full_data = [x]
    y_series_labels = xr_attrs["y_series_labels"]
    for key, data in kwargs.items():
        if re.fullmatch(r"y(\d+)?", key):
            if isinstance(data, Series):
                label = data.name if data.name else key
                y_series_labels.append(label)
                full_data.append(data.data)

                if data.error is not None:
                    xr_attrs["series_errors"][label] = data.error
                if data.error_beam_width is not None:
                    xr_attrs["series_errors_beam_widths"][label] = data.error_beam_width
            else:
                y_series_labels.append(key)
                full_data.append(data)

    return xr.DataArray(full_data, attrs=xr_attrs)

@dataclass
class RectROI():
    x: int = 0
    y: int = 0
    width: int = 100
    height: int = 100

    def of(self, data):
        if not isinstance(data, np.ndarray):
            raise ValueError(f"ROI input is a {type(data)} instead of an np.ndarray")
        elif data.ndim != 2:
            raise ValueError(f"ROI input must be 2D, but is actually: {data.ndim}D")

        return data[self.y:self.y + self.height, self.x:self.x + self.width]


@dataclass
class LinearROI():
    lower_bound: float = 0
    upper_bound: float = 10

    def of(self, data):
        if not isinstance(data, np.ndarray):
            raise ValueError(f"ROI input is a {type(data)} instead of an np.ndarray")
        elif data.ndim != 1:
            raise ValueError(f"ROI input must be 1D, but is actually: {data.ndim}D")

        return data[lower_bound:upper_bound]

    def slice(self):
        return slice(round(self.lower_bound), round(self.upper_bound))

class AnnotatedView(View, abstract=True):
    def __init__(self, *args, annotations=None, aspect_unlocked=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.annotations = annotations
        self.aspect_unlocked = aspect_unlocked

class ImageWithROIsView(AnnotatedView, output=ViewOutput.IMAGE):
    def __init__(self, *args, rois=None, **kwargs):
        super().__init__(*args, annotations=rois, **kwargs)

class VectorWithROIsView(AnnotatedView, output=ViewOutput.VECTOR):
    def __init__(self, *args, rois=None, **kwargs):
        super().__init__(*args, annotations=rois, **kwargs)

class ImageAspectUnlockedView(AnnotatedView, output=ViewOutput.IMAGE):
    def __init__(self, *args, rois=None, **kwargs):
        super().__init__(*args, aspect_unlocked=True, annotations=rois, **kwargs)

ViewDecorator.kwargs_symbols.update(WithROIs=dict(view_impl=ImageWithROIsView))
ViewDecorator.kwargs_symbols.update(WithROIs=dict(view_impl=VectorWithROIsView))
ViewDecorator.kwargs_symbols.update(AspectUnlocked=dict(view_impl=ImageAspectUnlockedView))
