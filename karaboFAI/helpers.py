"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

Helper functions.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import functools
import time


def profiler(info):
    def wrap(f):
        @functools.wraps(f)
        def timed_f(*args, **kwargs):
            t0 = time.perf_counter()
            result = f(*args, **kwargs)
            t1 = time.perf_counter()
            print(f"Profiler - [{info}]: {1000*(t1 - t0):.3f} ms")
            return result
        return timed_f
    return wrap
