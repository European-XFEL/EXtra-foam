"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import functools

import redis


def redis_except_handler(func):
    """Handler ConnectionError from Redis."""
    @functools.wraps(func)
    def catched_f(*args):
        try:
            return func(*args)
        except redis.exceptions.ConnectionError:
            # return None if ConnectionError was raised
            return
    return catched_f
