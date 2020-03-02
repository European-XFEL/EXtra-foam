"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import functools

import redis


def redis_except_handler(func):
    """Handler ConnectionError from Redis."""
    @functools.wraps(func)
    def catched_f(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except redis.exceptions.ConnectionError:
            # return None if ConnectionError was raised
            return
    return catched_f
