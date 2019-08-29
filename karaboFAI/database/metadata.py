"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

Proxy for metadata which is stored in Redis.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import functools

import redis

from karaboFAI.ipc import RedisConnection


class MetaMetadata(type):
    def __new__(mcs, name, bases, class_dict):
        proc_list = []
        for k, v in class_dict.items():
            if isinstance(v, str):
                s = v.split(":")
                if len(s) == 3 and s[1] == 'proc':
                    proc_list.append(s[2])

        class_dict['processors'] = proc_list
        cls = type.__new__(mcs, name, bases, class_dict)
        return cls


class Metadata(metaclass=MetaMetadata):

    SESSION = "meta:session"

    DATA_SOURCE = "meta:source"

    ANALYSIS_TYPE = "meta:analysis_type"

    GLOBAL_PROC = "meta:proc:global"
    IMAGE_PROC = "meta:proc:image"
    GEOMETRY_PROC = "meta:proc:geometry"
    AZIMUTHAL_INTEG_PROC = "meta:proc:azimuthal_integration"
    PUMP_PROBE_PROC = "meta:proc:pump_probe"
    ROI_PROC = "meta:proc:roi"
    XAS_PROC = "meta:proc:xas"
    CORRELATION_PROC = "meta:proc:correlation"
    BIN_PROC = "meta:proc:bin"
    STATISTICS_PROC = "meta:proc:pulse_fom"
    DATA_REDUCTION_PROC = "meta:proc:data_reduction"
    DARK_RUN_PROC = "meta:proc:dark_run"


def redis_except_handler(return_value):
    def wrap(f):
        @functools.wraps(f)
        def catched_f(*args, **kwargs):
            try:
                return f(*args, **kwargs)
            except redis.exceptions.ConnectionError:
                return return_value
        return catched_f
    return wrap


class MetaProxy:
    _db = RedisConnection()

    def reset(self):
        self._db = None

    @redis_except_handler(-1)
    def set(self, name, key, value):
        """Set a Hash.

        :returns: -1 if connection fails;
                   1 if created a new field;
                   0 if set on a new field.
        """
        return self._db.hset(name, key, value)

    @redis_except_handler(-1)
    def mset(self, name, mapping):
        return self._db.hmset(name, mapping)

    @redis_except_handler(None)
    def get(self, name, key):
        return self._db.hget(name, key)

    @redis_except_handler(None)
    def mget(self, name, keys):
        return self._db.hmget(name, keys)

    @redis_except_handler(None)
    def delete(self, name, key):
        return self._db.hdel(name, key)

    @redis_except_handler(None)
    def get_all(self, name):
        return self._db.hgetall(name)

    @redis_except_handler(None)
    def increase_by(self, name, key, amount=1):
        return self._db.hincrby(name, key, amount)

    @redis_except_handler(None)
    def increase_by_float(self, name, key, amount=1):
        return self._db.hincrbyfloat(name, key, amount)
