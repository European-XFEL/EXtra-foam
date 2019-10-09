"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from .db_utils import redis_except_handler
from ..ipc import RedisConnection


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


class MetaProxy:
    """Proxy for retrieving metadata."""
    _db = RedisConnection()

    def reset(self):
        self.__class__.__dict__["_db"].reset()

    @redis_except_handler
    def set(self, name, key, value):
        """Set a key-value pair of a hash.

        :returns: None if the connection failed;
                  1 if created a new field;
                  0 if set on an old field.
        """
        return self._db.hset(name, key, value)

    @redis_except_handler
    def mset(self, name, mapping):
        """Set a mapping of a hash.

        :return: None if the connection failed;
                 True if set.
        """
        return self._db.hmset(name, mapping)

    @redis_except_handler
    def get(self, name, key):
        """Get the value for a given key of a hash.

        :return: None if the connection failed or key was not found;
                 otherwise, the value.
        """
        return self._db.hget(name, key)

    @redis_except_handler
    def mget(self, name, keys):
        """Get values for a list of keys of a hash.

        :return: None if the connection failed;
                 otherwise, a list of values.
        """
        return self._db.hmget(name, keys)

    @redis_except_handler
    def delete(self, name, key):
        """Delete a key of a hash.

        :return: None if the connection failed;
                 1 if key was found and deleted;
                 0 if key was not found.
        """
        return self._db.hdel(name, key)

    @redis_except_handler
    def get_all(self, name):
        """Get all key-value pairs of a hash.

        :return: None if the connection failed;
                 otherwise, a dictionary of key-value pairs. If the hash
                 does not exist, an empty dictionary will be returned.
        """
        return self._db.hgetall(name)

    @redis_except_handler
    def increase_by(self, name, key, amount=1):
        """Increase the value of a key in a hash by the given amount.

        :return: None if the connection failed;
                 value after the increment if the initial value is an integer;
                 amount if key does not exist (set initial value to 0).

        :raise: redis.exceptions.ResponseError if value is not an integer.
        """
        return self._db.hincrby(name, key, amount)

    @redis_except_handler
    def increase_by_float(self, name, key, amount=1.0):
        """Increase the value of a key in a hash by the given amount.

        :return: None if the connection failed;
                 value after the increment if the initial value can be
                 converted to a float;
                 amount if key does not exist (set initial value to 0).

        :raise: redis.exceptions.ResponseError if value is not a float.
        """
        return self._db.hincrbyfloat(name, key, amount)
