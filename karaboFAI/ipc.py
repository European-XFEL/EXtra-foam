"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

Network communication.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import json

import redis

from .config import config
from .utils import _MetaSingleton


class _RedisQueueBase:
    def __init__(self, namespace, **kwargs):
        self._key = namespace
        self._redis = redis.Redis(**kwargs)


class RQProducer(_RedisQueueBase):
    def __init__(self, namespace):
        super().__init__(namespace)

    def put(self, key, value):
        """Put item into the queue."""
        self._redis.rpush(self._key, json.dumps({key: value}))


class RQConsumer(_RedisQueueBase):
    def __init__(self, namespace):
        super().__init__(namespace)

    def get(self, block=True, timeout=None):
        if block:
            item = self._redis.blpop(self._key, timeout=timeout)
        else:
            item = self._redis.lpop(self._key)

        if item:
            return json.loads(item, encoding='utf8')

    def get_nowait(self):
        return self.get(False)

    def get_all(self):
        msg = dict()
        while True:
            new_msg = self.get_nowait()
            if new_msg is None:
                break
            msg.update(new_msg)

        return msg


_GLOBAL_REDIS_CONNECTION = None
_GLOBAL_REDIS_CONNECTION_BYTES = None


def redis_connection(decode_responses=True):
    """Return a Redis connection."""
    if decode_responses:
        global _GLOBAL_REDIS_CONNECTION

        if _GLOBAL_REDIS_CONNECTION is None:
            connection = redis.Redis('localhost', config['REDIS_PORT'],
                                     password=config['REDIS_PASSWORD'],
                                     decode_responses=True)
            _GLOBAL_REDIS_CONNECTION = connection

        return _GLOBAL_REDIS_CONNECTION

    global _GLOBAL_REDIS_CONNECTION_BYTES

    if _GLOBAL_REDIS_CONNECTION_BYTES is None:
        connection = redis.Redis('localhost', config['REDIS_PORT'],
                                 password=config['REDIS_PASSWORD'])
        _GLOBAL_REDIS_CONNECTION_BYTES = connection

    return _GLOBAL_REDIS_CONNECTION_BYTES


class RedisConnection:
    """Lazily evaluated Redis connection on access."""
    def __init__(self, decode_responses=True):
        self._db = None
        self._decode_responses = decode_responses

    def __get__(self, instance, instance_type):
        if self._db is None:
            self._db = redis_connection(
                decode_responses=self._decode_responses)
        return self._db


class RedisSubscriber:
    """Lazily evaluated Redis subscriber."""
    def __init__(self, channel, decode_responses=True):
        self._sub = None
        self._decode_responses = decode_responses
        self._channel = channel

    def __get__(self, instance, instance_type):
        if self._sub is None:
            self._sub = redis_connection(
                decode_responses=self._decode_responses).pubsub()
            self._sub.subscribe(self._channel)
        return self._sub


class RedisPSubscriber:
    """Lazily evaluated Redis psubscriber."""
    def __init__(self, pattern, decode_responses=True):
        self._sub = None
        self._decode_responses = decode_responses
        self._pattern = pattern

    def __get__(self, instance, instance_type):
        if self._sub is None:
            self._sub = redis_connection(
                decode_responses=self._decode_responses).pubsub()
            self._sub.psubscribe(self._pattern)
        return self._sub


class ProcessWorkerLogger(metaclass=_MetaSingleton):
    """Worker which publishes log message in another Process."""

    _db = RedisConnection()

    def debug(self, msg):
        self._db.publish("log:debug", msg)

    def info(self, msg):
        self._db.publish("log:info", msg)

    def warning(self, msg):
        self._db.publish("log:warning", msg)

    def error(self, msg):
        self._db.publish("log:error", msg)
