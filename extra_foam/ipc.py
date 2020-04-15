"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import weakref

import json
import numpy as np

import redis

from .config import config
from .serialization import deserialize_image, serialize_image
from .file_io import read_image, read_numpy_array


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


# keep tracking lazily created connections
_global_connections = dict()


def init_redis_connection(host, port, *, password=None):
    """Initialize Redis client connection.

    :param str host: IP address of the Redis server.
    :param int port:: Port of the Redis server.
    :param str password: password for the Redis server.

    :return: Redis connection.
    """
    # reset all connections first
    global _GLOBAL_REDIS_CONNECTION
    _GLOBAL_REDIS_CONNECTION = None
    global _GLOBAL_REDIS_CONNECTION_BYTES
    _GLOBAL_REDIS_CONNECTION_BYTES = None

    for connections in _global_connections.values():
        for ref in connections:
            c = ref()
            if c is not None:
                c.reset()

    # initialize new connection
    if config["REDIS_UNIX_DOMAIN_SOCKET_PATH"]:
        raise NotImplementedError(
            "Unix domain socket connection is not supported!")
        # connection = redis.Redis(
        #     unix_socket_path=config["REDIS_UNIX_DOMAIN_SOCKET_PATH"],
        #     decode_responses=decode_responses
        # )
    else:
        # the following two must have different pools
        connection = redis.Redis(
            host, port, password=password, decode_responses=True)
        connection_byte = redis.Redis(
            host, port, password=password, decode_responses=False)

    _GLOBAL_REDIS_CONNECTION = connection
    _GLOBAL_REDIS_CONNECTION_BYTES = connection_byte
    return connection


def redis_connection(decode_responses=True):
    """Return a Redis connection."""
    if decode_responses:
        return _GLOBAL_REDIS_CONNECTION
    return _GLOBAL_REDIS_CONNECTION_BYTES


class MetaRedisConnection(type):
    def __call__(cls, *args, **kw):
        instance = super().__call__(*args, **kw)
        name = cls.__name__
        if name not in _global_connections:
            _global_connections[name] = []
        _global_connections[name].append(weakref.ref(instance))
        return instance


class RedisConnection(metaclass=MetaRedisConnection):
    """Lazily evaluated Redis connection on access."""
    def __init__(self, decode_responses=True):
        self._db = None
        self._decode_responses = decode_responses

    def __get__(self, instance, instance_type):
        if self._db is None:
            self._db = redis_connection(
                decode_responses=self._decode_responses)
        return self._db

    def reset(self):
        self._db = None


class RedisSubscriber(metaclass=MetaRedisConnection):
    """Lazily evaluated Redis subscriber.

    Read the code of pub/sub in Redis:
        https://making.pusher.com/redis-pubsub-under-the-hood/
    """
    def __init__(self, channel, decode_responses=True):
        self._sub = None
        self._decode_responses = decode_responses
        self._channel = channel

    def __get__(self, instance, instance_type):
        if self._sub is None:
            self._sub = redis_connection(
                decode_responses=self._decode_responses).pubsub(
                ignore_subscribe_messages=True)
            try:
                self._sub.subscribe(self._channel)
            except redis.ConnectionError:
                self._sub = None

        return self._sub

    def reset(self):
        if self._sub is not None:
            self._sub.close()
        self._sub = None


class RedisPSubscriber(metaclass=MetaRedisConnection):
    """Lazily evaluated Redis psubscriber."""
    def __init__(self, pattern, decode_responses=True):
        self._sub = None
        self._decode_responses = decode_responses
        self._pattern = pattern

    def __get__(self, instance, instance_type):
        if self._sub is None:
            self._sub = redis_connection(
                decode_responses=self._decode_responses).pubsub(
                ignore_subscribe_messages=True)
            try:
                self._sub.psubscribe(self._pattern)
            except redis.ConnectionError:
                self._sub = None

        return self._sub

    def reset(self):
        if self._sub is not None:
            self._sub.close()
        self._sub = None


class ProcessLogger:
    """Worker which publishes log message in another Process.

    Note: remember to change other part of the code if the log pattern
    changes.
    """

    _db = RedisConnection()

    def debug(self, msg):
        self._db.publish("log:debug", msg)

    def info(self, msg):
        self._db.publish("log:info", msg)

    def warning(self, msg):
        self._db.publish("log:warning", msg)

    def error(self, msg):
        self._db.publish("log:error", msg)


process_logger = ProcessLogger()


class ReferencePub:
    _db = RedisConnection()

    def set(self, filepath):
        """Publish the reference image filepath in Redis."""
        self._db.publish("reference_image", filepath)


class ReferenceSub:
    _sub = RedisSubscriber("reference_image")

    def update(self):
        """Parse all reference image operations."""
        sub = self._sub
        updated = False
        ref = None
        while True:
            msg = sub.get_message(ignore_subscribe_messages=True)
            if msg is None:
                break

            v = msg['data']
            if not v:
                ref = None
            else:
                ref = read_image(v)
            updated = True
        return updated, ref


class ImageMaskPub:
    _db = RedisConnection()

    def draw(self, mask_region):
        """Add a region to the current mask."""
        self._db.publish("image_mask:draw", str(mask_region))

    def erase(self, mask_region):
        """Erase a region from the current mask."""
        self._db.publish("image_mask:erase", str(mask_region))

    def set(self, mask):
        """Set the whole mask."""
        self._db.publish("image_mask:set",
                         serialize_image(mask, is_mask=True))

    def remove(self):
        """Completely remove all the mask."""
        self._db.publish("image_mask:remove", '')


class ImageMaskSub:
    _sub = RedisPSubscriber("image_mask:*", decode_responses=False)

    def update(self, mask, shape):
        """Parse all masking operations.

        :param numpy.ndarray mask: image mask. dtype = np.bool.
        :param tuple/list shape: shape of the image.

        :return numpy.ndarray: the updated mask.
        """
        sub = self._sub
        updated = False

        if mask is None:
            mask = np.zeros(shape, dtype=np.bool)

        # process all messages related to mask
        while True:
            msg = sub.get_message(ignore_subscribe_messages=True)
            if msg is None:
                break

            topic = msg['channel'].decode("utf-8").split(":")[-1]
            data = msg['data']
            if topic == 'set':
                mask = deserialize_image(data, is_mask=True)
            elif topic in ['draw', 'erase']:
                x, y, w, h = [int(v) for v
                              in data.decode("utf-8")[1:-1].split(',')]

                if topic == 'draw':
                    mask[y:y+h, x:x+w] = True
                else:
                    mask[y:y+h, x:x+w] = False
            else:  # data == 'remove'
                mask.fill(False)

            updated = True

        return updated, mask


class CalConstantsPub:
    _db = RedisConnection()

    def set_gain(self, filepath):
        """Publish the gain constants filepath in Redis.

        ：param str filepath: path of the gain constants file.
        """
        self._db.publish("cal_constants:gain", filepath)

    def set_offset(self, filepath):
        """Publish the offset constants filepath in Redis.

        ：param str filepath: path of the offset constants file.
        """
        self._db.publish("cal_constants:offset", filepath)


class CalConstantsSub:
    _sub = RedisPSubscriber("cal_constants:*")

    def update(self):
        """Parse all cal constants operations."""
        sub = self._sub
        gain = None
        gain_updated = False
        gain_fp = None
        offset = None
        offset_updated = False
        offset_fp = None
        while True:
            msg = sub.get_message(ignore_subscribe_messages=True)
            if msg is None:
                break

            topic = msg['channel'].split(":")[-1]
            v = msg['data']
            if topic == 'gain':
                if not v:
                    gain = None
                    gain_updated = True
                else:
                    gain_fp = v
            elif topic == 'offset':
                if not v:
                    offset = None
                    offset_updated = True
                else:
                    offset_fp = v

        if gain_fp is not None:
            gain = read_numpy_array(gain_fp, dimensions=(2, 3))
            gain_updated = True
        if offset_fp is not None:
            offset = read_numpy_array(offset_fp, dimensions=(2, 3))
            offset_updated = True

        return gain_updated, gain, offset_updated, offset
