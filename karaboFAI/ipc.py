"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

Network communication.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import weakref

import json
import numpy as np

import redis

from .config import config
from .utils import _MetaSingleton
from .serialization import deserialize_image, serialize_image


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


# keep track lazily created connections
_global_connections = dict()


def reset_redis_connections():
    global _GLOBAL_REDIS_CONNECTION
    _GLOBAL_REDIS_CONNECTION = None
    global _GLOBAL_REDIS_CONNECTION_BYTES
    _GLOBAL_REDIS_CONNECTION_BYTES = None

    for connections in _global_connections.values():
        for ref in connections:
            c = ref()
            if c is not None:
                c.reset()


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
    """Lazily evaluated Redis subscriber."""
    def __init__(self, channel, decode_responses=True):
        self._sub = None
        self._decode_responses = decode_responses
        self._channel = channel

    def __get__(self, instance, instance_type):
        if self._sub is None:
            self._sub = redis_connection(
                decode_responses=self._decode_responses).pubsub()
            try:
                self._sub.subscribe(self._channel)
            except redis.ConnectionError:
                self._sub = None

        return self._sub

    def reset(self):
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
                decode_responses=self._decode_responses).pubsub()
            try:
                self._sub.psubscribe(self._pattern)
            except redis.ConnectionError:
                self._sub = None

        return self._sub

    def reset(self):
        self._sub = None


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


class CommandProxy:
    _db = RedisConnection()
    _ref_sub = RedisSubscriber("command:reference_image",
                               decode_responses=False)
    _mask_sub = RedisSubscriber("command:image_mask",
                                decode_responses=False)

    def set_ref_image(self, image):
        """Publish the reference image in Redis."""
        if image is None:
            return

        self._db.publish("command:reference_image", 'next')
        self._db.publish("command:reference_image", serialize_image(image))

    def remove_ref_image(self):
        """Notify to remove the current reference image."""
        self._db.publish("command:reference_image", 'remove')

    def get_ref_image(self):
        """Try to get the reference image.

        :return: None for no update; numpy.ndarray for receiving a new
            reference image; -1 for removing the current reference image.
        """
        sub = self._ref_sub
        if sub is None:
            return

        ref = None
        # process all messages related to reference
        while True:
            msg = sub.get_message(ignore_subscribe_messages=True)
            if msg is None:
                # the channel is empty
                break

            action = msg['data']
            if action == b'next':
                ref = deserialize_image(sub.get_message()['data'])
            else:
                # remove reference
                ref = -1

        return ref

    def add_mask(self, mask_region):
        self._db.publish("command:image_mask", 'add')
        self._db.publish("command:image_mask", str(mask_region))

    def remove_mask(self, mask_region):
        self._db.publish("command:image_mask", 'remove')
        self._db.publish("command:image_mask", str(mask_region))

    def set_mask(self, mask):
        """Publish the image mask in Redis."""
        self._db.publish("command:image_mask", 'set')
        self._db.publish("command:image_mask",
                          serialize_image(mask, is_mask=True))

    def clear_mask(self):
        """Notify to completely clear all the image mask."""
        self._db.publish("command:image_mask", 'clear')

    def update_mask(self, mask, shape):
        """Parse all masking operations.

        :param numpy.ndarray mask: image mask. dtype = np.bool.
        :param tuple/list shape: shape of the image.

        :return: a list of masking operations.
        """
        sub = self._mask_sub
        if sub is None:
            return

        # process all messages related to mask
        while True:
            msg = sub.get_message(ignore_subscribe_messages=True)
            if msg is None:
                break

            action = msg['data']
            if action == b'set':
                mask = deserialize_image(sub.get_message()['data'], is_mask=True)
            elif action in [b'add', b'remove']:
                if mask is None:
                    mask = np.zeros(shape, dtype=np.bool)

                data = sub.get_message()['data'].decode("utf-8")
                x, y, w, h = [int(v) for v in data[1:-1].split(',')]
                if action == b'add':
                    mask[y:y+h, x:x+w] = True
                else:
                    mask[y:y+h, x:x+w] = False
            else:  # data == 'clear'
                mask = None

        return mask
