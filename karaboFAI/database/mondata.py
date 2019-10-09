"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import time

from .metadata import Metadata as mt
from .db_utils import redis_except_handler
from ..ipc import RedisConnection


MAX_TRAIN_ID = 999999999
MAX_PERFORMANCE_MONITOR_POINTS = 10 * 60 * 5  # 5 minutes at 10 Hz


class MonData:

    PERFORMANCE = "mon:trainId"


class MonProxy:
    """Proxy for adding and retrieving runtime information to redis."""
    _db = RedisConnection()

    def reset(self):
        self.__class__.__dict__["_db"].reset()

    @redis_except_handler
    def add_tid_with_timestamp(self, tid):
        """Add the current timestamp ranked by the given train ID."""
        return self._db.zadd(MonData.PERFORMANCE, {time.time(): tid})

    @redis_except_handler
    def get_latest_tids(self, num=MAX_PERFORMANCE_MONITOR_POINTS):
        """Get a list of latest (timestamp, tid).

        :param int num: maximum length of the returned list.
            Capped by MAX_PERFORMANCE_MONITOR_POINTS.
        """
        if num > MAX_PERFORMANCE_MONITOR_POINTS:
            # performance concern
            num = MAX_PERFORMANCE_MONITOR_POINTS
        return self._db.zrevrangebyscore(
            MonData.PERFORMANCE, MAX_TRAIN_ID, 0,
            start=0, num=num, withscores=True, score_cast_func=int)

    @redis_except_handler
    def get_last_tid(self):
        """Get the largest registered train ID with timestamp.

        :return: (timestamp, tid) or None if no train ID has been registered.
        """
        query = self._db.zrevrangebyscore(
            MonData.PERFORMANCE, MAX_TRAIN_ID, 0,
            start=0, num=1, withscores=True, score_cast_func=int)
        if query:
            return query[0]

    @redis_except_handler
    def get_all_analysis(self):
        """Query all the registered analysis types.

        :return: None if the connection failed;
                 otherwise, a dictionary of key-value pairs (
                 analysis type: number of registrations).
        """
        return self._db.hgetall(mt.ANALYSIS_TYPE)

    @redis_except_handler
    def get_processor_params(self, proc):
        """Query the metadata for a given processor.

        :return: None if the connection failed;
                 otherwise, a dictionary of key-value pairs. If the hash
                 does not exist, an empty dictionary will be returned.
        """
        return self._db.hgetall(f"meta:proc:{proc}")
