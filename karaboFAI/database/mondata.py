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
from .base_proxy import _AbstractProxy
from .db_utils import redis_except_handler
from ..config import config


MAX_TRAIN_ID = 999999999
MAX_PERFORMANCE_MONITOR_POINTS = 10 * 60 * 5  # 5 minutes at 10 Hz


class MonProxy(_AbstractProxy):
    """Proxy for adding and retrieving runtime information to redis."""

    MON_PERFORMANCE = "mon:train_id"
    MON_AVAILABLE_SOURCES = "mon:available_sources"

    @redis_except_handler
    def add_tid_with_timestamp(self, tid):
        """Add the current timestamp ranked by the given train ID."""
        return self._db.zadd(self.MON_PERFORMANCE, {time.time(): tid})

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
            self.MON_PERFORMANCE, MAX_TRAIN_ID, 0,
            start=0, num=num, withscores=True, score_cast_func=int)

    @redis_except_handler
    def get_last_tid(self):
        """Get the largest registered train ID with timestamp.

        :return: (timestamp, tid) or None if no train ID has been registered.
        """
        query = self._db.zrevrangebyscore(
            self.MON_PERFORMANCE, MAX_TRAIN_ID, 0,
            start=0, num=1, withscores=True, score_cast_func=int)
        if query:
            return query[0]

    def get_processor_params(self, proc):
        """Query the metadata for a given processor.

        :return: None if the connection failed;
                 otherwise, a dictionary of key-value pairs. If the hash
                 does not exist, an empty dictionary will be returned.
        """
        return self.hget_all(f"meta:proc:{proc}")

    def set_available_sources(self, mapping):
        """Set available sources.

        :return: None if the connection failed;
                 A list of results of 'DEL', 'hmset' and 'PEXPIRE'.
        """
        # TODO: this does not work if we want to match data after processing
        pipe = self._db.pipeline()
        key = self.MON_AVAILABLE_SOURCES
        pipe.execute_command('DEL', key)
        pipe.hmset(key, mapping)
        # key expiration time should be longer than sources update interval
        pipe.execute_command('PEXPIRE', key, config['SOURCES_EXPIRATION_TIME'])
        return pipe.execute()

    def get_available_sources(self):
        """Query available sources.

        :return: None if the connection failed;
                 otherwise, a dictionary of source name and train ID pairs.
        """
        return self.hget_all(self.MON_AVAILABLE_SOURCES)
