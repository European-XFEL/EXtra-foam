"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import time
from itertools import chain

from .base_proxy import _AbstractProxy
from .db_utils import redis_except_handler
from ..config import config


MAX_TRAIN_ID = 999999999
MAX_PERFORMANCE_MONITOR_POINTS = 10 * 60 * 5  # 5 minutes at 10 Hz


class MonProxy(_AbstractProxy):
    """Proxy for adding and retrieving runtime information to redis."""

    LATEST_TID = "mon:latest_tid"
    N_PROCESSED = "mon:n_processed"
    N_DROPPED = "mon:n_dropped"
    N_PROCESSED_P = "mon:n_processed_pulses"
    PERFORMANCE = "mon:train_id"
    TIMESTAMP_SOURCES = "mon:timestamp_sources"
    AVAILABLE_SOURCES = "mon:available_sources"
    MATCHED_SOURCES = "mon:matched_sources"

    @redis_except_handler
    def add_tid_with_timestamp(self, tid, n_pulses, *, dropped=False):
        """Add the current timestamp ranked by the given train ID.

        Also increase the count of # of processed or # of dropped
        depend on the flag.

        :param int n_pulses: number of processed pulses in this train.
        :param bool dropped: whether the train is dropped or not.
        """
        pipe = self._db.pipeline()
        pipe.zadd(self.PERFORMANCE, {time.time(): tid})
        pipe.execute_command("SET", self.LATEST_TID, tid)
        if dropped:
            pipe.execute_command("INCR", self.N_DROPPED)
        else:
            pipe.execute_command("INCR", self.N_PROCESSED).execute_command(
                "INCRBY", self.N_PROCESSED_P, str(n_pulses))
        return pipe.execute()

    @redis_except_handler
    def get_process_count(self):
        """Get the latest train ID and the process counts."""
        return self._db.execute_command("MGET", self.LATEST_TID,
                                                self.N_PROCESSED,
                                                self.N_DROPPED,
                                                self.N_PROCESSED_P)

    @redis_except_handler
    def reset_process_count(self):
        """Set the process counts to zero."""
        pipe = self._db.pipeline()
        return pipe.execute_command(
            "SET", self.N_PROCESSED, 0).execute_command(
            "SET", self.N_DROPPED, 0).execute_command(
            "SET", self.N_PROCESSED_P, 0).execute()

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
            self.PERFORMANCE, MAX_TRAIN_ID, 0,
            start=0, num=num, withscores=True, score_cast_func=int)

    @redis_except_handler
    def get_last_tid(self):
        """Get the largest registered train ID with timestamp.

        :return: (timestamp, tid) or None if no train ID has been registered.
        """
        query = self._db.zrevrangebyscore(
            self.PERFORMANCE, MAX_TRAIN_ID, 0,
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

    @redis_except_handler
    def set_available_sources(self, meta, matched):
        """Set available sources.

        :param dict meta: dictionary of the raw meta data.
        :param list matched: a list of matched source items.

        :return: None if the connection failed;
                 A list of results of 'DEL', 'hmset' and 'PEXPIRE'.
        """
        ts = time.time()
        pipe = self._db.pipeline()
        if meta:
            key_avail = self.AVAILABLE_SOURCES
            pipe.execute_command(
                'HSET',
                key_avail,
                *chain.from_iterable([(k, ts) for k in meta.keys()]))
        if matched:
            key_matched = self.MATCHED_SOURCES
            pipe.execute_command(
                'HSET',
                key_matched,
                *chain.from_iterable([(k, ts) for k in matched]))
        return pipe.execute()

    def get_available_sources(self):
        """Query available sources.

        :return: None if the connection failed;
                 otherwise, a dictionary of source name and train ID pairs.
        """
        key_avail = self.AVAILABLE_SOURCES
        key_matched = self.MATCHED_SOURCES

        # TODO: use Lua script
        pipe = self._db.pipeline()
        pipe.execute_command('HGETALL', key_avail)
        pipe.execute_command('HGETALL', key_matched)
        ret = pipe.execute()

        if ret is not None:
            raw, matched = ret
            expire_t = time.time() - config['SOURCE_EXPIRATION_TIMER']
            raw_avail = []
            raw_expired = []
            for k, v in raw.items():
                if float(v) < expire_t:
                    raw_expired.append(k)
                else:
                    raw_avail.append(k)
            matched_avail = []
            matched_expired = []
            for k, v in matched.items():
                if float(v) < expire_t:
                    matched_expired.append(k)
                else:
                    matched_avail.append(k)

            if raw_expired:
                pipe.execute_command('HDEL', key_avail, *raw_expired)
            if matched_expired:
                pipe.execute_command('HDEL', key_matched, *matched_expired)
            pipe.execute()

            return sorted(raw_avail), sorted(matched_avail)
