"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from collections import deque, OrderedDict
from queue import Empty, Full
from threading import Lock

from .data_model import ProcessedData
from ..ipc import process_logger as logger
from ..config import config


class SimpleQueue:
    """A thread-safe queue for passing data fast between threads.

    It does not provide the functionality of coordination among threads
    as threading.Queue, but is way more faster.
    """
    def __init__(self, maxsize=0):
        """Initialization.

        :param int maxsize: if maxsize is <= 0, the queue size is infinite.
        """
        super().__init__()

        self._queue = deque()
        self._maxsize = maxsize
        self._mutex = Lock()

    def get_nowait(self):
        """Pop an item from the queue without blocking."""
        return self.get()

    def get(self):
        with self._mutex:
            if len(self._queue) > 0:
                return self._queue.popleft()
            raise Empty

    def put_nowait(self, item):
        """Put an item into the queue without blocking."""
        self.put(item)

    def put(self, item):
        with self._mutex:
            if 0 < self._maxsize <= len(self._queue):
                raise Full
            self._queue.append(item)

    def put_pop(self, item):
        with self._mutex:
            if 0 < self._maxsize < len(self._queue):
                self._queue.popleft()
            self._queue.append(item)

    def qsize(self):
        with self._mutex:
            return len(self._queue)

    def empty(self):
        with self._mutex:
            return not len(self._queue)

    def full(self):
        with self._mutex:
            return 0 < self._maxsize <= len(self._queue)

    def clear(self):
        with self._mutex:
            self._queue.clear()


class CorrelateQueue(SimpleQueue):
    """CorrelateQueue class.

    A thread-safe queue which correlates data with the same train ID, i.e.,
    one can pop the data out of the queue only if all required data items
    are correlated.

    It has the same interface as the Python internal threading.Queue
    """
    _cache_size = config["CORRELATION_QUEUE_CACHE_SIZE"]

    def __init__(self, catalog, maxsize=0):
        """Initialization.

        :param SourceCatalog catalog: data source catalog.
        :param int maxsize: max number of items allowed in the queue. If
            it is less than or equal to zero, the size is infinite.
        """
        super().__init__(maxsize)

        self._catalog = catalog

        self._cached = OrderedDict()

        # keep the latest correlated data and tid
        self._correlated = None
        self._correlated_tid = -1

    def put(self, item, again=False):
        """Queue interface.

        :param dict item: data after being transformed by DataTransformer.
            It should have keys "meta", "raw" and "catalog" according to
            the protocol.
        :param bool again: whether this item has been tried to put into
            the queue before.
        """
        def _found_all(catalog, meta):
            for k in catalog:
                if k not in meta:
                    return False
            return True

        new_meta, new_raw = item['meta'], item['raw']
        if len(new_meta) == 0:
            return

        catalog = self._catalog

        tid = next(iter(new_meta.values()))["tid"]
        if tid > self._correlated_tid:
            # update cached data
            cached = self._cached.setdefault(
                tid, {'meta': dict(), 'raw': dict()})

            cached_meta = cached['meta']
            cached_raw = cached['raw']

            cached_meta.update(new_meta)
            cached_raw.update(new_raw)

            if _found_all(catalog, cached_meta):
                self._correlated = {
                    'catalog': catalog.__copy__(),
                    'meta': cached_meta,
                    'raw': cached_raw,
                    'processed': ProcessedData(tid)
                }
                self._correlated_tid = tid

                while True:
                    # delete old data
                    key, _ = self._cached.popitem(last=False)
                    if key == tid:
                        break
        else:
            if not again:
                logger.warning(f"Train ID of the new item: {tid} is smaller "
                               f"than the previous correlated train ID "
                               f"{self._correlated_tid}")

        if self._correlated is not None:
            # just correlated or the following line raises Full
            super().put(self._correlated)
            self._correlated = None

        if len(self._cached) > self._cache_size:
            k, v = self._cached.popitem(last=False)
            msg = f"Failed to correlate all the source items for train {k}! "
            logger.warning(msg + f"{len(v['meta'])} out of "
                                 f"{len(catalog)} are available.")

    def put_nowait(self, item, again=False):
        self.put(item, again=again)

    def clear(self):
        """Override."""
        self._cached.clear()
        self._correlated = None
        self._correlated_tid = -1
        super().clear()
