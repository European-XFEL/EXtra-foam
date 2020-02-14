"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from queue import Queue

from .data_model import ProcessedData
from ..ipc import process_logger as logger


class CorrelateQueue:
    """CorrelateQueue class.

    A thread-safe queue which correlates data with the same train ID, i.e.,
    one can pop the data out of the queue only if all required data items
    are correlated.

    It has the same interface as the Python internal threading.Queue
    """
    def __init__(self, maxsize=0):
        """Initialization.

        :param int maxsize: max number of items allowed in the queue. If
            it is less than or equal to zero, the size is infinite.
        """
        super().__init__()

        self._queue = Queue(maxsize=maxsize)

        # keep the latest correlated data and tid
        self._correlated = None
        self._correlated_tid = -1

    def put(self, item, block=True, timeout=None, again=False):
        """Queue interface.

        :param dict item: data after being transformed by DataTransformer.
            It should have keys "meta", "raw" and "catalog" according to
            the protocol.
        :param bool again: whether this item has been tried to put into
            the queue before.
        """
        metadata = item['meta']
        if len(metadata) == 0:
            return

        tid = next(iter(metadata.values()))["tid"]

        if tid > self._correlated_tid:
            # new train received
            item["processed"] = ProcessedData(tid)
            self._correlated = item
            self._correlated_tid = tid
        else:
            if not again:
                logger.warning(f"Train ID of the new item: {tid} is smaller "
                               f"than the previous correlated train ID "
                               f"{self._correlated_tid}")

        if self._correlated is not None:
            # just correlated or the following line raises Full
            self._queue.put(self._correlated, block=block, timeout=timeout)
            self._correlated = None

    def put_nowait(self, item):
        """Queue interface."""
        self.put(item, block=False)

    def get(self, block=True, timeout=None):
        """Queue interface."""
        return self._queue.get(block=block, timeout=timeout)

    def get_nowait(self):
        """Queue interface."""
        return self.get(block=False)

    def qsize(self):
        """Queue interface."""
        return self._queue.qsize()

    def empty(self):
        """Queue interface."""
        return self._queue.empty()

    def full(self):
        """Queue interface."""
        return self._queue.full()

    def task_done(self):
        """Queue interface."""
        self._queue.task_done()

    def join(self):
        """Queue interface."""
        self._queue.join()
