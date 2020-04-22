"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from collections import deque
from queue import Empty, Full
from threading import Lock


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
