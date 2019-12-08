"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from abc import ABC, abstractmethod
import multiprocessing as mp
import threading
from queue import Empty, Full, Queue
import time

import karabo_bridge as kb

from ..config import config, DataSource
from ..utils import profiler
from ..ipc import ProcessWorkerLogger
from ..database import MetaProxy
from ..database import Metadata as mt


class Pipe(ABC):
    """Abstract Pipe class.

    Pipe is used to transfer data between different processes via socket,
    file, shared memory, etc. Internally, it stores the data in a
    multi-threading queue and exchanges data with the owner process via
    this Queue.
    """

    def __init__(self, *, gui=False):
        """Initialization.

        :param bool gui: True for connecting to GUI and False for not.
        """
        self._gui = gui

        self._queue = Queue(maxsize=config["MAX_QUEUE_SIZE"])

        self.log = ProcessWorkerLogger()

        self._meta = MetaProxy()

        self._update_ev = threading.Event()

    def update(self):
        """Signal update."""
        self._update_ev.set()

    def run_in_thread(self, close_ev):
        """Run pipe in a thread.

        For input pipe, it starts to receive data from the client and put it
        into the internal queue; for output pipe, it starts to get data from
        the internal queue and send it to the client.
        """
        # clean the residual data
        self.clean()

        thread = threading.Thread(target=self.work,
                                  args=(close_ev, self._update_ev),
                                  daemon=True)
        thread.start()
        return thread

    @abstractmethod
    def work(self, close_ev, update_ev):
        """Target function for running in a thread.

        :param multiprocessing.Event close_ev: if this event is set, the
            target function running in a thread will be terminated.
        :param multithreading.Event update_ev: if this event is set, the
            pipe will update its status.
        """
        pass

    def clean(self):
        """Empty the data queue."""
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except Empty:
                break


class PipeIn(Pipe):
    """A pipe that receives incoming data."""
    def get(self, timeout=None):
        """Remove and return the first data item in the queue."""
        return self._queue.get(timeout=timeout)

    def get_nowait(self):
        return self._queue.get_nowait()

    @abstractmethod
    def connect(self, pipe_out):
        """Connect to an output Pipe."""
        pass


class PipeOut(Pipe):
    """A pipe that dispatches data outwards."""
    def put(self, item, timeout=None):
        """Add a new data item into the queue."""
        return self._queue.put(item, timeout=timeout)

    def put_nowait(self, item):
        return self._queue.put_nowait(item)

    def put_pop(self, item, timeout=None):
        """Add a new data item into the queue aggressively.

        If the queue is full, the first item will be removed and the
        new data item will be added again without blocking.
        """
        try:
            self._queue.put(item, timeout=timeout)
        except Full:
            self._queue.get_nowait()
            self.log.warning(f"Data dropped in {self} due to "
                             f"slowness of the pipeline")
            self._queue.put_nowait(item)


class KaraboBridgePipeIn(PipeIn):
    """Input pipe which uses a Karabo bridge client to receive data."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._client = None  # Karabo bridge client instance

    def work(self, close_ev, update_ev):
        """Override."""
        timeout = config['TIMEOUT']

        # the time when the previous data was received
        prev_data_arrival_time = None

        while not close_ev.is_set():
            if update_ev.is_set():
                cfg = self._meta.hget_all(mt.CONNECTION)
                endpoint = cfg['endpoint']
                src_type = DataSource(int(cfg['source_type']))

                self.connect(kb.Client(endpoint, timeout=timeout))
                self.log.debug(f"Instantiate a bridge client connected to "
                               f"{endpoint}")

                update_ev.clear()

            client = self._client
            if client is None:
                time.sleep(0.001)
                continue

            try:
                data = self._recv_imp(client)

                if prev_data_arrival_time is not None:
                    fps = 1.0 / (time.time() - prev_data_arrival_time)
                    self.log.debug(f"Bridge recv FPS: {fps:>4.1f} Hz")
                prev_data_arrival_time = time.time()

                # wait until data in the queue has been processed
                # Note: if the queue is full, whether the data should be
                #       dropped is determined by the main thread of its
                #       owner process.
                while not close_ev.is_set():
                    try:
                        self._queue.put({"raw": data[0],
                                         "meta": data[1],
                                         "source_type": src_type,
                                         "processed": None},
                                        timeout=timeout)
                        break
                    except Full:
                        continue

            except TimeoutError:
                pass

    @profiler("Receive Data from Bridge")
    def _recv_imp(self, client):
        return client.next()

    def connect(self, pipe_out):
        if isinstance(pipe_out, kb.Client):
            if self._client is not None:
                del self._client  # destroy the zmq socket
            self._client = pipe_out
        else:
            raise TypeError(f"{self.__class__} can only connect to "
                            f"{kb.Client}!")


class MpQueuePipeIn(PipeIn):
    """Input pipe which uses a multi-processing queue to receive data."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._client = mp.Queue(maxsize=config["MAX_QUEUE_SIZE"])

    def work(self, close_ev, update_ev):
        timeout = config['TIMEOUT']

        while not close_ev.is_set():
            try:
                data = self._client.get(timeout=timeout)
            except Empty:
                continue

            while not close_ev.is_set():
                try:
                    self._queue.put(data, timeout=timeout)
                    break
                except Full:
                    continue

        self._client.cancel_join_thread()

    def connect(self, pipe_out):
        if isinstance(pipe_out, MpQueuePipeOut):
            self._client = pipe_out.client
        else:
            raise TypeError(f"{self.__class__} can only connect to "
                            f"{MpQueuePipeOut}!")


class MpQueuePipeOut(PipeOut):
    """Output pipe which uses a multi-processing queue to dispatch data."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._client = mp.Queue(maxsize=config["MAX_QUEUE_SIZE"])

    @property
    def client(self):
        return self._client

    def work(self, close_ev, update_ev):
        timeout = config['TIMEOUT']

        while not close_ev.is_set():
            try:
                data = self._queue.get(timeout=timeout)
            except Empty:
                continue

            if self._gui:
                data_out = data['processed']
            else:
                data_out = {key: data[key] for key
                            in ['processed', 'source_type', 'meta', 'raw']}

            while not close_ev.is_set():
                try:
                    self._client.put(data_out, timeout=timeout)
                    break
                except Full:
                    continue

        self._client.cancel_join_thread()
