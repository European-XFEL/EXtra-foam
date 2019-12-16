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

from karabo_bridge import Client

from ..config import config, DataSource
from ..utils import profiler
from ..ipc import process_logger as logger
from ..database import MetaProxy, MonProxy
from ..database import Metadata as mt


class _PipeBase(ABC):
    """Abstract Pipe class.

    Pipe is used to transfer data between different processes via socket,
    file, shared memory, etc. Internally, it stores the data in a
    multi-threading queue and exchanges data within its own process via
    this Queue.
    """

    def __init__(self, *, drop=False, final=False):
        """Initialization.

        :param bool drop: True if data is allowed to be dropped.
        :param bool final: True if the pipe is the final one in the pipeline.
        """
        self._drop = drop
        self._final = final

        self._data = Queue(maxsize=config["MAX_QUEUE_SIZE"])

        self._meta = MetaProxy()
        self._mon = MonProxy()

        self._update_ev = threading.Event()

    def update(self):
        """Signal update."""
        self._update_ev.set()

    def start(self, close_ev):
        """Start to run pipe in a thread.

        For input pipe, it starts to receive data from the client and put it
        into the internal queue; for output pipe, it starts to get data from
        the internal queue and send it to the client.
        """
        # clean the residual data
        self.clean()

        thread = threading.Thread(target=self._work,
                                  args=(close_ev, self._update_ev),
                                  daemon=True)
        thread.start()
        return thread

    @abstractmethod
    def _work(self, close_ev, update_ev):
        """Target function for running in a thread.

        :param multiprocessing.Event close_ev: if this event is set, the
            target function running in a thread will be terminated.
        :param multithreading.Event update_ev: if this event is set, the
            pipe will update its state.
        """
        raise NotImplementedError

    def clean(self):
        """Empty the data queue."""
        while not self._data.empty():
            try:
                self._data.get_nowait()
            except Empty:
                break


class _PipeInBase(_PipeBase):
    """An abstract pipe that receives incoming data."""
    @abstractmethod
    def connect(self, pipe_out):
        """Connect to specified output pipe."""
        pass

    def get(self, timeout=None):
        """Remove and return the first data item in the queue."""
        return self._data.get(timeout=timeout)

    def get_nowait(self):
        return self._data.get_nowait()


class _PipeOutBase(_PipeBase):
    """An abstract pipe that dispatches data outwards."""
    @abstractmethod
    def accept(self, connection):
        """Accept a connection."""
        pass

    def put(self, item, timeout=None):
        """Add a new data item into the queue."""
        return self._data.put(item, timeout=timeout)

    def put_nowait(self, item):
        return self._data.put_nowait(item)

    def put_pop(self, item, timeout=None):
        """Add a new data item into the queue aggressively.

        If the queue is full, the first item will be removed and the
        new data item will be added again without blocking.
        """
        try:
            self._data.put(item, timeout=timeout)
        except Full:
            self._data.get_nowait()
            logger.warning(f"Data dropped due to slowness of the pipeline")
            self._data.put_nowait(item)


class KaraboBridge(_PipeInBase):
    """Karabo bridge client which is an input pipe."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _work(self, close_ev, update_ev):
        """Override."""
        timeout = config['TIMEOUT']

        # the time when the previous data was received
        prev_data_arrival_time = None
        # Karabo bridge client instance
        client = None
        while not close_ev.is_set():
            if update_ev.is_set():
                cfg = self._meta.hget_all(mt.CONNECTION)
                endpoint = cfg['endpoint']
                src_type = DataSource(int(cfg['source_type']))

                if client is None:
                    # destroy the zmq socket
                    del client
                # instantiate a new client
                client = Client(endpoint, timeout=timeout)
                logger.debug(f"Instantiate a bridge client connected to "
                             f"{endpoint}")
                update_ev.clear()

            if client is None:
                time.sleep(0.001)
                continue

            try:
                data = self._recv_imp(client)

                if prev_data_arrival_time is not None:
                    fps = 1.0 / (time.time() - prev_data_arrival_time)
                    logger.debug(f"Bridge recv FPS: {fps:>4.1f} Hz")
                prev_data_arrival_time = time.time()

                # wait until data in the queue has been processed
                # Note: if the queue is full, whether the data should be
                #       dropped is determined by the main thread of its
                #       owner process.
                while not close_ev.is_set():
                    try:
                        self._data.put({"raw": data[0],
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
        """Override."""
        pass


class MpInQueue(_PipeInBase):
    """A pipe which uses a multi-processing queue to receive data."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._client = mp.Queue(maxsize=config["MAX_QUEUE_SIZE"])

    def _work(self, close_ev, update_ev):
        """Override."""
        timeout = config['TIMEOUT']

        while not close_ev.is_set():
            try:
                data = self._client.get(timeout=timeout)
            except Empty:
                continue

            while not close_ev.is_set():
                try:
                    self._data.put(data, timeout=timeout)
                    break
                except Full:
                    continue

        self._client.cancel_join_thread()

    def connect(self, pipe_out):
        """Override."""
        if isinstance(pipe_out, MpOutQueue):
            pipe_out.accept(self._client)
        else:
            raise NotImplementedError(f"Cannot connect {self.__class__} "
                                      f"(input) to {type(pipe_out)} (output)")


class MpOutQueue(_PipeOutBase):
    """A pipe which uses a multi-processing queue to dispatch data."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._client = None

    def _put_pop_queue(self, item, timeout=None):
        """Add a new data item into the multiprocessing queue aggressively.

        If the queue is full, the first item will be removed and the
        new data item will be added again without blocking.
        """
        try:
            self._client.put(item, timeout=timeout)
        except Full:
            self._client.get_nowait()
            self._client.put_nowait(item)

    def _work(self, close_ev, update_ev):
        """Override."""
        timeout = config['TIMEOUT']

        while not close_ev.is_set():
            try:
                data = self._data.get(timeout=timeout)
            except Empty:
                continue

            if self._final:
                self._mon.add_tid_with_timestamp(data['processed'].tid)

            if self._drop:
                data_out = data['processed']
                try:
                    self._put_pop_queue(data_out, timeout=timeout)
                except Empty:
                    continue
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

    def accept(self, connection):
        """Override."""
        self._client = connection
