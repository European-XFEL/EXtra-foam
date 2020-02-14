"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from abc import ABC, abstractmethod
import copy
import multiprocessing as mp
import threading
from queue import Empty, Full, Queue
import time

from karabo_bridge import Client
from .f_queue import CorrelateQueue
from .processors.base_processor import _RedisParserMixin
from ..config import config, DataSource
from ..utils import profiler, run_in_thread
from ..ipc import RedisSubscriber
from ..ipc import process_logger as logger
from ..database import (
    DataTransformer, MetaProxy, MonProxy, SourceCatalog, SourceItem
)
from ..database import Metadata as mt


class _ProcessControlMixin:
    # Note: the following mixin methods could be called from a different
    #       process.
    @property
    def closing(self):
        return self._close_ev.is_set()

    @property
    def running(self):
        return self._pause_ev.is_set()

    def wait(self):
        self._pause_ev.wait()

    def resume(self):
        self._pause_ev.set()

    def pause(self):
        self._pause_ev.clear()


class _PipeBase(ABC, _ProcessControlMixin):
    """Abstract Pipe class.

    Pipe is used to transfer data between different processes via socket,
    file, shared memory, etc. Internally, it stores the data in a
    multi-threading queue and exchanges data within its own process via
    this Queue.
    """
    _pipeline_dtype = ('catalog', 'meta', 'raw', 'processed')

    def __init__(self, pause_ev, close_ev, *, final=False):
        """Initialization.

        :param bool final: True if the pipe is the final one in the pipeline.
        """
        super().__init__()

        self._pause_ev = pause_ev
        self._close_ev = close_ev
        self._final = final

        self._queue = Queue(maxsize=config["PIPELINE_MAX_QUEUE_SIZE"])

        self._meta = MetaProxy()
        self._mon = MonProxy()

    def start(self):
        """Start to run pipe in a thread.

        For input pipe, it starts to receive data from the client and put it
        into the internal queue; for output pipe, it starts to get data from
        the internal queue and send it to the client.
        """
        # clean the residual data
        self.clean()
        self.pre_run()
        self.run()

    def pre_run(self):
        """"""
        pass

    @abstractmethod
    def run(self):
        """Target function for running in a thread."""
        raise NotImplementedError

    def clean(self):
        """Empty the data queue."""
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
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
        return self._queue.get(timeout=timeout)

    def get_nowait(self):
        return self._queue.get_nowait()


class _PipeOutBase(_PipeBase):
    """An abstract pipe that dispatches data outwards."""
    @abstractmethod
    def accept(self, connection):
        """Accept a connection."""
        pass

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
            self._queue.put_nowait(item)


class KaraboBridge(_PipeInBase, _RedisParserMixin):
    """Karabo bridge client which is an input pipe."""

    _sub = RedisSubscriber(mt.DATA_SOURCE)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._queue = CorrelateQueue(maxsize=config["PIPELINE_MAX_QUEUE_SIZE"])

        self._catalog = SourceCatalog()

        self._lock = threading.Lock()

    def pre_run(self):
        self.update_source_items()

    @run_in_thread(daemon=True)
    def update_source_items(self):
        """Updated requested source items."""
        sub = self._sub
        while True:
            msg = sub.get_message(ignore_subscribe_messages=True)
            if msg is not None:
                src = msg['data']

                item = self._meta.hget_all(src)
                with self._lock:
                    if item:
                        # add a new source item
                        category = item['category']
                        modules = item['modules']
                        slicer = item['slicer']
                        vrange = item['vrange']

                        self._catalog.add_item(SourceItem(
                            category,
                            item['name'],
                            self.str2list(modules, handler=int)
                            if modules else None,
                            item['property'],
                            self.str2slice(slicer) if slicer else None,
                            self.str2tuple(vrange) if vrange else None))
                    else:
                        # remove a source item
                        if src not in self._catalog:
                            # Raised when there were two checked items in
                            # the data source tree with the same "device ID"
                            # and "property". The item has already been
                            # deleted when one of them was unchecked.
                            logger.error("Duplicated data source items")
                            continue
                        self._catalog.remove_item(src)

            time.sleep(0.001)

    @run_in_thread(daemon=True)
    def run(self):
        """Override."""
        timeout = config['PIPELINE_TIMEOUT']

        # the time when the previous data was received
        prev_data_arrival_time = None
        # Karabo bridge client instance
        client = None
        while not self.closing:
            if not self.running:
                self.wait()

                cons = self._meta.hget_all(mt.CONNECTION)
                endpoint = list(cons.keys())[0]
                src_type = DataSource(int(list(cons.values())[0]))

                if client is not None:
                    # destroy the zmq socket
                    del client
                # instantiate a new client
                client = Client(endpoint, timeout=timeout)
                logger.debug(f"Instantiate a bridge client connected to "
                             f"{endpoint}")

            if client is None:
                time.sleep(0.001)
                continue

            try:
                # first make a copy of the source item table
                with self._lock:
                    catalog = copy.deepcopy(self._catalog)

                if not catalog.main_detector:
                    # skip the pipeline if the main detector is not specified
                    logger.error(f"Unspecified {config['DETECTOR']} source!")
                    time.sleep(1)  # wait a bit longer
                    continue

                # receive data from the bridge
                raw, meta = self._recv_imp(client)

                self._update_available_sources(meta)

                # extract new raw and meta
                new_raw, new_meta, _ = DataTransformer.transform_euxfel(
                    raw, meta, catalog=catalog, source_type=src_type)

                if prev_data_arrival_time is not None:
                    fps = 1.0 / (time.time() - prev_data_arrival_time)
                    logger.debug(f"Bridge recv FPS: {fps:>4.1f} Hz")
                prev_data_arrival_time = time.time()

                # wait until data in the queue has been processed
                # Note: if the queue is full, whether the data should be
                #       dropped is determined by the main thread of its
                #       owner process.
                data = {"catalog": catalog, "meta": new_meta, "raw": new_raw}
                again = False
                while not self.closing:
                    try:
                        self._queue.put(data, timeout=timeout, again=again)
                        break
                    except Full:
                        again = True
                        continue

            except TimeoutError:
                pass

    def _update_available_sources(self, meta):
        sources = {k: v["timestamp.tid"] for k, v in meta.items()}
        self._mon.set_available_sources(sources)

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

        self._client = mp.Queue(maxsize=config["PIPELINE_MAX_QUEUE_SIZE"])

    @run_in_thread(daemon=True)
    def run(self):
        """Override."""
        timeout = config['PIPELINE_TIMEOUT']

        while not self.closing:
            if not self.running:
                self.wait()

            try:
                data = self._client.get(timeout=timeout)
            except Empty:
                continue

            while not self.closing:
                try:
                    self._queue.put(data, timeout=timeout)
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

    @run_in_thread(daemon=True)
    def run(self):
        """Override."""
        timeout = config['PIPELINE_TIMEOUT']

        while not self.closing:
            if not self.running:
                self.wait()

            try:
                data = self._queue.get(timeout=timeout)
            except Empty:
                continue

            if self._final:
                data_out = data['processed']

                tid = data_out.tid
                self._mon.add_tid_with_timestamp(tid)
                logger.info(f"Train {tid} processed!")
            else:
                data_out = {key: data[key] for key in self._pipeline_dtype}

            try:
                self._client.put(data_out, timeout=timeout)
            except Full:
                try:
                    self._client.get_nowait()
                except Empty:
                    continue
                self._client.put_nowait(data_out)

        self._client.cancel_join_thread()

    def accept(self, connection):
        """Override."""
        self._client = connection
