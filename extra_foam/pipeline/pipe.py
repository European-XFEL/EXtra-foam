"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from abc import ABC, abstractmethod
import multiprocessing as mp
from queue import Empty, Full
import time

from .f_zmq import BridgeProxy
from .f_queue import CorrelateQueue, SimpleQueue
from .processors.base_processor import _RedisParserMixin
from ..config import config, DataSource
from ..utils import profiler, run_in_thread
from ..ipc import RedisSubscriber
from ..ipc import process_logger as logger
from ..database import (
    DataTransformer, MetaProxy, MonProxy, SourceCatalog, SourceItem
)
from ..database import Metadata as mt


class _PipeBase(ABC):
    """Abstract Pipe class.

    Pipe is used to transfer data between different processes via socket,
    file, shared memory, etc. Internally, it stores the data in a
    multi-threading queue and exchanges data within its own process via
    this Queue.
    """
    _pipeline_dtype = ('catalog', 'meta', 'raw', 'processed')

    def __init__(self, update_ev, pause_ev, close_ev, *, final=False):
        """Initialization.

        :param bool final: True if the pipe is the final one in the pipeline.
        """
        super().__init__()

        self._update_ev = update_ev
        self._pause_ev = pause_ev
        self._close_ev = close_ev
        self._final = final

        # the queue is not used for queuing data, it serves as a cache here
        self._queue = SimpleQueue(maxsize=1)

        self._meta = MetaProxy()
        self._mon = MonProxy()

    def start(self):
        """Start to run pipe in a thread.

        For input pipe, it starts to receive data from the client and put it
        into the internal queue; for output pipe, it starts to get data from
        the internal queue and send it to the client.
        """
        self.clear()
        self.run()

    @abstractmethod
    def run(self):
        """Target function for running in a thread."""
        raise NotImplementedError

    @property
    def closing(self):
        return self._close_ev.is_set()

    @property
    def running(self):
        return self._pause_ev.is_set()

    @property
    def updating(self):
        return self._update_ev.is_set()

    def finish_updating(self):
        self._update_ev.clear()

    def clear(self):
        self._queue.clear()


class _PipeInBase(_PipeBase):
    """An abstract pipe that receives incoming data."""
    @abstractmethod
    def connect(self, pipe_out):
        """Connect to specified output pipe."""
        pass

    def get(self):
        return self._queue.get_nowait()


class _PipeOutBase(_PipeBase):
    """An abstract pipe that dispatches data outwards."""
    @abstractmethod
    def accept(self, connection):
        """Accept a connection."""
        pass

    def put(self, item):
        self._queue.put_nowait(item)

    def put_pop(self, item):
        self._queue.put_pop(item)


class KaraboBridge(_PipeInBase, _RedisParserMixin):
    """Karabo bridge client which is an input pipe."""

    _sub = RedisSubscriber(mt.DATA_SOURCE)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._catalog = SourceCatalog()

        # override SimpleQueue
        self._queue = CorrelateQueue(self._catalog, maxsize=1)

    def _update_source_items(self):
        """Updated requested source items."""
        sub = self._sub
        while True:
            msg = sub.get_message(ignore_subscribe_messages=True)
            if msg is None:
                break

            src = msg['data']
            item = self._meta.hget_all(src)
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

    def _update_connection(self, proxy):
        cons = self._meta.hget_all(mt.CONNECTION)
        endpoints = list(cons.keys())
        # cannot have different types for different endpoints
        src_type = DataSource(int(list(cons.values())[0]))

        endpoint = endpoints
        proxy.stop()
        proxy.connect(endpoints)
        logger.debug(f"Instantiate a bridge client connected to "
                     f"{endpoint}")
        proxy.start()
        return proxy, src_type

    @run_in_thread(daemon=True)
    def run(self):
        """Override."""
        data_in = None
        again = False
        proxy = BridgeProxy()
        while not self.closing:
            if self.updating:
                client, src_type = self._update_connection(proxy)

                data_in = None
                self.clear()
                self.finish_updating()

            self._update_source_items()

            if self.running and proxy.client is not None:
                if not self._catalog.main_detector:
                    # skip the pipeline if the main detector is not specified
                    logger.error(f"{config['DETECTOR']} source unspecified!")
                    time.sleep(1)  # sleep a little long
                    continue

                if data_in is None:
                    try:
                        # always pull the latest data from the bridge
                        raw, meta = self._recv_imp(proxy.client)
                        self._update_available_sources(meta)

                        # extract new raw and meta
                        new_raw, new_meta, _ = DataTransformer.transform_euxfel(
                            raw, meta, catalog=self._catalog, source_type=src_type)

                        data_in = {"meta": new_meta, "raw": new_raw}
                        again = False
                    except TimeoutError:
                        pass

                if data_in is not None:
                    try:
                        self._queue.put(data_in, again=again)
                        data_in = None
                        again = False
                    except Full:
                        again = True

            time.sleep(0.001)

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
        data_in = None
        while not self.closing:
            if self.updating:
                data_in = None
                self.clear()
                self.finish_updating()

            if data_in is None:
                try:
                    data_in = self._client.get_nowait()
                except Empty:
                    pass

            if data_in is not None:
                try:
                    self._queue.put_nowait(data_in)
                    data_in = None
                except Full:
                    pass

            time.sleep(0.001)

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
        data_out = None
        while not self.closing:
            if self.updating:
                data_out = None
                self.clear()
                self.finish_updating()

            if data_out is None:
                try:
                    data = self._queue.get_nowait()

                    if self._final:
                        data_out = data['processed']

                        tid = data_out.tid
                        self._mon.add_tid_with_timestamp(tid)
                        logger.info(f"Train {tid} processed!")
                    else:
                        data_out = {key: data[key] for key
                                    in self._pipeline_dtype}
                except Empty:
                    pass

            if data_out is not None:
                try:
                    self._client.put_nowait(data_out)
                    data_out = None
                except Full:
                    pass

            time.sleep(0.001)

        self._client.cancel_join_thread()

    def accept(self, connection):
        """Override."""
        self._client = connection
