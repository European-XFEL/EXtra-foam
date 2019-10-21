"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

Data pipes.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import multiprocessing as mp
import threading
from queue import Empty, Full, Queue
import time

from karabo_bridge import Client

from .data_model import ProcessedData
from ..config import config, DataSource
from ..utils import profiler
from ..ipc import ProcessWorkerLogger
from ..database import MetaProxy
from ..database import Metadata as mt


class Pipe:
    """Abstract Pipe class.

    Pipe is used to transfer data between different processes via socket,
    file, shared memory, etc. Internally, it stores the data in a
    multi-threading queue and exchanges data with the owner process via
    this Queue.
    """

    def __init__(self, name, *, daemon=True, gui=False):
        """Initialization.

        :param str name: name of the pipe
        :param bool daemon: daemonness of the pipe thread.
        :param bool gui: True for connecting to GUI and False for not.
        """
        self._name = name

        self._daemon = daemon
        self._gui = gui

        self._data = Queue(maxsize=config["MAX_QUEUE_SIZE"])

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
                                  daemon=self._daemon)
        thread.start()
        return thread

    def work(self, close_ev, update_ev):
        """Target function for running in a thread.

        :param multiprocessing.Event close_ev: if this event is set, the
            target function running in a thread will be terminated.
        :param multithreading.Event update_ev: if this event is set, the
            pipe will update its status.
        """
        raise NotImplementedError

    def clean(self):
        """Empty the data queue."""
        while not self._data.empty():
            try:
                self._data.get_nowait()
            except Empty:
                break


class PipeIn(Pipe):
    """A pipe that receives incoming data."""
    def get(self, timeout=None):
        """Remove and return the first data item in the queue."""
        return self._data.get(timeout=timeout)

    def get_nowait(self):
        return self._data.get_nowait()


class PipeOut(Pipe):
    """A pipe that dispatches data outwards."""
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
            self.log.warning(f"Data dropped by {self._name} due to "
                             f"slowness of the pipeline")
            self._data.put_nowait(item)


class KaraboBridge(PipeIn):
    """Karabo bridge client which is an input pipe."""
    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)

    def work(self, close_ev, update_ev):
        """Override."""
        timeout = config['TIMEOUT']

        # the time when the previous data was received
        prev_data_arrival_time = None
        # Karabo bridge client instance
        client = None
        while not close_ev.is_set():
            if update_ev.is_set():
                cfg = self._meta.get_all(mt.CONNECTION)
                endpoint = cfg['endpoint']
                src_type = DataSource(int(cfg['source_type']))

                if client is None:
                    # destroy the zmq socket
                    del client
                # instantiate a new client
                client = Client(endpoint, timeout=timeout)
                self.log.debug(f"Instantiate a bridge client connected to "
                               f"{endpoint}")
                update_ev.clear()

            if client is None:
                time.sleep(0.001)
                continue

            try:
                data = self._recv_imp(client)

                if prev_data_arrival_time is not None:
                    fps = 1.0 / (time.time() - prev_data_arrival_time)
                    self.log.debug(f"Bridge recv FPS: {fps:>4.1f} Hz")
                prev_data_arrival_time = time.time()

                data = self._preprocess(data, src_type)

                # wait until data in the queue has been processed
                # Note: if the queue is full, whether the data should be dropped
                #       is determined by the main thread of its owner process.
                while not close_ev.is_set():
                    try:
                        self._data.put(data, timeout=timeout)
                        break
                    except Full:
                        continue

            except TimeoutError:
                pass

    @profiler("Receive Data from Bridge")
    def _recv_imp(self, client):
        return client.next()

    def _preprocess(self, data, src_type):
        raw, meta = data

        # get the train ID of the first metadata
        # Note: this is better than meta[src_name] because:
        #       1. For streaming AGIPD/LPD data from files, 'src_name' does
        #          not work;
        #       2. Prepare for the scenario where a 2D detector is not
        #          mandatory.
        tid = next(iter(meta.values()))["timestamp.tid"]

        sources = sorted(meta.keys())
        processed = ProcessedData(tid, sources)

        return {
            "processed": processed,
            "raw": raw,
            "meta": {
                "source_type": src_type,
            },
        }


class MpInQueue(PipeIn):
    """A pipe which uses a multi-processing queue to receive data."""
    def __init__(self, name, **kwargs):
        super().__init__(name, daemon=True, **kwargs)

        self._client = mp.Queue(maxsize=config["MAX_QUEUE_SIZE"])

    def work(self, close_ev, update_ev):
        timeout = config['TIMEOUT']

        while not close_ev.is_set():
            # receive data from the client
            try:
                data = self._client.get(timeout=timeout)
            except Empty:
                continue

            while not close_ev.is_set():
                # store the newly received data
                try:
                    self._data.put(data, timeout=timeout)
                    break
                except Full:
                    continue

        self._client.cancel_join_thread()

    def connect(self, pipe_out):
        if isinstance(pipe_out, MpOutQueue):
            self._client = pipe_out._client
        else:
            raise NotImplementedError(f"Cannot connect {self.__class__} "
                                      f"(input) to {type(pipe_out)} (output)")


class MpOutQueue(PipeOut):
    """A pipe which uses a multi-processing queue to dispatch data."""
    def __init__(self, name, **kwargs):
        super().__init__(name, daemon=True, **kwargs)

        self._client = mp.Queue(maxsize=config["MAX_QUEUE_SIZE"])

    def work(self, close_ev, update_ev):
        timeout = config['TIMEOUT']

        while not close_ev.is_set():
            # pop the stored data
            try:
                data = self._data.get(timeout=timeout)
            except Empty:
                continue

            if self._gui:
                data_out = data['processed']
            else:
                data_out = {key: data[key] for key
                            in ['processed', 'raw', 'meta']}

            while not close_ev.is_set():
                # push the stored data into client
                try:
                    self._client.put(data_out, timeout=timeout)
                    break
                except Full:
                    continue

        self._client.cancel_join_thread()
