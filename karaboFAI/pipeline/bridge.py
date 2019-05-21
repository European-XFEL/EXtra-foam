"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

Data acquisition.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from queue import Full
import multiprocessing as mp

from zmq.error import ZMQError

from karabo_bridge import Client

from .exceptions import ProcessingError
from .worker import ProcessWorker
from ..metadata import Metadata as mt
from ..config import config, DataSource
from ..helpers import profiler


class Bridge(ProcessWorker):
    """Bridge running in a separate processor.

    This is main workhorse for large scale online data analysis.
    """
    def __init__(self, name="bridge"):
        """Initialization."""
        super().__init__(name)

        self._pause_event = mp.Event()

        self._clients = dict()

    def run(self):
        """Override."""
        self.empty_output()

        timeout = config['TIMEOUT']

        print("Bridge process started")

        while not self._shutdown_event.is_set():
            if not self._pause_event.is_set():
                self.empty_output()
                self.update()
                self._pause_event.wait(timeout=timeout)
                continue

            for client in self._clients.values():
                try:
                    data = self._recv(client)
                    print("data received")
                    if self._source_type == DataSource.BRIDGE:
                        # always keep the latest data in the queue
                        try:
                            self._output.put(data, timeout=timeout)
                        except Full:
                            self.pop_output()
                            print("Data dropped by the bridge")
                    elif self._source_type == DataSource.FILE:
                        # wait until data in the queue has been processed
                        while not self._shutdown_event.is_set():
                            try:
                                self._output.put(data, timeout=timeout)
                                break
                            except Full:
                                continue
                    else:
                        raise ProcessingError(
                            f"Unknown source type {self._source_type}!")
                except TimeoutError:
                    continue

        print("Bridge shutdown cleanly")

    def activate(self):
        self._pause_event.set()

    def pause(self):
        self._pause_event.clear()

    @profiler("Receive Data from Bridge")
    def _recv(self, client):
        return client.next()

    def update(self):
        super().update()

        endpoint = self._meta.get(mt.DATA_SOURCE, 'endpoint')
        if endpoint is None or endpoint in self._clients:
            return

        try:
            # destroy the old connections
            for client in self._clients.values():
                client._context.destroy(linger=0)
            self._clients.clear()

            client = Client(endpoint, timeout=config['TIMEOUT'])
            self._clients[endpoint] = client
        except ZMQError:
            return

    def __del__(self):
        for client in self._clients.values():
            client._context.destroy(linger=0)
