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

from zmq.error import ZMQError

from karabo_bridge import Client

from .exceptions import ProcessingError
from .worker import ProcessWorker
from ..metadata import Metadata as mt
from ..config import config, DataSource
from ..utils import profiler


class Bridge(ProcessWorker):
    """Bridge running in a separate processor.

    This is main workhorse for large scale online data analysis.
    """
    def __init__(self, name="bridge"):
        """Initialization."""
        super().__init__(name)

        self._clients = dict()

    def _run_once(self):
        timeout = self._timeout

        if not self.running:
            self.wait()
            # when the bridge is paused, we empty the output queue to avoid
            # sending old data when it is resumed.
            self.empty_output()
            # update the client when resumed
            self.update()

        for client in self._clients.values():
            try:
                data = self._recv_imp(client)
                if self._source_type == DataSource.BRIDGE:
                    # always keep the latest data in the queue
                    try:
                        self._output.put(data, timeout=timeout)
                    except Full:
                        self.pop_output()
                        print("Data dropped by the bridge")
                elif self._source_type == DataSource.FILE:
                    # wait until data in the queue has been processed
                    while not self.closing:
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

    @profiler("Receive Data from Bridge")
    def _recv_imp(self, client):
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
