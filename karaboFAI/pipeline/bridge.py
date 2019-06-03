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

from karabo_bridge import Client

from .exceptions import ProcessingError
from .worker import ProcessWorker
from ..metadata import Metadata as mt
from ..config import DataSource
from ..utils import profiler


class Bridge(ProcessWorker):
    """Bridge running in a separate processor.

    This is main workhorse for large scale online data analysis.
    """
    def __init__(self, name="bridge"):
        """Initialization."""
        super().__init__(name)

        self._client = None

    def _run_once(self):
        timeout = self._timeout

        if not self.running:
            self.wait()
            # when the bridge is paused, we empty the output queue to avoid
            # sending old data when it is resumed.
            self.empty_output()
            # update the client when resumed
            self.update()

        try:
            data = self._recv_imp(self._client)
            if self._source_type == DataSource.BRIDGE:
                # always keep the latest data in the queue
                try:
                    self._output.put(data, timeout=timeout)
                except Full:
                    self.pop_output()
                    self.log.info("Data dropped by the bridge")
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
            pass

    @profiler("Receive Data from Bridge")
    def _recv_imp(self, client):
        return client.next()

    def update(self):
        # get source type
        super().update()

        endpoint = self._meta.get(mt.DATA_SOURCE, 'endpoint')

        # destroy the old connection and make a new one
        if self._client is not None:
            del self._client
        self._client = Client(endpoint, timeout=self._timeout)
