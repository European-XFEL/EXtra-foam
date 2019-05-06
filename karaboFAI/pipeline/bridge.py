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

from PyQt5.QtCore import pyqtSlot

from zmq.error import ZMQError

from karabo_bridge import Client

from .worker import Worker
from ..config import config
from ..helpers import profiler


class Bridge(Worker):
    def __init__(self):
        """Initialization."""
        super().__init__()

        self._endpoint = None

    @pyqtSlot(str)
    def onEndpointChange(self, endpoint):
        self._endpoint = endpoint

    def run(self):
        """Override."""
        endpoint = self._endpoint
        self.empty_output()  # remove old data

        try:
            with Client(endpoint, timeout=config['TIMEOUT']) as client:
                self.info("Bind to server {}!".format(endpoint))
                while not self.isInterruptionRequested():
                    try:
                        data = self._recv(client)
                    except TimeoutError:
                        continue

                    # always keep the latest data in the queue
                    try:
                        self._output.put_nowait(data)
                    except Full:
                        self.pop_output()
        except ZMQError:
            self.error(f"ZMQError with endpoint: {endpoint}")
            raise

        self.info("Bridge client stopped!")

    @profiler("Receive Data from Bridge")
    def _recv(self, client):
        return client.next()
