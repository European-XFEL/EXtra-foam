"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

Data acquisition.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from queue import Queue, Full

from PyQt5 import QtCore

from zmq.error import ZMQError

from karabo_bridge import Client

from .worker import Worker
from ..config import config
from ..helpers import profiler


class Bridge(Worker):
    def __init__(self):
        """Initialization."""
        super().__init__()

        self._tcp_host = None
        self._tcp_port = None

    @QtCore.pyqtSlot(str)
    def onTcpHostChange(self, hostname):
        self._tcp_host = hostname

    @QtCore.pyqtSlot(int)
    def onTcpPortChange(self, port):
        self._tcp_port = port

    def run(self):
        """Override."""
        endpoint = f"tcp://{self._tcp_host}:{self._tcp_port}"

        self._running = True
        try:
            with Client(endpoint, timeout=1) as client:
                self.log("Bind to server {}!".format(endpoint))
                while self._running:
                    try:
                        data = self._recv(client)
                    except TimeoutError:
                        continue

                    while self._running:
                        try:
                            self._output.put(data, timeout=config['TIMEOUT'])
                            break
                        except Full:
                            continue
        except ZMQError:
            self.log(f"ZMQError with endpoint: {endpoint}")
            raise

        self.log("Bridge client stopped!")

    @profiler("Receive Data from Bridge")
    def _recv(self, client):
        return client.next()
