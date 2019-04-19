"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

Data acquisition.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import queue

from karabo_bridge import Client

from .worker import Worker
from ..config import config
from ..gui import QtCore
from ..helpers import profiler


class DataAcquisition(Worker):
    def __init__(self, out_queue):
        """Initialization."""
        super().__init__()

        self._tcp_host = None
        self._tcp_port = None

        self._out_queue = out_queue

    @QtCore.pyqtSlot(str)
    def onTcpHostChange(self, hostname):
        self._tcp_host = hostname

    @QtCore.pyqtSlot(int)
    def onTcpPortChange(self, port):
        self._tcp_port = port

    def run(self):
        """Override."""
        end_point = f"tcp://{self._tcp_host}:{self._tcp_port}"

        self._running = True
        with Client(end_point, timeout=1) as client:
            self.log("Bind to server {}!".format(end_point))
            while self._running:
                try:
                    data = self._recv(client)
                except TimeoutError:
                    continue

                while self._running:
                    try:
                        self._out_queue.put(data, timeout=config['TIMEOUT'])
                        break
                    except queue.Full:
                        continue

        self.log("DAQ stopped!")

    @profiler("Receive Data from Bridge")
    def _recv(self, client):
        return client.next()
