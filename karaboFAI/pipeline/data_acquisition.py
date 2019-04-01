"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

Data acquisition.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import time
import queue

from karabo_bridge import Client
import zmq

from .worker import Worker
from ..config import config
from ..gui import QtCore
from ..logger import logger


class TimeoutClient(Client):
    """A karabo-bridge client with timeout."""
    def __init__(self, *args, timeout=None, **kwargs):
        super().__init__(*args, **kwargs)
        # timeout setting
        if timeout is not None:
            self._socket.RCVTIMEO = 1000 * timeout

        self._recv_ready = False

    def next(self):
        """Override."""
        if self._pattern == zmq.REQ and not self._recv_ready:
            self._socket.send(b'next')
            self._recv_ready = True

        msg = self._socket.recv_multipart(copy=False)
        self._recv_ready = False
        return self._deserialize(msg)


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
        with TimeoutClient(end_point, timeout=1) as client:
            self.log("Bind to server {}!".format(end_point))
            while self._running:
                t0 = time.perf_counter()

                try:
                    data = client.next()
                except zmq.error.Again:
                    continue

                logger.debug(
                    "Time for retrieving data from the server: {:.1f} ms"
                    .format(1000 * (time.perf_counter() - t0)))

                while self._running:
                    try:
                        self._out_queue.put(data, timeout=config['TIMEOUT'])
                        break
                    except queue.Full:
                        continue

        self.log("DAQ stopped!")
