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
from ..config import config, DataSource
from ..helpers import profiler


class Bridge(Worker):
    def __init__(self):
        """Initialization."""
        super().__init__()

        self._source_type = None
        self._endpoint = None

    @pyqtSlot(str)
    def onEndpointChange(self, endpoint):
        self._endpoint = endpoint

    def run(self):
        """Override."""
        endpoint = self._endpoint
        self.empty_output()  # remove old data

        timeout = config['TIMEOUT']
        try:
            with Client(endpoint, timeout=timeout) as client:
                self.info("Bind to server {}!".format(endpoint))
                while not self.isInterruptionRequested():
                    try:
                        data = self._recv(client)
                    except TimeoutError:
                        continue

                    # Note: the pipeline is not reliable since for whatever
                    #       reason the output queue could be filled because
                    #       the consumer has not started. Then, since the
                    #       bridge is always faster than the processing
                    #       pipeline, the output queue will stay filled, which
                    #       make the specification of queue length useless.

                    if self._source_type == DataSource.BRIDGE:
                        # always keep the latest data in the queue
                        try:
                            self._output.put_nowait(data)
                        except Full:
                            self.pop_output()
                    else:  # self._source_type == DataSource.FILE:
                        # wait until data in the queue has been processed
                        while not self.isInterruptionRequested():
                            try:
                                self._output.put(data, timeout=timeout)
                                break
                            except Full:
                                continue

        except ZMQError:
            self.error(f"ZMQError with endpoint: {endpoint}")
            raise

        self.info("Bridge client stopped!")

    @profiler("Receive Data from Bridge")
    def _recv(self, client):
        return client.next()

    @pyqtSlot(int)
    def onSourceTypeChange(self, value):
        self._source_type = value
