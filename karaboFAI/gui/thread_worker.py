"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

QThread workers

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from queue import Empty, Full, Queue

from PyQt5.QtCore import pyqtSignal, pyqtSlot, QThread

from zmq.error import ZMQError

from karabo_bridge import Client

from ..config import config, DataSource
from ..utils import profiler


class QThreadWorker(QThread):
    """Base worker class for small online data analysis."""
    # post messages in the main GUI
    debug_sgn = pyqtSignal(str)
    info_sgn = pyqtSignal(str)
    warning_sgn = pyqtSignal(str)
    error_sgn = pyqtSignal(str)

    def __init__(self):
        super().__init__()

        self._input = Queue(maxsize=config["MAX_QUEUE_SIZE"])
        self._output = Queue(maxsize=config["MAX_QUEUE_SIZE"])

    @property
    def output(self):
        return self._output

    def connect_input(self, worker):
        if not isinstance(worker, QThreadWorker):
            raise TypeError("QThreadWorker is only allowed to connect "
                            "QThreadWorker instance.")

        self._input = worker.output

    def debug(self, msg):
        """Log debug information in the main GUI."""
        self.debug_sgn.emit(msg)

    def info(self, msg):
        """Log info information in the main GUI."""
        self.info_sgn.emit(msg)

    def warning(self, msg):
        """Log warning information in the main GUI."""
        self.warning_sgn.emit(msg)

    def error(self, msg):
        """Log error information in the main GUI."""
        self.error_sgn.emit(msg)

    def log_on_main_thread(self, instance):
        self.debug_sgn.connect(instance.onDebugReceived)
        self.info_sgn.connect(instance.onInfoReceived)
        self.warning_sgn.connect(instance.onWarningReceived)
        self.error_sgn.connect(instance.onErrorReceived)

    def empty_output(self):
        """Empty the output queue."""
        while not self._output.empty():
            try:
                self._output.get_nowait()
            except Empty:
                break

    def pop_output(self):
        """Remove and return an item from the output queue"""
        try:
            return self._output.get_nowait()
        except Empty:
            pass


class QThreadBridge(QThreadWorker):
    """QThreadBridge used in small online data visualization app."""
    def __init__(self, *args):
        """Initialization."""
        super().__init__(*args)

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
                while not self.isInterruptionRequested():
                    self.info("Bind to server {}!".format(endpoint))
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
                            self._output.put(data, timeout=timeout)
                        except Full:
                            self.pop_output()
                            self.debug("Data dropped by the bridge")
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
