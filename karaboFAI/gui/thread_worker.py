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
import time

from redis import ConnectionError

from PyQt5.QtCore import pyqtSignal, pyqtSlot, QObject, QThread

from zmq.error import ZMQError

from karabo_bridge import Client

from ..config import config, DataSource, redis_connection
from ..utils import profiler


class QThreadWorker(QThread):
    """Base worker class for small online data analysis."""
    log_debug_sgn = pyqtSignal(str)
    log_info_sgn = pyqtSignal(str)
    log_warning_sgn = pyqtSignal(str)
    log_error_sgn = pyqtSignal(str)

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

    def connectToMainThread(self, instance):
        """Connect all log signals to slots in the Main Thread."""
        self.log_debug_sgn.connect(instance.onLogDebugReceived)
        self.log_info_sgn.connect(instance.onLogInfoReceived)
        self.log_warning_sgn.connect(instance.onLogWarningReceived)
        self.log_error_sgn.connect(instance.onLogErrorReceived)

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


class ThreadLoggerBridge(QObject):
    """QThread which subscribes logs the Redis server.

    This QThread forward the message from the Redis server and send
    it to the MainGUI via signal-slot connection.
    """
    log_debug_sgn = pyqtSignal(str)
    log_info_sgn = pyqtSignal(str)
    log_warning_sgn = pyqtSignal(str)
    log_error_sgn = pyqtSignal(str)

    def __init__(self):
        super().__init__()

    def recv_messages(self):
        sub = redis_connection().pubsub()
        sub.psubscribe("log:*")

        while True:
            try:
                msg = sub.get_message()
                if msg and isinstance(msg['data'], str):
                    channel = msg['channel']
                    log_msg = msg['data']

                    if channel == 'log:debug':
                        self.log_debug_sgn.emit(log_msg)
                    elif channel == 'log:info':
                        self.log_info_sgn.emit(log_msg)
                    elif channel == 'log:warning':
                        self.log_warning_sgn.emit(log_msg)
                    elif channel == 'log:error':
                        self.log_error_sgn.emit(log_msg)

            except ConnectionError:
                pass

            # TODO: find a magic number
            time.sleep(0.001)

    def connectToMainThread(self, instance):
        """Connect all log signals to slots in the Main Thread."""
        self.log_debug_sgn.connect(instance.onLogDebugReceived)
        self.log_info_sgn.connect(instance.onLogInfoReceived)
        self.log_warning_sgn.connect(instance.onLogWarningReceived)
        self.log_error_sgn.connect(instance.onLogErrorReceived)
