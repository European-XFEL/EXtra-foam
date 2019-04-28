"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

Abstract class worker.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from queue import Queue

# import QtCore inside package pipeline from package gui will result in
# circle import
from PyQt5 import QtCore

from ..config import config


class Worker(QtCore.QThread):
    # post messages in the main GUI
    message = QtCore.pyqtSignal(str)

    def __init__(self):
        super().__init__()

        queue_size = config["MAX_QUEUE_SIZE"]
        self._input = Queue(maxsize=queue_size)
        self._output = Queue(maxsize=queue_size)

        self._running = False

    def connect(self, worker):
        if not isinstance(worker, Worker):
            raise ValueError

        worker._input = self._output

    def clear_queue(self):
        with self._input.mutex:
            self._input.queue.clear()
        with self._output.mutex:
            self._output.queue.clear()

    def log(self, msg):
        """Log information in the main GUI.

        :param str msg: message string.
        """
        self.message.emit(msg)

    def terminate(self):
        self._running = False
