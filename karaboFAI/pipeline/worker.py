"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

Abstract class worker.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from queue import Empty, Queue

# import QtCore inside package pipeline from package gui will result in
# circle import
from PyQt5.QtCore import pyqtSignal, QThread

from ..config import config


class Worker(QThread):
    # post messages in the main GUI
    debug_sgn = pyqtSignal(str)
    info_sgn = pyqtSignal(str)
    warning_sgn = pyqtSignal(str)
    error_sgn = pyqtSignal(str)

    def __init__(self):
        super().__init__()

        self._input = None
        self._output = Queue(maxsize=config["MAX_QUEUE_SIZE"])

    @property
    def output(self):
        return self._output

    def connect_input(self, worker):
        if not isinstance(worker, Worker):
            raise ValueError

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
