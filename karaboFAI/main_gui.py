"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

Abstract main GUI.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import os
import time
import logging
from queue import Queue, Empty
from weakref import WeakKeyDictionary

import zmq

from .logger import logger
from .widgets.pyqtgraph import QtCore, QtGui
from .widgets import GuiLogger
from .windows import DrawMaskWindow, ImageToolWindow
from .data_acquisition import DataAcquisition
from .data_processing import Data4Visualization
from .file_server import FileServer
from .config import config


class Mediator(QtCore.QObject):
    roi_intensity_window_sgn = QtCore.pyqtSignal(int)
    roi_hist_clear_sgn = QtCore.pyqtSignal()
    roi1_region_changed_sgn = QtCore.pyqtSignal(bool, int, int, int, int)
    roi2_region_changed_sgn = QtCore.pyqtSignal(bool, int, int, int, int)

    threshold_mask_change_sgn = QtCore.pyqtSignal(float, float)

    @QtCore.pyqtSlot()
    def onRoiIntensityWindowChange(self):
        v = int(self.sender().text())
        self.roi_intensity_window_sgn.emit(v)

    @QtCore.pyqtSlot()
    def onRoiHistClear(self):
        self.roi_hist_clear_sgn.emit()

    @QtCore.pyqtSlot(bool, int, int, int, int)
    def onRoi1Changed(self, activated, w, h, cx, cy):
        self.roi1_region_changed_sgn.emit(activated, w, h, cx, cy)

    @QtCore.pyqtSlot(bool, int, int, int, int)
    def onRoi2Changed(self, activated, w, h, cx, cy):
        self.roi2_region_changed_sgn.emit(activated, w, h, cx, cy)

    @QtCore.pyqtSlot(float, float)
    def onThresholdMaskChange(self, lb, ub):
        self.threshold_mask_change_sgn.emit(lb, ub)


class MainGUI(QtGui.QMainWindow):
    """Abstract main GUI."""
    _root_dir = os.path.dirname(os.path.abspath(__file__))

    image_mask_sgn = QtCore.pyqtSignal(str)  # filename

    daq_started_sgn = QtCore.pyqtSignal()
    daq_stopped_sgn = QtCore.pyqtSignal()
    file_server_started_sgn = QtCore.pyqtSignal()
    file_server_stopped_sgn = QtCore.pyqtSignal()

    def __init__(self, detector, screen_size=None):
        """Initialization.

        :param str detector: detector name, e.g. "AGIPD", "LPD".
        """
        super().__init__()

        self._mediator = Mediator()

        # update global configuration
        config.load(detector)

        self._pulse_resolved = config["PULSE_RESOLVED"]

        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)

        self.title = detector + " Azimuthal Integration"
        self.setWindowTitle(self.title + " - main GUI")

        self._cw = QtGui.QWidget()
        self.setCentralWidget(self._cw)

        # *************************************************************
        # Tool bar
        # Note: the order of 'addAction` affect the unittest!!!
        # *************************************************************
        self._tool_bar = self.addToolBar("Control")

        #
        self._start_at = QtGui.QAction(
            QtGui.QIcon(os.path.join(self._root_dir, "icons/start.png")),
            "Start DAQ",
            self)
        self._tool_bar.addAction(self._start_at)
        self._start_at.triggered.connect(self.onStartDAQ)

        #
        self._stop_at = QtGui.QAction(
            QtGui.QIcon(os.path.join(self._root_dir, "icons/stop.png")),
            "Stop DAQ",
            self)
        self._tool_bar.addAction(self._stop_at)
        self._stop_at.triggered.connect(self.onStopDAQ)
        self._stop_at.setEnabled(False)

        #
        self._draw_mask_at = QtGui.QAction(
            QtGui.QIcon(os.path.join(self._root_dir, "icons/draw_mask.png")),
            "Draw mask",
            self)
        self._draw_mask_at.triggered.connect(
            lambda: DrawMaskWindow(self._data, parent=self))
        self._tool_bar.addAction(self._draw_mask_at)

        #
        load_mask_at = QtGui.QAction(
            QtGui.QIcon(os.path.join(self._root_dir, "icons/load_mask.png")),
            "Load mask",
            self)
        load_mask_at.triggered.connect(self.loadMaskImage)
        self._tool_bar.addAction(load_mask_at)

        #
        image_tool_at = QtGui.QAction(
            QtGui.QIcon(os.path.join(self._root_dir, "icons/image_tool.png")),
            "Image tool",
            self)
        image_tool_at.triggered.connect(
            lambda: ImageToolWindow(
                self._data, mediator=self._mediator, parent=self))
        self._tool_bar.addAction(image_tool_at)

        # *************************************************************
        # Miscellaneous
        # *************************************************************

        self._data = Data4Visualization()

        # book-keeping opened windows
        self._windows = WeakKeyDictionary()

        # book-keeping control widgets
        self._ctrl_widgets = []

        self._mask_image = None

        self._disabled_widgets_during_daq = [
            load_mask_at,
        ]

        self._logger = GuiLogger(self) 
        logging.getLogger().addHandler(self._logger)

        self._file_server = None

        if screen_size is None:
            self.move(0, 0)
        else:
            self.move(screen_size.width()/2 - self.width()/2,
                      screen_size.height()/20)

        self._daq_queue = Queue(maxsize=config["MAX_QUEUE_SIZE"])
        self._proc_queue = Queue(maxsize=config["MAX_QUEUE_SIZE"])

        # a DAQ worker which acquires the data in another thread
        self._daq_worker = DataAcquisition(self._daq_queue)
        # a data processing worker which processes the data in another thread
        self._proc_worker = None

        # For real time plot
        self._running = False
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.updateAll)
        self.timer.start(config["TIMER_INTERVAL"])

    def initConnection(self):
        """Set up all signal and slot connections."""
        self._daq_worker.message.connect(self.onMessageReceived)

        self.data_ctrl_widget.data_source_sgn.connect(
            self._proc_worker.onSourceChanged)
        self.data_ctrl_widget.server_tcp_sgn.connect(
            self._daq_worker.onServerTcpChanged)

        self._proc_worker.message.connect(self.onMessageReceived)

        self.image_mask_sgn.connect(self._proc_worker.onImageMaskChanged)
        self.data_ctrl_widget.data_source_sgn.connect(
            self._proc_worker.onSourceChanged)

    def initUI(self):
        raise NotImplementedError

    def updateAll(self):
        """Update all the plots in the main and child windows."""
        if not self._running:
            return

        # TODO: improve plot updating
        # Use multithreading for plot updating. However, this is not the
        # bottleneck for the performance.

        try:
            self._data.set(self._proc_queue.get_nowait())
        except Empty:
            return

        # clear the previous plots no matter what comes next
        for w in self._windows.keys():
            w.clear()

        if self._data.get().empty():
            logger.info("Bad train with ID: {}".format(self._data.get().tid))
            return

        t0 = time.perf_counter()

        # update the all the plots
        for w in self._windows.keys():
            w.update()

        logger.debug("Time for updating the plots: {:.1f} ms"
                     .format(1000 * (time.perf_counter() - t0)))

        logger.info("Updated train with ID: {}".format(self._data.get().tid))

    def registerWindow(self, instance):
        self._windows[instance] = 1

    def unregisterWindow(self, instance):
        del self._windows[instance]

    def registerCtrlWidget(self, instance):
        self._ctrl_widgets.append(instance)

    def loadMaskImage(self):
        filename = QtGui.QFileDialog.getOpenFileName()[0]
        if not filename:
            logger.error("Please specify the image mask file!")
        self.image_mask_sgn.emit(filename)

    def onStartDAQ(self):
        """Actions taken before the start of a 'run'."""
        self.clearQueues()
        self._running = True  # starting to update plots

        if not self.updateSharedParameters():
            return
        self._proc_worker.start()
        self._daq_worker.start()

        self._start_at.setEnabled(False)
        self._stop_at.setEnabled(True)
        for widget in self._disabled_widgets_during_daq:
            widget.setEnabled(False)
        self.daq_started_sgn.emit()

    def onStopDAQ(self):
        """Actions taken before the end of a 'run'."""
        self._running = False

        self.clearWorkers()
        self.clearQueues()

        self._start_at.setEnabled(True)
        self._stop_at.setEnabled(False)
        for widget in self._disabled_widgets_during_daq:
            widget.setEnabled(True)
        self.daq_stopped_sgn.emit()

    def clearWorkers(self):
        self._proc_worker.terminate()
        self._daq_worker.terminate()
        self._proc_worker.wait()
        self._daq_worker.wait()

    def clearQueues(self):
        with self._daq_queue.mutex:
            self._daq_queue.queue.clear()
        with self._proc_queue.mutex:
            self._proc_queue.queue.clear()

    def onStartServeFile(self):
        """Actions taken before the start of file serving."""
        # process can only be start once
        folder, port = self.data_ctrl_widget.file_server
        self._file_server = FileServer(folder, port)
        try:
            # TODO: signal the end of file serving
            self._file_server.start()
            logger.info("Start serving file in the folder {} through port {}"
                        .format(folder, port))
        except FileNotFoundError:
            logger.info("{} does not exist!".format(folder))
            return
        except zmq.error.ZMQError:
            logger.info("Port {} is already in use!".format(port))
            return

        self.file_server_started_sgn.emit()

    def onStopServeFile(self):
        """Actions taken before the end of file serving."""
        self._file_server.terminate()

        self.file_server_stopped_sgn.emit()

    def updateSharedParameters(self):
        """Update shared parameters for all child windows.

        :returns bool: True if all shared parameters successfully parsed
            and emitted, otherwise False.
        """
        total_info = ""
        for widget in self._ctrl_widgets:
            info = widget.updateSharedParameters()
            if info is None:
                return False
            total_info += info

        logger.info(total_info)
        return True

    @QtCore.pyqtSlot(str)
    def onMessageReceived(self, msg):
        logger.info(msg)

    def closeEvent(self, QCloseEvent):
        self.clearWorkers()

        if self._file_server is not None and self._file_server.is_alive():
            self._file_server.terminate()

        super().closeEvent(QCloseEvent)
