"""
Offline and online data analysis and visualization tool for Centre  of
mass analysis from different data acquired with various detectors at
European XFEL.

Main Bragg GUI.

Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import os
import time
import logging
from queue import Queue, Empty
from weakref import WeakKeyDictionary

from .config import config
from .data_acquisition import DataAcquisition
from .data_processing import COMDataProcessor as DataProcessor, ProcessedData
from .logger import logger
from .widgets.pyqtgraph import QtCore, QtGui
from .widgets import (
    DataSrcWidget, ExpSetUpWidget, GmtSetUpWidget, GuiLogger
)
from .windows import BraggSpotsWindow, DrawMaskWindow


class MainBraggGUI(QtGui.QMainWindow):
    """The main GUI for azimuthal integration."""

    class Data4Visualization:
        """Data shared between all the windows and widgets.

        The internal data is only modified in MainGUI.updateAll()
        """

        def __init__(self):
            self.__value = ProcessedData(-1)

        def get(self):
            return self.__value

        def set(self, value):
            self.__value = value

    image_mask_sgn = QtCore.pyqtSignal(str)  # filename

    daq_started_sgn = QtCore.pyqtSignal()
    daq_stopped_sgn = QtCore.pyqtSignal()
    file_server_started_sgn = QtCore.pyqtSignal()
    file_server_stopped_sgn = QtCore.pyqtSignal()

    _height = 600  # window height, in pixel
    _width = 1200  # window width, in pixel

    def __init__(self, topic, screen_size=None):
        """Initialization.

        :param str topic: detector topic, allowed options "SPB", "FXE"
        """
        super().__init__()

        # update global configuration
        config.load(topic)

        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.setFixedSize(self._width, self._height)

        self.title = topic + " Centre of Mass Analysis"
        self.setWindowTitle(self.title + " - main GUI")

        self._cw = QtGui.QWidget()
        self.setCentralWidget(self._cw)

        # *************************************************************
        # Tool bar
        # *************************************************************
        tool_bar = self.addToolBar("Control")

        root_dir = os.path.dirname(os.path.abspath(__file__))

        #
        self._start_at = QtGui.QAction(
            QtGui.QIcon(os.path.join(root_dir, "icons/start.png")),
            "Start DAQ",
            self)
        tool_bar.addAction(self._start_at)
        self._start_at.triggered.connect(self.onStartDAQ)

        #
        self._stop_at = QtGui.QAction(
            QtGui.QIcon(os.path.join(root_dir, "icons/stop.png")),
            "Stop DAQ",
            self)
        tool_bar.addAction(self._stop_at)
        self._stop_at.triggered.connect(self.onStopDAQ)
        self._stop_at.setEnabled(False)

        #
        open_bragg_spots_window_at = QtGui.QAction(
            QtGui.QIcon(os.path.join(root_dir, "icons/bragg_spots.png")),
            "Bragg spots",
            self)
        open_bragg_spots_window_at.triggered.connect(
            lambda: BraggSpotsWindow(self._data, parent=self))
        tool_bar.addAction(open_bragg_spots_window_at)

        #
        self._draw_mask_at = QtGui.QAction(
            QtGui.QIcon(os.path.join(root_dir, "icons/draw_mask.png")),
            "Draw mask",
            self)
        self._draw_mask_at.triggered.connect(
            lambda: DrawMaskWindow(self._data, parent=self))
        tool_bar.addAction(self._draw_mask_at)

        #
        self._load_mask_at = QtGui.QAction(
            QtGui.QIcon(os.path.join(root_dir, "icons/load_mask.png")),
            "Load mask",
            self)
        self._load_mask_at.triggered.connect(self.loadMaskImage)
        tool_bar.addAction(self._load_mask_at)

        #
        self._load_geometry_file_at = QtGui.QAction(
            QtGui.QIcon(
                self.style().standardIcon(QtGui.QStyle.SP_DriveCDIcon)),
            "geometry file",
            self)
        self._load_geometry_file_at.triggered.connect(
            self.loadGeometryFile)
        tool_bar.addAction(self._load_geometry_file_at)

        # *************************************************************
        # Miscellaneous
        # *************************************************************
        self._data = self.Data4Visualization()

        # book-keeping opened widgets and windows
        self._plot_windows = WeakKeyDictionary()

        self._mask_image = None

        self._disabled_widgets_during_daq = [
            self._load_mask_at,
            self._load_geometry_file_at,
        ]
        self.gmt_setup_widget = GmtSetUpWidget(parent=self)
        self.exp_setup_widget = ExpSetUpWidget(parent=self)
        self.data_src_widget = DataSrcWidget(parent=self)

        # *************************************************************
        # log window
        # *************************************************************

        self._logger = GuiLogger(self)
        logging.getLogger().addHandler(self._logger)

        self._file_server = None

        self.initUI()

        if screen_size is None:
            self.move(0, 0)
        else:
            self.move(screen_size.width()/2 - self._width/2,
                      screen_size.height()/20)

        self._daq_queue = Queue(maxsize=config["MAX_QUEUE_SIZE"])
        self._proc_queue = Queue(maxsize=config["MAX_QUEUE_SIZE"])

        # a DAQ worker which acquires the data in another thread
        self._daq_worker = DataAcquisition(self._daq_queue)
        # a data processing worker which processes the data in another thread
        self._proc_worker = DataProcessor(self._daq_queue, self._proc_queue)

        self.initConnection()
        # For real time plot
        self._running = False
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.updateAll)
        self.timer.start(config["TIMER_INTERVAL"])
        self.show()

    def initConnection(self):
        """Set up all signal and slot connections."""
        self._daq_worker.message.connect(self.onMessageReceived)

        self._proc_worker.message.connect(self.onMessageReceived)

        self.image_mask_sgn.connect(self._proc_worker.onImageMaskChanged)

        self.gmt_setup_widget.geometry_sgn.connect(
            self._proc_worker.onGeometryChanged)
        # self.exp_setup_widget.mask_range_sgn.connect(
        #     self._proc_worker.onMaskRangeChanged)
        self.exp_setup_widget.photon_energy_sgn.connect(
            self._proc_worker.onPhotonEnergyChanged)

        self.data_src_widget.data_source_sgn.connect(
            self._proc_worker.onSourceChanged)
        self.data_src_widget.pulse_range_sgn.connect(
            self._proc_worker.onPulseRangeChanged)
        self.data_src_widget.server_tcp_sgn.connect(
            self._daq_worker.onServerTcpChanged)

    def initUI(self):
        layout = QtGui.QGridLayout()

        layout.addWidget(self.gmt_setup_widget, 0, 0, 4, 1)
        layout.addWidget(self.exp_setup_widget, 0, 1, 4, 1)
        layout.addWidget(self.data_src_widget, 0, 2, 7, 1)
        layout.addWidget(self._logger.widget, 4, 0, 3, 2)
        self._cw.setLayout(layout)

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
        for w in self._plot_windows.keys():
            w.clear()

        if self._data.get().empty_image():
            logger.info("Bad train with ID: {}".format(self._data.get().tid))
            return

        t0 = time.perf_counter()

        # update the all the plots
        for w in self._plot_windows.keys():
            w.update()

        logger.debug("Time for updating the plots: {:.1f} ms"
                     .format(1000 * (time.perf_counter() - t0)))

        logger.info("Updated train with ID: {}".format(self._data.get().tid))

    def registerPlotWindow(self, instance):
        self._plot_windows[instance] = 1

    def unregisterPlotWindow(self, instance):
        del self._plot_windows[instance]

    def loadGeometryFile(self):
        filename = QtGui.QFileDialog.getOpenFileName()[0]
        if filename:
            self._gmt_children.geom_file_le.setText(filename)

    def loadMaskImage(self):
        filename = QtGui.QFileDialog.getOpenFileName()[0]
        if not filename:
            logger.error("Please specify the image mask file!")
        self.image_mask_sgn.emit(filename)

    def onStartDAQ(self):
        """Actions taken before the start of a 'run'."""
        self.clearQueues()
        self._running = True  # starting to update plots
        if not self.updateSharedParameters(log=True):
            return

        self._proc_worker.start()
        self._daq_worker.start()

        self._start_at.setEnabled(False)
        self._stop_at.setEnabled(True)
        for widget in self._disabled_widgets_during_daq:
            widget.setEnabled(False)

    def onStopDAQ(self):
        """Actions taken before the end of a 'run'."""
        self._running = False

        self.clearWorkers()
        self.clearQueues()

        self._start_at.setEnabled(True)
        self._stop_at.setEnabled(False)
        for widget in self._disabled_widgets_during_daq:
            widget.setEnabled(True)

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
        folder, port = self.data_src_widget.file_server
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

    def updateSharedParameters(self, log=False):
        """Update shared parameters for all child windows.

        :params bool log: True for logging shared parameters and False
            for not.

        Returns bool: True if all shared parameters successfully parsed
            and emitted, otherwise False.
        """
        ctrl_widgets = [
            self.gmt_setup_widget, self.exp_setup_widget, self.data_src_widget,
        ]
        for widget in ctrl_widgets:
            if not widget.updateSharedParameters(log=log):
                return False
        return True

    @QtCore.pyqtSlot(str)
    def onMessageReceived(self, msg):
        logger.info(msg)

    def closeEvent(self, QCloseEvent):
        super().closeEvent(QCloseEvent)

        self.clearWorkers()

        if self._file_server is not None and self._file_server.is_alive():
            self._file_server.terminate()
