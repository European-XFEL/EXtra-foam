import sys
import logging
import time
from collections import deque

import numpy as np

from karabo_bridge import Client

from .pyqtgraph.Qt import QtCore, QtGui
from .pyqtgraph import mkPen, intColor
from .logging import logger, GuiLogger
from .plot_widgets import LinePlotWidget, ImageViewWidget, LinePlotWindow
from .data_acquisition import DaqWorker
from .config import Config as cfg
from .config import DataSource


class InputDialogWithCheckBox(QtGui.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

    @classmethod
    def getResult(cls, parent, window_title, input_label, checkbox_label):
        dialog = cls(parent)

        dialog.setWindowTitle(window_title)

        label = QtGui.QLabel(input_label)
        text_le = QtGui.QLineEdit()

        ok_cb = QtGui.QCheckBox(checkbox_label)
        ok_cb.setChecked(True)

        buttons = QtGui.QDialogButtonBox(
            QtGui.QDialogButtonBox.Ok | QtGui.QDialogButtonBox.Cancel,
            QtCore.Qt.Horizontal, dialog
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)

        layout = QtGui.QVBoxLayout()
        layout.addWidget(label)
        layout.addWidget(text_le)
        layout.addWidget(ok_cb)
        layout.addWidget(buttons)
        dialog.setLayout(layout)

        result = dialog.exec_()

        return (text_le.text(), ok_cb.isChecked()), \
               result == QtGui.QDialog.Accepted


class MainGUI(QtGui.QMainWindow):
    """The main GUI."""
    def __init__(self, screen_size=None):
        """Initialization."""
        super().__init__()

        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.setFixedSize(cfg.WINDOW_WIDTH, cfg.WINDOW_HEIGHT)
        self.setWindowTitle('FXE Azimuthal Integration')

        # drop the oldest element is queue is full
        self._daq_queue = deque(maxlen=cfg.MAX_QUEUE_SIZE)
        # A worker which process the data in another thread
        self._daq_worker = None
        self._geom_file = cfg.DEFAULT_GEOMETRY_FILE

        self.threadpool = QtCore.QThreadPool()

        # *************************************************************
        # Tool bar
        # *************************************************************
        tool_bar = self.addToolBar("Control")

        self._start_at = QtGui.QAction(
            QtGui.QIcon(self.style().standardIcon(QtGui.QStyle.SP_MediaPlay)),
            "Start",
            self)
        tool_bar.addAction(self._start_at)
        self._start_at.triggered.connect(self.on_enter_running)

        self._stop_at = QtGui.QAction(
            QtGui.QIcon(self.style().standardIcon(QtGui.QStyle.SP_MediaStop)),
            "Stop",
            self)
        tool_bar.addAction(self._stop_at)
        self._stop_at.triggered.connect(self.on_exit_running)
        self._stop_at.setEnabled(False)

        self._insert_image_at = QtGui.QAction(
            QtGui.QIcon(self.style().standardIcon(QtGui.QStyle.SP_FileDialogListView)),
            "Plot individual pulse",
            self)
        self._insert_image_at.triggered.connect(
            self._show_individual_pulse_dialog)
        tool_bar.addAction(self._insert_image_at)

        self._open_geometry_file_at = QtGui.QAction(
            QtGui.QIcon(self.style().standardIcon(QtGui.QStyle.SP_DriveCDIcon)),
            "Specify geometry file",
            self)
        self._open_geometry_file_at.triggered.connect(
            self._choose_geometry_file)
        tool_bar.addAction(self._open_geometry_file_at)

        # *************************************************************
        # Plots
        # *************************************************************
        self._plot = LinePlotWidget()
        self._image = ImageViewWidget()

        # *************************************************************
        # Other UI
        # *************************************************************

        self._opened_windows_count = 0
        self._opened_windows = dict()  # book keeping opened windows

        self._cw = QtGui.QWidget()
        self.setCentralWidget(self._cw)

        self._pulse_plot_bt = QtGui.QPushButton("Show Individual Pulses")
        self._pulse_plot_bt.clicked.connect(self._open_individual_pulse_window)

        self._src_calibrated_file_rbt = QtGui.QRadioButton("Calibrated (file)")
        self._src_calibrated_rbt = QtGui.QRadioButton("Calibrated (bridge)")
        self._src_assembled_rbt = QtGui.QRadioButton("Assembled (bridge)")
        self._src_processed_rbt = QtGui.QRadioButton("Processed (bridge)")
        self._src_calibrated_rbt.setChecked(True)
        self._src_processed_rbt.setEnabled(False)

        self._is_normalized_cb = QtGui.QCheckBox("Normalize")
        self._is_normalized_cb.setChecked(False)

        self._hostname_le = QtGui.QLineEdit(cfg.DEFAULT_SERVER_ADDR)
        self._hostname_le.returnPressed.connect(self._update_client)

        self._port_le = QtGui.QLineEdit(cfg.DEFAULT_SERVER_PORT)
        self._port_le.returnPressed.connect(self._update_client)

        self._log_window = QtGui.QPlainTextEdit()
        self._log_window.setReadOnly(True)
        self._log_window.setMaximumBlockCount(cfg.MAX_LOGGING)
        logger_font = QtGui.QFont()
        logger_font.setPointSize(cfg.LOGGER_FONT_SIZE)
        self._log_window.setFont(logger_font)
        self._logger = GuiLogger(self._log_window)
        logging.getLogger().addHandler(self._logger)

        self._title = QtGui.QWidget()

        self._ctrl_pannel = QtGui.QWidget()

        self._initCtrlUI()
        self._initUI()

        if screen_size is None:
            self.move(0, 0)
        else:
            self.move(screen_size.width()/2 - cfg.WINDOW_WIDTH/2,
                      screen_size.height()/20)

        self._client_addr = None
        self._client = None
        self._update_client()

        # For real time plot
        self._is_running = False
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self._update)
        self.timer.start(100)

        self.show()

    def _initUI(self):
        layout = QtGui.QGridLayout()

        layout.addWidget(self._image, 0, 0, 4, 2)
        layout.addWidget(self._ctrl_pannel, 0, 2, 4, 3)
        layout.addWidget(self._plot, 4, 0, 3, 5)
        layout.addWidget(self._log_window, 7, 0, 3, 5)

        self._cw.setLayout(layout)

    def _initCtrlUI(self):
        addr_layout = QtGui.QHBoxLayout()
        hostname_lb = QtGui.QLabel("Hostname: ")
        self._hostname_le.setAlignment(QtCore.Qt.AlignCenter)
        self._hostname_le.setFixedHeight(30)
        port_lb = QtGui.QLabel("Port: ")
        self._port_le.setAlignment(QtCore.Qt.AlignCenter)
        self._port_le.setFixedHeight(30)
        addr_layout.addWidget(hostname_lb, 2)
        addr_layout.addWidget(self._hostname_le, 3)
        addr_layout.addWidget(port_lb, 1)
        addr_layout.addWidget(self._port_le, 2)

        data_src_gp = QtGui.QGroupBox("Data source")
        data_src_gp.setStyleSheet(
            'QGroupBox:title {'
                'border: 1px;'
                'subcontrol-origin: margin;'
                'subcontrol-position: top left;'
                'padding-left: 10px;'
                'padding-top: 10px; '
                'margin-top: 0.2em;}'
        )
        layout = QtGui.QVBoxLayout()
        layout.addWidget(self._src_calibrated_file_rbt)
        layout.addWidget(self._src_calibrated_rbt)
        layout.addWidget(self._src_assembled_rbt)
        layout.addWidget(self._src_processed_rbt)
        data_src_gp.setLayout(layout)

        plot_option_gp = QtGui.QGroupBox()
        layout = QtGui.QVBoxLayout()
        layout.addWidget(self._is_normalized_cb)
        plot_option_gp.setLayout(layout)

        self._pulse_plot_bt.setStyleSheet(
            'QPushButton {'
                'color: white;'
                'font: bold;'
                'padding: 5px;'
                'background-color: "#610B4B";}'
        )

        layout = QtGui.QGridLayout()
        layout.addLayout(addr_layout, 0, 0, 1, 2)
        layout.addWidget(plot_option_gp, 1, 0, 1, 1)
        layout.addWidget(data_src_gp, 1, 1, 1, 1)

        self._ctrl_pannel.setLayout(layout)

    def _update(self):
        """"""
        if self._is_running is False:
            return

        # TODO: improve plot updating
        # Use multithreading for plot updating. However, this is not the
        # bottleneck for the performance.

        try:
            data = self._daq_queue.pop()
        except IndexError:
            return

        # clear the old plots
        self._plot.clear_()
        self._image.clear_()
        for w in self._opened_windows.values():
            w.clear()

        # update the plots in the main panel
        t0 = time.perf_counter()
        if data is not None:
            max_intensity = data["intensity"].max()
            for i, intensity in enumerate(data["intensity"]):
                if self._is_normalized_cb.isChecked() is True:
                    # data["intensity"] is also changed, so the plots in
                    # the other windows also become normalized.
                    intensity /= max_intensity

                self._plot.update(data["momentum"], intensity,
                                  pen=mkPen(intColor(i, hues=9, values=5),
                                            width=2))
            self._plot.set_title("Train ID: {}, No. pulses: {}".
                                 format(data["tid"], i+1))
            self._image.update(np.mean(data["image"], axis=0))

        # update the plots in other opened windows
        for w in self._opened_windows.values():
            w.update(data)

        logger.debug("Time for updating the plots: {:.1f} ms"
                     .format(1000 * (time.perf_counter() - t0)))

    def _show_individual_pulse_dialog(self):
        result, ok = InputDialogWithCheckBox.getResult(
            self,
            'Input Dialog',
            'Enter pulse IDs (separated by comma):',
            "Include LPD image")

        if ok is True:
            self._open_individual_pulse_window(result)

    def _open_individual_pulse_window(self, result):
        text = result[0]
        show_image = result[1]
        if not text:
            logger.info("Invalid input! Please specify pulse IDs!")
            return

        try:
            pulse_ids = text.split(",")
            pulse_ids = [int(i.strip()) for i in pulse_ids]
        except ValueError:
            logger.info(
                "Invalid input! Please specify pulse IDs separated by ','.")
            return

        if pulse_ids:
            window_id = "{:06d}".format(self._opened_windows_count)
            w = LinePlotWindow(window_id, pulse_ids,
                               parent=self,
                               show_image=show_image)
            self._opened_windows_count += 1
            self._opened_windows[window_id] = w
            logger.info("Open new window for pulse(s): {}".
                        format(", ".join(str(i) for i in pulse_ids)))
            w.show()
        else:
            logger.info("Please specify the pulse id(s)!")

    def _choose_geometry_file(self):
        self._geom_file = QtGui.QFileDialog.getOpenFileName()[0]

    def remove_window(self, w_id):
        del self._opened_windows[w_id]

    def on_exit_running(self):
        """Actions taken at the beginning of run state."""
        self._is_running = False
        logger.info("DAQ stopped!")

        self._daq_worker.terminate()

        self._open_geometry_file_at.setEnabled(True)
        self._hostname_le.setEnabled(True)
        self._port_le.setEnabled(True)
        self._start_at.setEnabled(True)
        self._stop_at.setEnabled(False)
        self._src_calibrated_file_rbt.setEnabled(True)
        self._src_calibrated_rbt.setEnabled(True)
        self._src_assembled_rbt.setEnabled(True)
#         self._src_processed_rbt.setEnabled(True)

    def on_enter_running(self):
        """Actions taken at the end of run state."""
        self._is_running = True

        self._client = Client(self._client_addr)
        logger.info("Bind to {}".format(self._client_addr))

        if self._src_calibrated_rbt.isChecked() is True:
            data_source = DataSource.CALIBRATED_FILE
        elif self._src_calibrated_rbt.isChecked() is True:
            data_source = DataSource.CALIBRATED
        elif self._src_assembled_rbt.isChecked() is True:
            data_source = DataSource.ASSEMBLED
        else:
            data_source = DataSource.PROCESSED

        try:
            self._daq_worker = DaqWorker(self._client,
                                         self._daq_queue,
                                         data_source,
                                         geom_file=self._geom_file)
        except OSError as e:
            logger.info(e)
            return

        self._daq_worker.start()

        logger.info("DAQ started!")

        self._open_geometry_file_at.setEnabled(False)
        self._hostname_le.setEnabled(False)
        self._port_le.setEnabled(False)
        self._start_at.setEnabled(False)
        self._stop_at.setEnabled(True)
        self._src_calibrated_file_rbt.setEnabled(False)
        self._src_calibrated_rbt.setEnabled(False)
        self._src_assembled_rbt.setEnabled(False)
        self._src_processed_rbt.setEnabled(False)

    def _update_client(self):
        self._client_addr = "tcp://" \
                            + self._hostname_le.text().strip() \
                            + ":" \
                            + self._port_le.text().strip()


def fxe():
    app = QtGui.QApplication(sys.argv)
    screen_size = app.primaryScreen().size()
    ex = MainGUI(screen_size=screen_size)
    app.exec_()
