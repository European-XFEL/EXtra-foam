import sys
import logging
import time

import numpy as np

from PyQt5 import QtCore, QtGui
from PyQt5.QtWidgets import (
    QMainWindow,
    QApplication, QWidget, QGridLayout, QLabel, QPlainTextEdit,
    QTextEdit, QSlider, QHBoxLayout
)
from PyQt5.QtCore import Qt

from karabo_bridge import Client

import pyqtgraph as pg
from logger import log, GuiLogger
from data_processing import process_data
from buttons import RunButton
from plots import LinePlotWidget, ImageViewWidget


WINDOW_HEIGHT = 900
WINDOW_WIDTH = 600

MAX_LOGGING = 1000

LOGGER_FONT = QtGui.QFont()
LOGGER_FONT.setPointSize(10)


class MainGUI(QMainWindow):
    """The main GUI."""
    def __init__(self, screen_size=None):
        """Initialization."""
        super().__init__()

        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.setFixedSize(WINDOW_WIDTH, WINDOW_HEIGHT)
        self.setWindowTitle('FXE')

        self._cw = QWidget()
        self.setCentralWidget(self._cw)

        self._btn_run = RunButton()
        self._btn_run.attach(self)
        self._btn_run.clicked.connect(self._btn_run.update)

        self._log_window = QPlainTextEdit()
        self._log_window.setReadOnly(True)
        self._log_window.setMaximumBlockCount(MAX_LOGGING)
        self._log_window.setFont(LOGGER_FONT)
        self._logger = GuiLogger(self._log_window)
        logging.getLogger().addHandler(self._logger)

        self._param1 = None
        self._param2 = None
        self._param3 = None
        self._param4 = None

        self._title = QWidget()
        self._image = ImageViewWidget(280, 280)
        self._lines = LinePlotWidget(WINDOW_WIDTH - 40, 400)
        self._ctrl_pannel = QWidget()
        self.initTitleUI()
        self.initCtrlUI()
        self.initUI()

        if screen_size is None:
            self.move(0, 0)
        else:
            self.move(screen_size.width()/2 - WINDOW_WIDTH/2,
                      screen_size.height()/20)

        self.client = Client("tcp://localhost:1236")

        # For real time plot
        self.is_running = False
        self.timer = pg.QtCore.QTimer()
        self.timer.timeout.connect(self._update)
        self.timer.start(200)

        self.show()

    def initUI(self):
        layout = QGridLayout()
        # 1st row
        layout.addWidget(self._title, 0, 0, 1, 6)
        # 2nd row
        layout.addWidget(self._ctrl_pannel, 1, 3, 3, 3)
        layout.addWidget(self._image, 1, 0, 3, 3)
        # 3rd row
        layout.addWidget(self._lines, 4, 0, 3, 6)
        # 4th row
        layout.addWidget(self._log_window, 7, 0, 3, 6)

        self._cw.setLayout(layout)

    def initTitleUI(self):
        title = QLabel()
        title.setText("LPD Azimuthal Integration")
        title.setAlignment(Qt.AlignCenter)

        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        font.setPointSize(16)

        title.setFont(font)

        layout = QHBoxLayout()
        layout.addWidget(title)

        self._title.setLayout(layout)

    def initCtrlUI(self):
        self._btn_run.setFixedSize(100, 30)

        param1 = QLabel("Param1")
        param1_edit = QTextEdit()
        param1_edit.setFixedSize(100, 30)

        param2 = QLabel("Param2")
        param2_edit = QTextEdit()
        param2_edit.setFixedSize(100, 30)

        param3 = QLabel("Param3")
        param3_slider = QSlider(Qt.Horizontal)
        param3_slider.setTickPosition(QSlider.TicksBelow)
        param3_slider.setFixedSize(100, 30)

        param4 = QLabel("Param4")
        param4_slider = QSlider(Qt.Horizontal)
        param4_slider.setTickPosition(QSlider.TicksBelow)
        param4_slider.setFixedSize(100, 30)

        layout = QGridLayout()
        layout.addWidget(self._btn_run, 0, 0, 1, 2)
        layout.addWidget(param1, 1, 0)
        layout.addWidget(param1_edit, 1, 1)
        layout.addWidget(param2, 2, 0)
        layout.addWidget(param2_edit, 2, 1)
        layout.addWidget(param3, 3, 0)
        layout.addWidget(param3_slider, 3, 1)
        layout.addWidget(param4, 4, 0)
        layout.addWidget(param4_slider, 4, 1)

        self._ctrl_pannel.setLayout(layout)

    def _update(self):
        """"""
        if self.is_running is False:
            return

        # retrieve
        t0 = time.perf_counter()
        kb_data = self.client.next()
        log.info("Time for retrieving data from the server: {:.1f} ms"
                 .format(1000 * (time.perf_counter() - t0)))

        # process
        t0 = time.perf_counter()
        data = process_data(kb_data)
        log.info("Time for processing the data: {:.1f} ms"
                 .format(1000 * (time.perf_counter() - t0)))
        if data is None:
            return

        # show
        t0 = time.perf_counter()

        self._lines.p1.clear()
        for i, intensity in enumerate(data["intensity"]):
            self._lines.p1.plot(data["momentum"], intensity)

        self._image.set_image(np.nan_to_num(data["images"][0]))

        log.info("Time for updating the plots: {:.1f} ms"
                 .format(1000 * (time.perf_counter() - t0)))

        self.first_loop = False


def main_gui():
    app = QApplication(sys.argv)
    screen_size = app.primaryScreen().size()
    ex = MainGUI(screen_size=screen_size)
    app.exec_()


if __name__ == "__main__":
    main_gui()
