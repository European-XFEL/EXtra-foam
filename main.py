import sys
import logging
import time

import numpy as np

from PyQt5 import QtCore, QtGui
from PyQt5.QtWidgets import (
    QMainWindow, QGroupBox,
    QApplication, QWidget, QGridLayout, QLabel, QPlainTextEdit,
    QTextEdit, QSlider, QRadioButton, QHBoxLayout
)
from PyQt5.QtCore import Qt

from karabo_bridge import Client

import pyqtgraph as pg
from logger import log, GuiLogger
from widgets import RunButton, PlotButton
from plots import LinePlotWidget, ImageViewWidget, LinePlotWindow
from data_acquisition import acquire_data
import config as cfg


class MainGUI(QMainWindow):
    """The main GUI."""
    def __init__(self, screen_size=None):
        """Initialization."""
        super().__init__()

        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.setFixedSize(cfg.WINDOW_WIDTH, cfg.WINDOW_HEIGHT)
        self.setWindowTitle('FXE')

        self._image = ImageViewWidget(280, 280)
        self._plot = LinePlotWidget(2, size=(cfg.WINDOW_WIDTH - 40, 200))

        self._cw = QWidget()
        self.setCentralWidget(self._cw)

        self._btn_run = None
        self._btn_plt = None
        self._show_image_rbtn = QRadioButton()

        self._log_window = QPlainTextEdit()
        self._log_window.setReadOnly(True)
        self._log_window.setMaximumBlockCount(cfg.MAX_LOGGING)
        logger_font = QtGui.QFont()
        logger_font.setPointSize(cfg.LOGGER_FONT_SIZE)
        self._log_window.setFont(logger_font)
        self._logger = GuiLogger(self._log_window)
        logging.getLogger().addHandler(self._logger)

        self._pulses_to_show = QTextEdit()
        self._param1 = None
        self._param2 = None
        self._param3 = None

        self._title = QWidget()
        self._ctrl_pannel = QWidget()
        self.initTitleUI()
        self.initCtrlUI()
        self.initUI()

        if screen_size is None:
            self.move(0, 0)
        else:
            self.move(screen_size.width()/2 - cfg.WINDOW_WIDTH/2,
                      screen_size.height()/20)

        self._client = Client("tcp://localhost:1236")

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
        layout.addWidget(self._plot, 4, 0, 3, 6)
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
        self._btn_run = RunButton()
        self._btn_run.attach(self)
        self._btn_run.clicked.connect(self._btn_run.update)

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

        pulses_to_show_gp = QGroupBox()

        self._btn_plt = PlotButton("Individual Pulses")
        self._btn_plt.clicked.connect(self._new_window)

        self._pulses_to_show.setFixedSize(100, 30)

        self._show_image_rbtn.setText("Include LPD Image")
        self._show_image_rbtn.setChecked(True)

        gp_layout = QGridLayout()
        gp_layout.addWidget(self._btn_plt, 0, 0, 1, 2)
        gp_layout.addWidget(self._pulses_to_show, 0, 2, 1, 3)
        gp_layout.addWidget(self._show_image_rbtn, 1, 0, 1, 3)
        pulses_to_show_gp.setLayout(gp_layout)

        layout = QGridLayout()
        layout.addWidget(self._btn_run, 0, 0, 1, 2)
        layout.addWidget(param1, 1, 0)
        layout.addWidget(param1_edit, 1, 1)
        layout.addWidget(param2, 2, 0)
        layout.addWidget(param2_edit, 2, 1)
        layout.addWidget(param3, 3, 0)
        layout.addWidget(param3_slider, 3, 1)
        layout.addWidget(pulses_to_show_gp, 4, 0, 1, 2)

        self._ctrl_pannel.setLayout(layout)

    def _update(self):
        """"""
        if self.is_running is False:
            return

        data = acquire_data(self._client)
        if data is None:
            return

        # show
        t0 = time.perf_counter()

        p1 = self._plots.plot_items[0]
        p1.clear()
        for i, intensity in enumerate(data["intensity"]):
            p1.plot(data["momentum"], intensity)

        self._image.set_image(np.nan_to_num(data["images"][0]))

        log.info("Time for updating the plots: {:.1f} ms"
                 .format(1000 * (time.perf_counter() - t0)))

        self.first_loop = False

    def _new_window(self):
        text = self._pulses_to_show.toPlainText()
        pulse_ids = []
        try:
            if text:
                pulse_ids = text.split(",")
                pulse_ids = [int(i.strip()) for i in pulse_ids]
        except ValueError:
            log.error(
                "Invalid input! Please specify pulse ids separated by ','.")
            return

        if pulse_ids:
            w = LinePlotWindow(len(pulse_ids),
                               parent=self,
                               show_image=self._show_image_rbtn.isChecked())
            log.info("Open line plots for pulse(s): {}".
                     format(", ".join(str(i) for i in pulse_ids)))
            w.show()
        else:
            log.info("Please specify the pulse id(s)!")


def main_gui():
    app = QApplication(sys.argv)
    screen_size = app.primaryScreen().size()
    ex = MainGUI(screen_size=screen_size)
    app.exec_()


if __name__ == "__main__":
    main_gui()
