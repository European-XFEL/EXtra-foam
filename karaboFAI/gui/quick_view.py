"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

FaiQuickView.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import argparse
import os.path as osp
from queue import Empty, Full

from .pyqtgraph import QtCore, QtGui

from .ctrl_widgets import GeometryCtrlWidget
from .. import __version__
from ..config import config
from ..logger import logger
from ..pipeline import (
    AssemblingError, Data4Visualization, ImageAssemblerFactory,
    QThreadBridge, QThreadWorker
)


class _QuickViewImageAssembler(QThreadWorker):
    def __init__(self):
        super().__init__()
        self._image_assembler = ImageAssemblerFactory.create(config['DETECTOR'])

    def run(self):
        """Run the data processor."""
        self.empty_output()  # remove old data

        timeout = config['TIMEOUT']
        self.info("Scheduler started!")
        while not self.isInterruptionRequested():
            try:
                data = self._input.get(timeout=timeout)
            except Empty:
                continue

            raw, meta = data
            tid = next(iter(meta.values()))["timestamp.tid"]

            try:
                assembled = self._image_assembler.assemble(raw)
            except AssemblingError as e:
                self.error(f"Train ID: {tid}: " + repr(e))
                continue
            except Exception as e:
                self.error(f"Unexpected Exception: Train ID: {tid}: " + repr(e))
                raise

            # always keep the latest data in the queue
            try:
                self._output.put(assembled, timeout=timeout)
            except Full:
                self.pop_output()
                self.debug("Data dropped by the scheduler")

        self.info("Scheduler stopped!")


class FaiQuickView(QtGui.QMainWindow):
    """FaiQuickView class.

    A QMainWindow which only shows the image.
    """

    _root_dir = osp.dirname(osp.abspath(__file__))

    start_sgn = QtCore.pyqtSignal()
    stop_sgn = QtCore.pyqtSignal()

    closed_sgn = QtCore.pyqtSignal()

    def __init__(self):
        """Initialization."""
        super().__init__()

        self._pulse_resolved = config["PULSE_RESOLVED"]

        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)

        self.title = f"karaboFAI-view {__version__} ({config['DETECTOR']})"
        self.setWindowTitle(self.title)

        self._cw = QtGui.QWidget()
        self.setCentralWidget(self._cw)

        self._bridge = QThreadBridge()
        self._assembler = _QuickViewImageAssembler()
        self._assembler.connect_input(self._bridge.output)
        self._input = self._assembler.output

        # *************************************************************
        # Tool bar
        # *************************************************************
        self._tool_bar = self.addToolBar("Control")

        self._start_at = self._addAction("Start bridge", "start.png")
        self._start_at.triggered.connect(self.onStart)

        self._stop_at = self._addAction("Stop bridge", "stop.png")
        self._stop_at.triggered.connect(self.onStop)
        self._stop_at.setEnabled(False)

        # *************************************************************
        # Miscellaneous
        # *************************************************************

        self._data = Data4Visualization()

        # For real time plot
        self._running = False
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.updateAll)
        self.timer.start(config["TIMER_INTERVAL"])

        # *************************************************************
        # control widgets
        # *************************************************************

        if config['REQUIRE_GEOMETRY']:
            self.geometry_ctrl_widget = GeometryCtrlWidget(parent=self)

        self.initUI()

        self.setFixedHeight(self.minimumSizeHint().height())

        self.show()

    def initUI(self):
        layout = QtGui.QVBoxLayout()
        if config['REQUIRE_GEOMETRY']:
            layout.addWidget(self.geometry_ctrl_widget)

        self._cw.setLayout(layout)

    def updateAll(self):
        """Update all the plots in the main and child windows."""
        if not self._running:
            return

        # TODO: improve plot updating
        # Use multithreading for plot updating. However, this is not the
        # bottleneck for the performance.

        try:
            self._data.set(self._input.get_nowait())
        except Empty:
            return

        if self._data.get().image is None:
            logger.info("Bad train with ID: {}".format(self._data.get().tid))
            return

        self._updateAllPlots()

        logger.info("Updated train with ID: {}".format(self._data.get().tid))

    def onStart(self):
        if not self.updateMetaData():
            return

        self.start_sgn.emit()

    def onBridgeStarted(self):
        """Actions taken before the start of a 'run'."""
        self._running = True  # starting to update plots

        self._start_at.setEnabled(False)
        self._stop_at.setEnabled(True)

        for widget in self._ctrl_widgets:
            widget.onBridgeStarted()

    def onBridgeStopped(self):
        """Actions taken before the end of a 'run'."""
        self._running = False

        self._start_at.setEnabled(True)
        self._stop_at.setEnabled(False)

        for widget in self._ctrl_widgets:
            widget.onBridgeStopped()

    def updateMetaData(self):
        """Update metadata from all the ctrl widgets.

        :returns bool: True if all metadata successfully parsed
            and emitted, otherwise False.
        """
        for widget in self._ctrl_widgets:
            succeeded = widget.updateMetaData()
            if not succeeded:
                return False
        return True

    @QtCore.pyqtSlot(str)
    def onDebugReceived(self, msg):
        logger.debug(msg)

    @QtCore.pyqtSlot(str)
    def onInfoReceived(self, msg):
        logger.info(msg)

    @QtCore.pyqtSlot(str)
    def onWarningReceived(self, msg):
        logger.warning(msg)

    @QtCore.pyqtSlot(str)
    def onErrorReceived(self, msg):
        logger.error(msg)

    def closeEvent(self, QCloseEvent):
        self.closed_sgn.emit()

        super().closeEvent(QCloseEvent)


def application():
    parser = argparse.ArgumentParser(prog="karaboFAI-view")
    parser.add_argument('-V', '--version', action='version',
                        version="%(prog)s " + __version__)
    parser.add_argument("detector", help="detector name (case insensitive)",
                        choices=[det.upper() for det in config.detectors],
                        type=lambda s: s.upper())
    parser.add_argument('--debug', action='store_true',
                        help="Run in debug mode")

    args = parser.parse_args()

    if args.debug:
        import faulthandler
        faulthandler.enable()
        logger.debug("'faulthandler enabled")
    else:
        logger.setLevel("INFO")

    detector = args.detector
    if detector == 'JUNGFRAU':
        detector = 'JungFrau'
    elif detector == 'FASTCCD':
        detector = 'FastCCD'
    elif detector == 'BASLERCAMERA':
        detector = 'BaslerCamera'
    else:
        detector = detector.upper()

    server = FaiQuickView(detector)

    server.start()


if __name__ == "__main__":

    application()
