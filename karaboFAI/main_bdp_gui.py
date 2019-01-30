"""
Offline and online data analysis and visualization tool for Centre  of
mass analysis from different data acquired with various detectors at
European XFEL.

Main Bragg diffraction peak GUI.

Author: Ebad Kamil <ebad.kamil@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import os
import sys
import argparse

from .data_processing import BdpDataProcessor as DataProcessor
from .widgets.pyqtgraph import QtGui
from .widgets import (
    AnalysisCtrlWidget, DataCtrlWidget, GeometryCtrlWidget,
    PumpProbeCtrlWidget
)
from .windows import BraggSpotsWindow
from .main_gui import MainGUI


class MainBdpGUI(MainGUI):
    """The main GUI for azimuthal integration."""

    def __init__(self, *args, **kwargs):
        """Initialization."""
        super().__init__(*args, **kwargs)

        # *************************************************************
        # Tool bar
        # *************************************************************
        #
        open_bragg_spots_window_at = QtGui.QAction(
            QtGui.QIcon(os.path.join(self._root_dir, "icons/bragg_diffraction_peak.png")),
            "Bragg spots",
            self)
        open_bragg_spots_window_at.triggered.connect(
            lambda: BraggSpotsWindow(self._data, parent=self))
        self._tool_bar.addAction(open_bragg_spots_window_at)

        # *************************************************************
        # control widgets
        # *************************************************************

        self.geometry_ctrl_widget = GeometryCtrlWidget(parent=self)
        self.analysis_ctrl_widget = AnalysisCtrlWidget(parent=self)
        self.pump_probe_ctrl_widget = PumpProbeCtrlWidget(parent=self)
        self.data_ctrl_widget = DataCtrlWidget(parent=self)

        self._proc_worker = DataProcessor(self._daq_queue, self._proc_queue)

        self.initUI()
        self.initConnection()

        self.show()

    def initConnection(self):
        """Set up all signal and slot connections."""
        super().initConnection()

        self.geometry_ctrl_widget.geometry_sgn.connect(
            self._proc_worker.onGeometryChanged)

        self.analysis_ctrl_widget.pulse_id_range_sgn.connect(
            self._proc_worker.onPulseRangeChanged)
        self.analysis_ctrl_widget.image_mask_range_sgn.connect(
            self._proc_worker.onMaskRangeChanged)

    def initUI(self):
        layout = QtGui.QVBoxLayout()

        layout1 = QtGui.QHBoxLayout()
        layout1.addWidget(self.geometry_ctrl_widget)
        layout1.addWidget(self.analysis_ctrl_widget)
        layout1.addWidget(self.pump_probe_ctrl_widget)
        layout1.addWidget(self.data_ctrl_widget)

        layout2 = QtGui.QHBoxLayout()
        layout2.addWidget(self._logger.widget)

        layout.addLayout(layout1)
        layout.addLayout(layout2)
        self._cw.setLayout(layout)


def main_bdp_gui():
    parser = argparse.ArgumentParser(prog="karaboBDP")
    parser.add_argument("detector", help="detector name (case insensitive)",
                        choices=['AGIPD', 'LPD', 'JUNGFRAU'],
                        type=lambda s: s.upper())

    args = parser.parse_args()

    detector = args.detector
    if detector == 'JUNGFRAU':
        detector = 'JungFrau'
    else:
        detector = detector.upper()

    app = QtGui.QApplication(sys.argv)
    screen_size = app.primaryScreen().size()
    ex = MainBdpGUI(detector, screen_size=screen_size)
    app.exec_()
