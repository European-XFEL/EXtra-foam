"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

Main FAI GUI.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import os
import sys
import argparse

from .widgets.pyqtgraph import QtGui
from .widgets import (
    AiCtrlWidget, AnalysisCtrlWidget, DataCtrlWidget, GeometryCtrlWidget
)
from .windows import LaserOnOffWindow, OverviewWindow
from .main_gui import MainGUI


class MainFaiGUI(MainGUI):
    """The main GUI for azimuthal integration."""
    _height = 700  # window height, in pixel
    _width = 1200  # window width, in pixel

    def __init__(self, *args, **kwargs):
        """Initialization."""
        super().__init__(*args, **kwargs)

        # *************************************************************
        # Tool bar
        # *************************************************************
        #
        open_overview_window_at = QtGui.QAction(
            QtGui.QIcon(os.path.join(self._root_dir, "icons/overview.png")),
            "Overview",
            self)
        open_overview_window_at.triggered.connect(
            lambda: OverviewWindow(self._data, parent=self))
        self._tool_bar.addAction(open_overview_window_at)

        #
        open_laseronoff_window_at = QtGui.QAction(
            QtGui.QIcon(os.path.join(self._root_dir, "icons/on_off_pulses.png")),
            "On- and off- pulses",
            self)
        open_laseronoff_window_at.triggered.connect(
            lambda: LaserOnOffWindow(self._data, parent=self))
        self._tool_bar.addAction(open_laseronoff_window_at)

        # *************************************************************
        # control widgets
        # *************************************************************

        self.ai_ctrl_widget = AiCtrlWidget(parent=self)
        self.geometry_ctrl_widget = GeometryCtrlWidget(parent=self)
        self.analysis_ctrl_widget = AnalysisCtrlWidget(parent=self)
        self.data_ctrl_widget = DataCtrlWidget(parent=self)
        self._ctrl_widgets = [
            self.ai_ctrl_widget, self.geometry_ctrl_widget,
            self.analysis_ctrl_widget, self.data_ctrl_widget,
        ]

        self.initUI()
        self.initConnection()

        self.show()

    def initConnection(self):
        """Set up all signal and slot connections."""
        super().initConnection()

        self.geometry_ctrl_widget.geometry_sgn.connect(
            self._proc_worker.onGeometryChanged)
        self.ai_ctrl_widget.sample_distance_sgn.connect(
            self._proc_worker.onSampleDistanceChanged)
        self.ai_ctrl_widget.center_coordinate_sgn.connect(
            self._proc_worker.onCenterCoordinateChanged)
        self.ai_ctrl_widget.integration_method_sgn.connect(
            self._proc_worker.onIntegrationMethodChanged)
        self.ai_ctrl_widget.integration_range_sgn.connect(
            self._proc_worker.onIntegrationRangeChanged)
        self.ai_ctrl_widget.integration_points_sgn.connect(
            self._proc_worker.onIntegrationPointsChanged)
        self.analysis_ctrl_widget.photon_energy_sgn.connect(
            self._proc_worker.onPhotonEnergyChanged)
        self.analysis_ctrl_widget.mask_range_sgn.connect(
            self._proc_worker.onMaskRangeChanged)

    def initUI(self):
        layout = QtGui.QGridLayout()

        layout.addWidget(self.ai_ctrl_widget, 0, 0, 3, 2)
        layout.addWidget(self.analysis_ctrl_widget, 0, 2, 3, 2)
        layout.addWidget(self.data_ctrl_widget, 0, 4, 3, 2)

        layout.addWidget(self._logger.widget, 3, 0, 1, 3)
        layout.addWidget(self.geometry_ctrl_widget, 3, 3, 1, 3)

        self._cw.setLayout(layout)


def main_fai_gui():
    parser = argparse.ArgumentParser(prog="karaboFAI")
    parser.add_argument("detector", help="detector name")

    args = parser.parse_args()

    valid_detectors = ['AGIPD', 'LPD', 'JUNGFRAU']
    if args.detector.upper() not in valid_detectors:
        raise ValueError("Unknown detector. Valid options are: {}.".
                         format(valid_detectors))

    app = QtGui.QApplication(sys.argv)
    screen_size = app.primaryScreen().size()
    ex = MainFaiGUI(args.detector.upper(), screen_size=screen_size)
    app.exec_()
