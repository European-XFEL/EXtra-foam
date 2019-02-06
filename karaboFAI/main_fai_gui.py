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

from .data_processing import DataProcessor
from .widgets.pyqtgraph import QtGui
from .widgets import (
    AiCtrlWidget, AnalysisCtrlWidget, CorrelationCtrlWidget, DataCtrlWidget,
    GeometryCtrlWidget, PumpProbeCtrlWidget
)
from .windows import (
    LaserOnOffWindow, OverviewWindow, OverviewWindowTrainResolved
)
from .main_gui import MainGUI
from .config import config


class MainFaiGUI(MainGUI):
    """The main GUI for azimuthal integration."""

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
        if self._pulse_resolved:
            open_overview_window_at.triggered.connect(
                lambda: OverviewWindow(self._data, parent=self))
        else:
            open_overview_window_at.triggered.connect(
                lambda: OverviewWindowTrainResolved(
                    self._data, mediator=self._mediator, parent=self))

        self._tool_bar.addAction(open_overview_window_at)

        #
        open_laseronoff_window_at = QtGui.QAction(
            QtGui.QIcon(os.path.join(self._root_dir, "icons/on_off_pulses.png")),
            "On- and off- pulses",
            self)
        open_laseronoff_window_at.triggered.connect(
            lambda: LaserOnOffWindow(self._data,
                                     parent=self,
                                     pulse_resolved=self._pulse_resolved))
        self._tool_bar.addAction(open_laseronoff_window_at)

        # *************************************************************
        # control widgets
        # *************************************************************

        self.ai_ctrl_widget = AiCtrlWidget(parent=self)
        if config['REQUIRE_GEOMETRY']:
            self.geometry_ctrl_widget = GeometryCtrlWidget(parent=self)

        self.analysis_ctrl_widget = AnalysisCtrlWidget(
            parent=self, pulse_resolved=self._pulse_resolved)
        self.correlation_ctrl_widget = CorrelationCtrlWidget(
            parent=self, pulse_resolved=self._pulse_resolved)
        self.pump_probe_ctrl_widget = PumpProbeCtrlWidget(
            parent=self, pulse_resolved=self._pulse_resolved)

        self.data_ctrl_widget = DataCtrlWidget(
            parent=self, pulse_resolved=self._pulse_resolved)

        self._proc_worker = DataProcessor(self._daq_queue, self._proc_queue)

        self.initUI()
        self.initConnection()

        self.resize(self.sizeHint().width(), self.minimumSizeHint().height())

        self.show()

    def initConnection(self):
        """Set up all signal and slot connections."""
        super().initConnection()

        if config['REQUIRE_GEOMETRY']:
            self.geometry_ctrl_widget.geometry_sgn.connect(
                self._proc_worker.onGeometryChanged)

        self.ai_ctrl_widget.photon_energy_sgn.connect(
            self._proc_worker.onPhotonEnergyChanged)
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

        self.analysis_ctrl_widget.pulse_id_range_sgn.connect(
            self._proc_worker.onPulseRangeChanged)

        self._mediator.roi_hist_clear_sgn.connect(
            self._proc_worker.onRoiHistClear)
        self._mediator.roi1_region_changed_sgn.connect(
            self._proc_worker.onRoi1Changed)
        self._mediator.roi2_region_changed_sgn.connect(
            self._proc_worker.onRoi2Changed)
        self._mediator.threshold_mask_change_sgn.connect(
            self._proc_worker.onThresholdMaskChange)

        self.analysis_ctrl_widget.enable_ai_cb.stateChanged.connect(
            self._proc_worker.onEnableAiStateChange)

    def initUI(self):
        misc_layout = QtGui.QHBoxLayout()
        misc_layout.addWidget(self.ai_ctrl_widget)
        if config['REQUIRE_GEOMETRY']:
            misc_layout.addWidget(self.geometry_ctrl_widget)
        misc_layout.addWidget(self.data_ctrl_widget)

        right_layout = QtGui.QVBoxLayout()
        right_layout.addLayout(misc_layout)
        right_layout.addWidget(self._logger.widget)

        analysis_layout = QtGui.QVBoxLayout()
        analysis_layout.addWidget(self.analysis_ctrl_widget)
        analysis_layout.addWidget(self.correlation_ctrl_widget)
        analysis_layout.addWidget(self.pump_probe_ctrl_widget)

        layout = QtGui.QHBoxLayout()
        layout.addLayout(analysis_layout)
        layout.addLayout(right_layout)
        self._cw.setLayout(layout)


def main_fai_gui():
    parser = argparse.ArgumentParser(prog="karaboFAI")
    parser.add_argument("detector", help="detector name (case insensitive)",
                        choices=['AGIPD', 'LPD', 'JUNGFRAU', 'FASTCCD'],
                        type=lambda s: s.upper())

    args = parser.parse_args()

    detector = args.detector
    if detector == 'JUNGFRAU':
        detector = 'JungFrau'
    elif detector == 'FASTCCD':
        detector = 'FastCCD'
    else:
        detector = detector.upper()

    app = QtGui.QApplication(sys.argv)
    screen_size = app.primaryScreen().size()
    ex = MainFaiGUI(detector, screen_size=screen_size)
    app.exec_()
