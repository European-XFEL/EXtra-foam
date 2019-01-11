"""
Offline and online data analysis and visualization tool for Centre  of
mass analysis from different data acquired with various detectors at
European XFEL.

Main Bragg GUI.

Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import os

from .widgets.pyqtgraph import QtGui
from .widgets import AnalysisCtrlWidget, DataCtrlWidget, GeometryCtrlWidget
from .windows import BraggSpotsWindow
from .main_gui import MainGUI


class MainBraggGUI(MainGUI):
    """The main GUI for azimuthal integration."""

    _height = 600  # window height, in pixel
    _width = 1200  # window width, in pixel

    def __init__(self, *args, **kwargs):
        """Initialization."""
        super().__init__(*args, **kwargs)

        # *************************************************************
        # Tool bar
        # *************************************************************
        #
        open_bragg_spots_window_at = QtGui.QAction(
            QtGui.QIcon(os.path.join(self._root_dir, "icons/bragg_spots.png")),
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
        self.data_ctrl_widget = DataCtrlWidget(parent=self)
        self._ctrl_widgets = [
            self.geometry_ctrl_widget, self.analysis_ctrl_widget,
            self.data_ctrl_widget,
        ]

        self.initUI()
        self.initConnection()

        self.show()

    def initConnection(self):
        """Set up all signal and slot connections."""
        super().initConnection()

        self.geometry_ctrl_widget.geometry_sgn.connect(
            self._proc_worker.onGeometryChanged)
        self.analysis_ctrl_widget.photon_energy_sgn.connect(
            self._proc_worker.onPhotonEnergyChanged)
        self.analysis_ctrl_widget.mask_range_sgn.connect(
            self._proc_worker.onMaskRangeChanged)

    def initUI(self):
        layout = QtGui.QGridLayout()

        layout.addWidget(self.geometry_ctrl_widget, 0, 0, 4, 1)
        layout.addWidget(self.analysis_ctrl_widget, 0, 1, 4, 1)
        layout.addWidget(self.data_ctrl_widget, 0, 2, 7, 1)
        layout.addWidget(self._logger.widget, 4, 0, 3, 2)
        self._cw.setLayout(layout)
