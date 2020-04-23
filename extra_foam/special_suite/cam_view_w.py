"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIntValidator
from PyQt5.QtWidgets import QGridLayout, QSplitter

from .cam_view_proc import CamViewProcessor
from .special_analysis_base import (
    create_special, QThreadKbClient, _BaseAnalysisCtrlWidgetS,
    _SpecialAnalysisBase
)
from ..gui.plot_widgets import ImageViewF, PlotWidgetF
from ..gui.ctrl_widgets.smart_widgets import (
    SmartBoundaryLineEdit, SmartLineEdit, SmartStringLineEdit
)


# a non-empty place holder
_DEFAULT_OUTPUT_CHANNEL = "camera:output"
# default is for Basler camera
_DEFAULT_PROPERTY = "data.image.pixels"


class CamViewCtrlWidget(_BaseAnalysisCtrlWidgetS):
    """Camera view control widget."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.output_ch_le = SmartStringLineEdit(_DEFAULT_OUTPUT_CHANNEL)
        self.property_le = SmartStringLineEdit(_DEFAULT_PROPERTY)

        self.ma_window_le = SmartLineEdit("1")
        validator = QIntValidator()
        validator.setBottom(1)
        self.ma_window_le.setValidator(validator)

        self._non_reconfigurable_widgets = [
            self.output_ch_le,
            self.property_le,
        ]

        self.initUI()
        self.initConnections()

    def initUI(self):
        """Override."""
        layout = QGridLayout()

        self.addRows(layout, [
            ("Output channel", self.output_ch_le),
            ("Property", self.property_le),
            ("M.A. window", self.ma_window_le),
        ])

        self.setLayout(layout)

    def initConnections(self):
        """Override."""
        pass


class CameraView(ImageViewF):
    """CameraView class.

    Visualize the camera image.
    """
    def __init__(self, *, parent=None):
        """Initialization."""
        super().__init__(has_roi=True, parent=parent)

    def updateF(self, data):
        """Override."""
        self.setImage(data['displayed'])


@create_special(CamViewCtrlWidget, CamViewProcessor, QThreadKbClient)
class CamViewWindow(_SpecialAnalysisBase):
    """Main GUI for camera view."""

    _title = "Camera view"
    _long_title = "Camera view"

    def __init__(self, topic):
        super().__init__(topic)

        self._view = CameraView(parent=self)

        self.initUI()
        self.initConnections()
        self.startWorker()

    def initUI(self):
        """Override."""
        right_panel = QSplitter(Qt.Vertical)
        right_panel.addWidget(self._view)

        cw = self.centralWidget()
        cw.addWidget(right_panel)
        cw.setSizes([self._TOTAL_W / 4, 3 * self._TOTAL_W / 4])

        self.resize(self._TOTAL_W, self._TOTAL_H)

    def initConnections(self):
        """Override."""
        self._ctrl_widget_st.output_ch_le.value_changed_sgn.connect(
            self._worker_st.onOutputChannelChanged)
        self._ctrl_widget_st.output_ch_le.returnPressed.emit()

        self._ctrl_widget_st.property_le.value_changed_sgn.connect(
            self._worker_st.onPropertyChanged)
        self._ctrl_widget_st.property_le.returnPressed.emit()

        self._ctrl_widget_st.ma_window_le.value_changed_sgn.connect(
            self._worker_st.onMaWindowChanged)
        self._ctrl_widget_st.ma_window_le.returnPressed.emit()
