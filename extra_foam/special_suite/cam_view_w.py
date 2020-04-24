"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from string import Template

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIntValidator
from PyQt5.QtWidgets import QGridLayout, QSplitter

from extra_foam.gui.plot_widgets import HistMixin, ImageViewF, PlotWidgetF
from extra_foam.gui.ctrl_widgets.smart_widgets import (
    SmartBoundaryLineEdit, SmartLineEdit, SmartStringLineEdit
)

from .cam_view_proc import (
    CamViewProcessor, _DEFAULT_N_BINS, _DEFAULT_BIN_RANGE
)
from .special_analysis_base import (
    create_special, QThreadKbClient, _BaseAnalysisCtrlWidgetS,
    _SpecialAnalysisBase
)

_MAX_N_BINS = 999

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

        self.bin_range_le = SmartBoundaryLineEdit(_DEFAULT_BIN_RANGE)
        self.n_bins_le = SmartLineEdit(str(_DEFAULT_N_BINS))
        self.n_bins_le.setValidator(QIntValidator(1, _MAX_N_BINS))

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
            ("Bin range", self.bin_range_le),
            ("# of bins", self.n_bins_le),
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


class CameraViewRoiHist(HistMixin, PlotWidgetF):
    """CameraViewRoiHist class

    Visualize the ROI histogram.
    """
    def __init__(self, *, parent=None):
        super().__init__(parent=parent)

        self._plot = self.plotBar()

        self._title_template = Template(
            f"mean: $mean, median: $median, std: $std")
        self.updateTitle()
        self.setLabel('left', 'Occurence')
        self.setLabel('bottom', 'Pixel value')

    def updateF(self, data):
        """Override."""
        hist, bin_centers, mean, median, std = data['roi_hist']
        if bin_centers is None:
            self.reset()
        else:
            self._plot.setData(bin_centers, hist)
            self.updateTitle(mean, median, std)


@create_special(CamViewCtrlWidget, CamViewProcessor, QThreadKbClient)
class CamViewWindow(_SpecialAnalysisBase):
    """Main GUI for camera view."""

    _title = "Camera view"
    _long_title = "Camera view"

    def __init__(self, topic):
        super().__init__(topic)

        self._view = CameraView(parent=self)
        self._roi_hist = CameraViewRoiHist(parent=self)

        self.initUI()
        self.initConnections()
        self.startWorker()

    def initUI(self):
        """Override."""
        right_panel = QSplitter(Qt.Vertical)
        right_panel.addWidget(self._view)
        right_panel.addWidget(self._roi_hist)
        right_panel.setSizes([3 * self._TOTAL_H / 4, self._TOTAL_H / 4])

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

        self._ctrl_widget_st.bin_range_le.value_changed_sgn.connect(
            self._worker_st.onBinRangeChanged)
        self._ctrl_widget_st.bin_range_le.returnPressed.emit()

        self._ctrl_widget_st.n_bins_le.value_changed_sgn.connect(
            self._worker_st.onNoBinsChanged)
        self._ctrl_widget_st.n_bins_le.returnPressed.emit()
