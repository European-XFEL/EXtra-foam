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
from PyQt5.QtWidgets import QCheckBox, QGridLayout, QSplitter

from extra_foam.gui.plot_widgets import (
    HistMixin, ImageViewF, PlotWidgetF
)
from extra_foam.gui.misc_widgets import FColor
from extra_foam.gui.ctrl_widgets.smart_widgets import (
    SmartBoundaryLineEdit, SmartLineEdit, SmartSliceLineEdit,
    SmartStringLineEdit
)

from .gotthard_proc import (
    GotthardProcessor, _DEFAULT_BIN_RANGE, _DEFAULT_N_BINS
)
from .special_analysis_base import (
    create_special, QThreadKbClient, _BaseAnalysisCtrlWidgetS,
    _SpecialAnalysisBase
)
from .config import _MAX_N_GOTTHARD_PULSES, GOTTHARD_DEVICE

_MAX_N_BINS = 999


class GotthardCtrlWidget(_BaseAnalysisCtrlWidgetS):
    """Gotthard analysis control widget."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.output_ch_le = SmartStringLineEdit(
            GOTTHARD_DEVICE.get(self.topic, "Gotthard:output"))

        self.ma_window_le = SmartLineEdit("1")
        validator = QIntValidator()
        validator.setBottom(1)
        self.ma_window_le.setValidator(validator)

        self.pulse_slicer_le = SmartSliceLineEdit(":")

        self.poi_index_le = SmartLineEdit("0")
        self.poi_index_le.setValidator(
            QIntValidator(0, _MAX_N_GOTTHARD_PULSES - 1))

        self.bin_range_le = SmartBoundaryLineEdit(_DEFAULT_BIN_RANGE)
        self.n_bins_le = SmartLineEdit(str(_DEFAULT_N_BINS))
        self.n_bins_le.setValidator(QIntValidator(1, _MAX_N_BINS))
        self.hist_over_ma_cb = QCheckBox("Histogram over M.A. train")

        self._non_reconfigurable_widgets = [
            self.output_ch_le
        ]

        self.initUI()
        self.initConnections()

    def initUI(self):
        """Override."""
        layout = QGridLayout()

        self.addRows(layout, [
            ("Output channel", self.output_ch_le),
            ("M.A. window", self.ma_window_le),
            ("Pulse slicer ", self.pulse_slicer_le),
            ("P.O.I. (sliced)", self.poi_index_le),
            ("Bin range", self.bin_range_le),
            ("# of bins", self.n_bins_le),
            ("", self.hist_over_ma_cb),
        ])

        self.setLayout(layout)

    def initConnections(self):
        """Override."""
        pass


class GotthardAvgPlot(PlotWidgetF):
    """GotthardAvgPlot class.

    Visualize signals of the averaged pulse over a train as well as its
    moving average.
    """
    def __init__(self, *, parent=None):
        super().__init__(parent=parent, show_indicator=True)

        self.setLabel('left', "ADU")
        self.setLabel('bottom', "Pixel")
        self.addLegend(offset=(-40, 20))

        self._mean = self.plotCurve(name="mean", pen=FColor.mkPen("p"))
        self._mean_ma = self.plotCurve(
            name="mean (moving average)", pen=FColor.mkPen("g"))

    def updateF(self, data):
        """Override."""
        self._mean.setData(data['spectrum_mean'])
        self._mean_ma.setData(data['spectrum_ma_mean'])


class GotthardPulsePlot(PlotWidgetF):
    """GotthardPulsePlot class.

    Visualize signals of a single pulse as well as its moving average.
    """
    def __init__(self, *, parent=None):
        super().__init__(parent=parent, show_indicator=True)

        self.setLabel('left', "ADU")
        self.setLabel('bottom', "Pixel")
        self.addLegend(offset=(-40, 20))

        self._idx = 0

        self._poi = None
        self._poi_ma = None
        self._initPlots()

    def _initPlots(self):
        self._poi = self.plotCurve(
            name=f"Pulse {self._idx}", pen=FColor.mkPen("p"))
        self._poi_ma = self.plotCurve(
            name=f"Pulse {self._idx} (moving average)", pen=FColor.mkPen("g"))

    def updateF(self, data):
        """Override."""
        idx = data['poi_index']
        if idx != self._idx:
            self._idx = idx
            # I don't known an easy way to change the legend
            self.clear()
            self._initPlots()

        self._poi.setData(data['spectrum'][idx])
        self._poi_ma.setData(data['spectrum_ma'][idx])


class GotthardImageView(ImageViewF):
    """GotthardImageView class.

    Visualize the heatmap of pulse-resolved Gotthard data in a train.
    """
    def __init__(self, *, parent=None):
        super().__init__(has_roi=True, roi_size=(100, 10), parent=parent)

        self.setAspectLocked(False)

        self.setTitle('ADU heatmap')
        self.setLabel('left', "Pulse index (sliced)")
        self.setLabel('bottom', "Pixel")

    def updateF(self, data):
        """Override."""
        self.setImage(data['spectrum'])


class GotthardHist(HistMixin, PlotWidgetF):
    """GotthardHist class

    Visualize the ADU histogram in a train.
    """
    def __init__(self, *, parent=None):
        super().__init__(parent=parent)

        self._plot = self.plotBar()

        self._title_template = Template(
            f"mean: $mean, median: $median, std: $std")
        self.updateTitle()
        self.setLabel('left', 'Occurence')
        self.setLabel('bottom', 'ADU')

    def updateF(self, data):
        """Override."""
        hist, bin_centers, mean, median, std = data['hist']
        if bin_centers is None:
            self.reset()
        else:
            self._plot.setData(bin_centers, hist)
            self.updateTitle(mean, median, std)


@create_special(GotthardCtrlWidget, GotthardProcessor, QThreadKbClient)
class GotthardWindow(_SpecialAnalysisBase):
    """Main GUI for Gotthard analysis."""

    icon = "Gotthard.png"
    _title = "Gotthard"
    _long_title = "Gotthard analysis"

    def __init__(self, topic):
        super().__init__(topic)

        self._poi_plots = GotthardPulsePlot(parent=self)
        self._mean_plots = GotthardAvgPlot(parent=self)
        self._heatmap = GotthardImageView(parent=self)
        self._hist = GotthardHist(parent=self)

        self.initUI()
        self.initConnections()
        self.startWorker()

    def initUI(self):
        """Override."""
        middle_panel = QSplitter(Qt.Vertical)
        middle_panel.addWidget(self._poi_plots)
        middle_panel.addWidget(self._mean_plots)

        right_panel = QSplitter(Qt.Vertical)
        right_panel.addWidget(self._hist)
        right_panel.addWidget(self._heatmap)
        right_panel.setSizes([self._TOTAL_H / 2, self._TOTAL_H / 2])

        cw = self.centralWidget()
        cw.addWidget(middle_panel)
        cw.addWidget(right_panel)
        cw.setSizes([self._TOTAL_W / 3, self._TOTAL_W / 3, self._TOTAL_W / 3])

        self.resize(self._TOTAL_W, self._TOTAL_H)

    def initConnections(self):
        """Override."""
        self._ctrl_widget_st.output_ch_le.value_changed_sgn.connect(
            self._worker_st.onOutputChannelChanged)
        self._ctrl_widget_st.output_ch_le.returnPressed.emit()

        self._ctrl_widget_st.poi_index_le.value_changed_sgn.connect(
            lambda x: self._worker_st.onPoiIndexChanged(int(x)))
        self._ctrl_widget_st.poi_index_le.returnPressed.emit()

        self._ctrl_widget_st.pulse_slicer_le.value_changed_sgn.connect(
            self._worker_st.onPulseSlicerChanged)
        self._ctrl_widget_st.pulse_slicer_le.returnPressed.emit()

        self._ctrl_widget_st.ma_window_le.value_changed_sgn.connect(
            self._worker_st.onMaWindowChanged)
        self._ctrl_widget_st.ma_window_le.returnPressed.emit()

        self._ctrl_widget_st.bin_range_le.value_changed_sgn.connect(
            self._worker_st.onBinRangeChanged)
        self._ctrl_widget_st.bin_range_le.returnPressed.emit()

        self._ctrl_widget_st.n_bins_le.value_changed_sgn.connect(
            self._worker_st.onNoBinsChanged)
        self._ctrl_widget_st.n_bins_le.returnPressed.emit()

        self._ctrl_widget_st.hist_over_ma_cb.toggled.connect(
            self._worker_st.onHistOverMaChanged)
        self._ctrl_widget_st.hist_over_ma_cb.toggled.emit(
            self._ctrl_widget_st.hist_over_ma_cb.isChecked())
