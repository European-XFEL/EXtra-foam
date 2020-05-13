"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIntValidator
from PyQt5.QtWidgets import QSplitter

from extra_foam.gui.plot_widgets import ImageViewF, PlotWidgetF
from extra_foam.gui.misc_widgets import FColor
from extra_foam.gui.ctrl_widgets import (
    SmartLineEdit, SmartSliceLineEdit, SmartStringLineEdit
)

from .gotthard_pump_probe_proc import GotthardPpProcessor
from .special_analysis_base import (
    create_special, QThreadKbClient, _BaseAnalysisCtrlWidgetS,
    _SpecialAnalysisBase
)
from .config import _MAX_N_GOTTHARD_PULSES, GOTTHARD_DEVICE


class GotthardPpCtrlWidget(_BaseAnalysisCtrlWidgetS):
    """Gotthard pump-probe analysis control widget."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.output_ch_le = SmartStringLineEdit(
            GOTTHARD_DEVICE.get(self.topic, "Gotthard:output"))

        self.ma_window_le = SmartLineEdit("1")
        validator = QIntValidator()
        validator.setBottom(1)
        self.ma_window_le.setValidator(validator)

        self.on_slicer_le = SmartSliceLineEdit("0:50:2")
        self.off_slicer_le = SmartSliceLineEdit("1:50:2")
        # Actual, POI index should be within on-pulse indices
        self.poi_index_le = SmartLineEdit("0")
        self.poi_index_le.setValidator(
            QIntValidator(0, _MAX_N_GOTTHARD_PULSES - 1))

        self.dark_slicer_le = SmartSliceLineEdit("100:120")
        self.dark_poi_index_le = SmartLineEdit("0")
        self.dark_poi_index_le.setValidator(
            QIntValidator(0, _MAX_N_GOTTHARD_PULSES - 1))

        self._non_reconfigurable_widgets = [
            self.output_ch_le
        ]

        self.initUI()
        self.initConnections()

    def initUI(self):
        """Override."""
        layout = self.layout()

        layout.addRow("Output channel: ", self.output_ch_le)
        layout.addRow("M.A. window: ", self.ma_window_le)
        layout.addRow("On-pulse slicer: ", self.on_slicer_le)
        layout.addRow("Off-pulse slicer: ", self.off_slicer_le)
        layout.addRow("Pump-probe P.O.I.: ", self.poi_index_le)
        layout.addRow("Dark-pulse slicer: ", self.dark_slicer_le)
        layout.addRow("Dark P.O.I.: ", self.dark_poi_index_le)

    def initConnections(self):
        """Override."""
        pass


class GotthardPpFomMeanPlot(PlotWidgetF):
    """GotthardPpFomMeanPlot class.

    Visualize averaged VFOM over a train as well as its moving average.
    """
    def __init__(self, *, parent=None):
        super().__init__(parent=parent, show_indicator=True)

        self.setTitle("Averaged FOM over train")
        self.setLabel('left', "ADU")
        self.setLabel('bottom', "Pixel")
        self.addLegend(offset=(10, 5))

        self._mean = self.plotCurve(name="Current", pen=FColor.mkPen("p"))
        self._mean_ma = self.plotCurve(name="Moving average",
                                       pen=FColor.mkPen("g"))

    def updateF(self, data):
        """Override."""
        self._mean.setData(data['vfom_mean'])
        self._mean_ma.setData(data['vfom_ma_mean'])


class GotthardPpFomPulsePlot(PlotWidgetF):
    """GotthardPpFomPulsePlot class.

    Visualize VFOM of a single pump-probe pulse as well as its moving average.
    """
    def __init__(self, *, parent=None):
        """Initialization."""
        super().__init__(parent=parent, show_indicator=True)

        self._idx = 0

        self._updateTitle()
        self.setLabel('left', "ADU")
        self.setLabel('bottom', "Pixel")
        self.addLegend(offset=(10, 5))

        self._poi = self.plotCurve(name="Current", pen=FColor.mkPen("p"))
        self._poi_ma = self.plotCurve(name=f"Moving average",
                                      pen=FColor.mkPen("g"))

    def _updateTitle(self):
        self.setTitle(f"Pulse of interest (FOM): {self._idx}")

    def updateF(self, data):
        """Override."""
        idx = data['poi_index']
        if idx != self._idx:
            self._idx = idx
            self._updateTitle()

        self._poi.setData(data['vfom'][idx])
        self._poi_ma.setData(data['vfom_ma'][idx])


class GotthardPpRawPulsePlot(PlotWidgetF):
    """GotthardPpRawPulsePlot class.

    Visualize raw data a pair of on/off pulses.
    """
    def __init__(self, *, parent=None):
        """Initialization."""
        super().__init__(parent=parent, show_indicator=True)

        self._idx = 0

        self._updateTitle()
        self.setLabel('left', "ADU")
        self.setLabel('bottom', "Pixel")
        self.addLegend(offset=(10, 5))

        self._on = self.plotCurve(name="On", pen=FColor.mkPen("r"))
        self._off = self.plotCurve(name="Off", pen=FColor.mkPen("b"))

    def _updateTitle(self):
        self.setTitle(f"Pulse of interest (raw): {self._idx}")

    def updateF(self, data):
        """Override."""
        idx = data['poi_index']
        if idx != self._idx:
            self._idx = idx
            self._updateTitle()

        self._on.setData(data['raw'][data['on_slicer']][idx])
        self._off.setData(data['raw'][data['off_slicer']][idx])


class GotthardPpDarkPulsePlot(PlotWidgetF):
    """GotthardPpDarkPulsePlot class.

    Visualize raw data of a dark pulse.
    """
    def __init__(self, *, parent=None):
        """Initialization."""
        super().__init__(parent=parent, show_indicator=True)

        self._idx = 0

        self._updateTitle()
        self.setLabel('left', "ADU")
        self.setLabel('bottom', "Pixel")

        self._plot = self.plotCurve(pen=FColor.mkPen("k"))

    def _updateTitle(self):
        self.setTitle(f"Pulse of interest (dark): {self._idx}")

    def updateF(self, data):
        """Override."""
        idx = data['dark_poi_index']
        if idx != self._idx:
            self._idx = idx
            self._updateTitle()

        self._plot.setData(data['raw'][data['dark_slicer']][idx])


class GotthardPpImageView(ImageViewF):
    """GotthardPpImageView class.

    Visualize the heatmap of pulse-resolved Gotthard data in a train
    after dark subtraction.
    """
    def __init__(self, *, parent=None):
        super().__init__(parent=parent)

        self.setAspectLocked(False)

        self.setTitle('ADU heatmap (corrected)')
        self.setLabel('left', "Pulse index")
        self.setLabel('bottom', "Pixel")

    def updateF(self, data):
        """Override."""
        self.setImage(data['corrected'])


@create_special(GotthardPpCtrlWidget, GotthardPpProcessor, QThreadKbClient)
class GotthardPumpProbeWindow(_SpecialAnalysisBase):
    """Main GUI for Gotthard pump-probe analysis."""

    icon = "Gotthard_pump_probe.png"
    _title = "Gotthard (pump-probe)"
    _long_title = "Gotthard pump-probe analysis"

    def __init__(self, topic):
        super().__init__(topic, with_dark=False)

        self._fom_poi_plot = GotthardPpFomPulsePlot(parent=self)
        self._fom_mean_plot = GotthardPpFomMeanPlot(parent=self)
        self._heatmap = GotthardPpImageView(parent=self)
        self._raw_poi_plot = GotthardPpRawPulsePlot(parent=self)
        self._dark_poi_plot = GotthardPpDarkPulsePlot(parent=self)

        self.initUI()
        self.initConnections()
        self.startWorker()

    def initUI(self):
        """Override."""
        right_panel = QSplitter(Qt.Vertical)
        right_up_panel = QSplitter()
        right_up_panel.addWidget(self._fom_poi_plot)
        right_up_panel.addWidget(self._fom_mean_plot)

        right_down_panel = QSplitter(Qt.Horizontal)
        sub_right_down_panel = QSplitter(Qt.Vertical)
        sub_right_down_panel.addWidget(self._raw_poi_plot)
        sub_right_down_panel.addWidget(self._dark_poi_plot)
        right_down_panel.addWidget(self._heatmap)
        right_down_panel.addWidget(sub_right_down_panel)
        right_down_panel.setSizes([self._TOTAL_W / 3, self._TOTAL_W / 3])

        right_panel.addWidget(right_up_panel)
        right_panel.addWidget(right_down_panel)
        right_panel.setSizes([self._TOTAL_H / 2, self._TOTAL_H / 2])

        cw = self.centralWidget()
        cw.addWidget(right_panel)
        cw.setSizes([self._TOTAL_W / 3, 2 * self._TOTAL_W / 3])

        self.resize(self._TOTAL_W, self._TOTAL_H)

    def initConnections(self):
        """Override."""
        self._ctrl_widget_st.output_ch_le.value_changed_sgn.connect(
            self._worker_st.onOutputChannelChanged)
        self._ctrl_widget_st.output_ch_le.returnPressed.emit()

        self._ctrl_widget_st.on_slicer_le.value_changed_sgn.connect(
            self._worker_st.onOnSlicerChanged)
        self._ctrl_widget_st.on_slicer_le.returnPressed.emit()

        self._ctrl_widget_st.off_slicer_le.value_changed_sgn.connect(
            self._worker_st.onOffSlicerChanged)
        self._ctrl_widget_st.off_slicer_le.returnPressed.emit()

        self._ctrl_widget_st.poi_index_le.value_changed_sgn.connect(
            lambda x: self._worker_st.onPoiIndexChanged(int(x)))
        self._ctrl_widget_st.poi_index_le.returnPressed.emit()

        self._ctrl_widget_st.dark_slicer_le.value_changed_sgn.connect(
            self._worker_st.onDarkSlicerChanged)
        self._ctrl_widget_st.dark_slicer_le.returnPressed.emit()

        self._ctrl_widget_st.dark_poi_index_le.value_changed_sgn.connect(
            lambda x: self._worker_st.onDarkPoiIndexChanged(int(x)))
        self._ctrl_widget_st.dark_poi_index_le.returnPressed.emit()

        self._ctrl_widget_st.ma_window_le.value_changed_sgn.connect(
            self._worker_st.onMaWindowChanged)
        self._ctrl_widget_st.ma_window_le.returnPressed.emit()
