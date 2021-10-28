"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QSplitter

from extra_foam.gui.ctrl_widgets.smart_widgets import (
    SmartStringLineEdit
)
from extra_foam.gui.misc_widgets import FColor
from extra_foam.gui.plot_widgets import (
    ImageViewF, TimedPlotWidgetF
)

from .xes_timing_proc import XesTimingProcessor
from .special_analysis_base import (
    create_special, ClientType, _BaseAnalysisCtrlWidgetS,
    _SpecialAnalysisBase
)

_MAX_N_BINS = 999


class XesTimingCtrlWidget(_BaseAnalysisCtrlWidgetS):
    """XES timing control widget."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.output_ch_le = SmartStringLineEdit(
            "FXE_XAD_JF1M/DET/RECEIVER-2:daqOutput")
        self.delay_device_le = SmartStringLineEdit(
            "FXE_AUXT_LIC/DOOCS/PPODL")

        self.output_ch_le = SmartStringLineEdit(
            "camera1:output")
        self.delay_device_le = SmartStringLineEdit(
            "data_generator")

        self._non_reconfigurable_widgets = [
            self.output_ch_le,
            self.delay_device_le
        ]

        self.initUI()
        self.initConnections()

    def initUI(self):
        """Override."""
        layout = self.layout()

        layout.addRow("Output channel: ", self.output_ch_le)
        layout.addRow("Delay device: ", self.delay_device_le)

    def initConnections(self):
        """Override."""
        pass


class XesTimingView(ImageViewF):
    """XesTimingView class.

    Visualize the detector image.
    """
    def __init__(self, *, parent=None):
        """Initialization."""
        super().__init__(has_roi=True, parent=parent)

    def updateF(self, data):
        """Override."""
        self.setImage(data['displayed'])


class XesTimingDelayScan(TimedPlotWidgetF):
    """XesTimingDelayScan class.

    Visualize path of laser delay scan.
    """
    def __init__(self, *, parent=None):
        """Initialization."""
        super().__init__(parent=parent)

        self.setTitle("Laser delay scan")
        self.setLabel('left', "Position (arb.)")
        self.setLabel('bottom', "Train ID")

        self._plot = self.plotCurve(pen=FColor.mkPen('b'))

    def refresh(self):
        """Override."""
        self._plot.setData(*self._data['delay_scan'])


@create_special(XesTimingCtrlWidget, XesTimingProcessor)
class XesTimingWindow(_SpecialAnalysisBase):
    """Main GUI for XES timing."""

    icon = "xes_timing.png"
    _title = "XES timing"
    _long_title = "X-ray emission spectroscopy timing tool"
    _client_support = ClientType.KARABO_BRIDGE

    def __init__(self, topic):
        super().__init__(topic)

        self._view = XesTimingView(parent=self)
        self._delay_scan = XesTimingDelayScan(parent=self)

        self.initUI()
        self.initConnections()
        self.startWorker()

    def initUI(self):
        """Override."""
        right_panel = QSplitter(Qt.Vertical)
        right_panel.addWidget(self._view)
        right_panel.addWidget(self._delay_scan)
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

        self._ctrl_widget_st.delay_device_le.value_changed_sgn.connect(
            self._worker_st.onDelayDeviceChanged)
        self._ctrl_widget_st.delay_device_le.returnPressed.emit()
