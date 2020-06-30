"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from functools import partial

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QDoubleValidator
from PyQt5.QtWidgets import QSplitter, QTabWidget

from extra_foam.gui.ctrl_widgets import SmartLineEdit, SmartStringLineEdit
from extra_foam.gui.plot_widgets import TimedPlotWidgetF
from extra_foam.gui.misc_widgets import FColor

from .special_analysis_base import (
    create_special, QThreadKbClient, _SpecialAnalysisBase
)
from .xas_tim_xmcd_proc import XasTimXmcdProcessor, _DEFAULT_CURRENT_THRESHOLD
from .xas_tim_w import (
    XasTimCtrlWidget, XasTimCorrelationPlot, XasTimXgmPulsePlot,
    XasTimDigitizerPulsePlot, XasTimXgmSpectrumPlot
)

_DIGITIZER_CHANNEL_COLORS = ['r', 'b', 'o', 'k']


class XasTimXmcdCtrlWidget(XasTimCtrlWidget):
    """XAS-TIM-XMCD analysis control widget.

    XMCD stands for X-ray magnetic circular dichroism.
    """

    # True if spectrum from one MCP channel can be visualized at a time.
    _MCP_EXCLUSIVE = True

    def __init__(self, *args, **kwargs):

        self.magnet_device_le = SmartStringLineEdit(
            "SCS_CDIFFT_MAG/ASENS/CURRENT")

        self.current_threshold_le = SmartLineEdit(
            str(_DEFAULT_CURRENT_THRESHOLD))
        validator = QDoubleValidator()
        validator.setBottom(_DEFAULT_CURRENT_THRESHOLD)
        self.current_threshold_le.setValidator(validator)

        super().__init__(*args, **kwargs)

        self._non_reconfigurable_widgets.append(self.magnet_device_le)

    def initUI(self):
        """Override."""
        self.spectra_displayed.setExclusive(True)
        self.spectra_displayed.button(0).setChecked(True)
        super().initUI()

        layout = self.layout()
        layout.insertRow(4, "Magnet device ID: ", self.magnet_device_le)
        layout.insertRow(
            8, "Magnet current threshold: ", self.current_threshold_le)

    def initConnections(self):
        """Override."""
        pass


class XasTimXmcdSlowScanPlot(TimedPlotWidgetF):
    """XasTimXmcdSlowScanPlot class.

    Visualize path of soft mono energy scan and magnet current scan.
    """
    def __init__(self, *, parent=None):
        """Initialization."""
        super().__init__(parent=parent)

        self.setLabel('left', "Energy (eV)")
        self.setLabel('right', "Current (A)")
        self.setLabel('bottom', "Train ID")
        self.addLegend(offset=(-40, 20))

        self._energy = self.plotCurve(name="Energy", pen=FColor.mkPen('r'))
        self._current = self.plotCurve(
            name="Current", pen=FColor.mkPen('b'), y2=True)

    def refresh(self):
        """Override."""
        self._energy.setData(*self._data['energy_scan'])
        self._current.setData(*self._data['current_scan'])


class XasTimXmcdAbsorpPnSpectraPlot(TimedPlotWidgetF):
    """XasTimXmcdAbsorpPnSpectraPlot class.

    Visualize positive and negative spectra for a single MCP.
    """
    def __init__(self, *, parent=None):
        """Initialization."""
        super().__init__(parent=parent)

        self.setTitle("XAS-p & XAS-n")
        self.setLabel('left', "Absorption (arb.)")
        self.setLabel('right', "Count")
        self.setLabel('bottom', "Energy (eV)")
        self.addLegend(offset=(-40, 20))

        self._displayed = [False] * 4

        self._xas_p = self.plotCurve(name="XAS-p", pen=FColor.mkPen('r'))
        self._xas_n = self.plotCurve(name="XAS-n", pen=FColor.mkPen('b'))
        self._count = self.plotBar(
            y2=True, brush=FColor.mkBrush('i', alpha=70))

    def refresh(self):
        """Override."""
        stats, centers, counts = self._data["spectra"]

        xas_p, xas_n = stats[self._displayed]
        if xas_p is None or xas_n is None:
            self._xas_p.setData([], [])
            self._xas_n.setData([], [])
            self._count.setData([], [])
        else:
            self._xas_p.setData(centers, xas_p)
            self._xas_n.setData(centers, xas_n)
            self._count.setData(centers, counts)

    def onSpectraDisplayedChanged(self, index: int, value: bool):
        if value:
            self._displayed = index


class XasTimXmcdSpectraPlot(TimedPlotWidgetF):
    """XasTimXmcdSpectraPlot class.

    XAS = sigma_+ + sigma_- + sigma_z,

    with sigma+, sigma- and sigma_z spectra recorded with different X-ray
    polarization, and sigma_z ~ 1/2(sigma+ + sigma-). It is why we visualize
    1.5 * (XAS-p + XAS-n).

    Visualize XMCD and sum of XAS for a single MCP.
    """
    def __init__(self, *, parent=None):
        """Initialization."""
        super().__init__(parent=parent)

        self.setTitle("XAS & XMCD")
        self.setLabel('left', "Absorption (arb.)")
        self.setLabel('right', "XMCD (arb.)")
        self.setLabel('bottom', "Energy (eV)")
        self.addLegend(offset=(-40, 20))

        self._displayed = 0

        self._xas = self.plotCurve(name="XAS", pen=FColor.mkPen('p'))
        self._xmcd = self.plotCurve(pen=FColor.mkPen('g'), y2=True)

    def refresh(self):
        """Override."""
        stats, centers, counts = self._data["spectra"]
        xas_p, xas_n = stats[self._displayed]
        if xas_p is None or xas_n is None:
            self._xas.setData([], [])
            self._xmcd.setData([], [])
        else:
            self._xas.setData(centers, 1.5 * (xas_n + xas_p))
            self._xmcd.setData(centers, xas_n - xas_p)

    def onSpectraDisplayedChanged(self, index: int, value: bool):
        if value:
            self._displayed = index


@create_special(XasTimXmcdCtrlWidget, XasTimXmcdProcessor, QThreadKbClient)
class XasTimXmcdWindow(_SpecialAnalysisBase):
    """Main GUI for XAS-TIM-XMCD analysis."""

    icon = "xas_tim_xmcd.png"
    _title = "XAS-TIM-XMCD"
    _long_title = "X-ray Absorption Spectroscopy with transmission " \
                  "intensity monitor for X-ray Magnetic Circular Dichroism"

    def __init__(self, topic):
        super().__init__(topic, with_dark=False, with_levels=False)

        self._xgm = XasTimXgmPulsePlot(parent=self)
        self._digitizer = XasTimDigitizerPulsePlot(parent=self)
        self._scan = XasTimXmcdSlowScanPlot(parent=self)

        self._correlations = [XasTimCorrelationPlot(i, parent=self)
                              for i in range(4)]
        self._pn_spectra = XasTimXmcdAbsorpPnSpectraPlot(parent=self)
        self._xas_xmcd_spectra = XasTimXmcdSpectraPlot(parent=self)
        self._i0_spectrum = XasTimXgmSpectrumPlot(parent=self)

        self.initUI()
        self.initConnections()
        self.startWorker()

    def initUI(self):
        """Override."""
        right_panel = QTabWidget()

        right_panel1 = QSplitter(Qt.Vertical)
        right_panel1.addWidget(self._xgm)
        right_panel1.addWidget(self._digitizer)
        right_panel1.addWidget(self._scan)

        right_panel2 = QSplitter(Qt.Horizontal)
        correlation_panel = QSplitter(Qt.Vertical)
        for w in self._correlations:
            correlation_panel.addWidget(w)
        spectra_panel = QSplitter(Qt.Vertical)
        spectra_panel.addWidget(self._pn_spectra)
        spectra_panel.addWidget(self._xas_xmcd_spectra)
        spectra_panel.addWidget(self._i0_spectrum)
        right_panel2.addWidget(correlation_panel)
        right_panel2.addWidget(spectra_panel)
        right_panel2.setSizes([100, 200])

        right_panel.addTab(right_panel1, "Raw data")
        right_panel.addTab(right_panel2, "Correlation and spectra")
        right_panel.setTabPosition(QTabWidget.TabPosition.South)

        cw = self.centralWidget()
        cw.addWidget(right_panel)
        cw.setSizes([self._TOTAL_W / 4, 3 * self._TOTAL_W / 4])

        self.resize(self._TOTAL_W, self._TOTAL_H)

    def initConnections(self):
        """Override."""
        self._ctrl_widget_st.xgm_output_ch_le.value_changed_sgn.connect(
            self._worker_st.onXgmOutputChannelChanged)
        self._ctrl_widget_st.xgm_output_ch_le.returnPressed.emit()

        self._ctrl_widget_st.digitizer_output_ch_le.value_changed_sgn.connect(
            self._worker_st.onDigitizerOutputChannelChanged)
        self._ctrl_widget_st.digitizer_output_ch_le.returnPressed.emit()

        for i, cb in enumerate(self._ctrl_widget_st.digitizer_channels.buttons()):
            cb.toggled.connect(
                partial(self._worker_st.onDigitizerChannelsChanged, i))
            cb.toggled.emit(cb.isChecked())

        self._ctrl_widget_st.mono_device_le.value_changed_sgn.connect(
            self._worker_st.onMonoDeviceChanged)
        self._ctrl_widget_st.mono_device_le.returnPressed.emit()

        self._ctrl_widget_st.n_pulses_per_train_le.value_changed_sgn.connect(
            self._worker_st.onNPulsesPerTrainChanged)
        self._ctrl_widget_st.n_pulses_per_train_le.returnPressed.emit()

        self._ctrl_widget_st.apd_stride_le.value_changed_sgn.connect(
            self._worker_st.onApdStrideChanged)
        self._ctrl_widget_st.apd_stride_le.returnPressed.emit()

        self._ctrl_widget_st.i0_threshold_le.value_changed_sgn.connect(
            self._worker_st.onI0ThresholdChanged)
        self._ctrl_widget_st.i0_threshold_le.returnPressed.emit()

        self._ctrl_widget_st.pulse_window_le.value_changed_sgn.connect(
            self._worker_st.onPulseWindowChanged)
        self._ctrl_widget_st.pulse_window_le.returnPressed.emit()

        self._ctrl_widget_st.correlation_window_le.value_changed_sgn.connect(
            self._worker_st.onCorrelationWindowChanged)
        self._ctrl_widget_st.correlation_window_le.returnPressed.emit()

        self._ctrl_widget_st.n_bins_le.value_changed_sgn.connect(
            self._worker_st.onNoBinsChanged)
        self._ctrl_widget_st.n_bins_le.returnPressed.emit()

        self._ctrl_widget_st.magnet_device_le.value_changed_sgn.connect(
            self._worker_st.onMagnetDeviceChanged)
        self._ctrl_widget_st.magnet_device_le.returnPressed.emit()

        self._ctrl_widget_st.current_threshold_le.value_changed_sgn.connect(
            self._worker_st.onMagnetThresholdChanged)
        self._ctrl_widget_st.current_threshold_le.returnPressed.emit()

        for i, cb in enumerate(self._ctrl_widget_st.spectra_displayed.buttons()):
            cb.toggled.connect(
                partial(self._pn_spectra.onSpectraDisplayedChanged, i))
            cb.toggled.connect(
                partial(self._xas_xmcd_spectra.onSpectraDisplayedChanged, i))
            cb.toggled.emit(cb.isChecked())
