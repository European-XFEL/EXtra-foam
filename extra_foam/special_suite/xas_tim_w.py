"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from functools import partial

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QDoubleValidator, QIntValidator
from PyQt5.QtWidgets import (
    QButtonGroup, QCheckBox, QHBoxLayout, QSplitter, QTabWidget
)

from extra_foam.gui.ctrl_widgets import SmartLineEdit, SmartStringLineEdit
from extra_foam.gui.misc_widgets import FColor
from extra_foam.gui.plot_widgets import PlotWidgetF, TimedPlotWidgetF

from .config import config
from .xas_tim_proc import (
    XasTimProcessor, _DEFAULT_N_PULSES_PER_TRAIN, _DEFAULT_I0_THRESHOLD,
    _MAX_WINDOW, _MAX_CORRELATION_WINDOW, _DIGITIZER_CHANNEL_NAMES,
    _DEFAULT_N_BINS, _MAX_N_BINS,
)
from .special_analysis_base import (
    create_special, QThreadKbClient, _BaseAnalysisCtrlWidgetS,
    _SpecialAnalysisBase
)

_DIGITIZER_CHANNEL_COLORS = ['r', 'b', 'o', 'k']


class XasTimCtrlWidget(_BaseAnalysisCtrlWidgetS):
    """XAS-TIM analysis control widget.

    XAS-TIM stands for X-ray Absorption Spectroscopy with transmission
    intensity monitor.
    """

    # True if spectrum from one MCP channel can be visualized at a time.
    _MCP_EXCLUSIVE = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.xgm_output_ch_le = SmartStringLineEdit(
            "SCS_BLU_XGM/XGM/DOOCS:output")

        self.digitizer_output_ch_le = SmartStringLineEdit(
            "SCS_UTC1_ADQ/ADC/1:network")
        self.digitizer_channels = QButtonGroup()
        self.digitizer_channels.setExclusive(False)
        for i, ch in enumerate(_DIGITIZER_CHANNEL_NAMES, 1):
            cb = QCheckBox(ch, self)
            cb.setChecked(True)
            self.digitizer_channels.addButton(cb, i-1)

        self.mono_device_le = SmartStringLineEdit(
            "SA3_XTD10_MONO/MDL/PHOTON_ENERGY")

        self.n_pulses_per_train_le = SmartLineEdit(
            str(_DEFAULT_N_PULSES_PER_TRAIN))
        self.n_pulses_per_train_le.setValidator(
            QIntValidator(1, config["MAX_N_PULSES_PER_TRAIN"]))

        self.apd_stride_le = SmartLineEdit("1")

        self.spectra_displayed = QButtonGroup()
        self.spectra_displayed.setExclusive(self._MCP_EXCLUSIVE)
        for i, _ in enumerate(_DIGITIZER_CHANNEL_NAMES, 1):
            cb = QCheckBox(f"MCP{i}", self)
            cb.setChecked(True)
            self.spectra_displayed.addButton(cb, i-1)

        self.i0_threshold_le = SmartLineEdit(str(_DEFAULT_I0_THRESHOLD))
        self.i0_threshold_le.setValidator(QDoubleValidator())

        self.pulse_window_le = SmartLineEdit(str(_MAX_WINDOW))
        self.pulse_window_le.setValidator(QIntValidator(1, _MAX_WINDOW))

        self.correlation_window_le = SmartLineEdit(
            str(_MAX_CORRELATION_WINDOW))
        self.correlation_window_le.setValidator(
            QIntValidator(1, _MAX_CORRELATION_WINDOW))

        self.n_bins_le = SmartLineEdit(str(_DEFAULT_N_BINS))
        self.n_bins_le.setValidator(QIntValidator(1, _MAX_N_BINS))

        self._non_reconfigurable_widgets = [
            self.xgm_output_ch_le,
            self.digitizer_output_ch_le,
            *self.digitizer_channels.buttons(),
            self.mono_device_le,
        ]

        self.initUI()
        self.initConnections()

    def initUI(self):
        """Override."""
        layout = self.layout()

        digitizer_channels_layout = QHBoxLayout()
        for cb in self.digitizer_channels.buttons():
            digitizer_channels_layout.addWidget(cb)

        spectra_displayed_layout = QHBoxLayout()
        for cb in self.spectra_displayed.buttons():
            spectra_displayed_layout.addWidget(cb)

        layout.addRow("XGM output channel: ", self.xgm_output_ch_le)
        layout.addRow("Digitizer output channel: ", self.digitizer_output_ch_le)
        layout.addRow("Digitizer channels: ", digitizer_channels_layout)
        layout.addRow("Mono device ID: ", self.mono_device_le)
        layout.addRow("# of pulses/train: ", self.n_pulses_per_train_le)
        layout.addRow("APD stride: ", self.apd_stride_le)
        layout.addRow('XGM intensity threshold: ', self.i0_threshold_le)
        layout.addRow('Pulse window: ', self.pulse_window_le)
        layout.addRow('Correlation window: ', self.correlation_window_le)
        layout.addRow('# of energy bins: ', self.n_bins_le)
        layout.addRow("Show spectra: ", spectra_displayed_layout)

    def initConnections(self):
        """Override."""
        pass


class XasTimXgmPulsePlot(PlotWidgetF):
    """XasTimXgmPulsePlot class.

    Visualize XGM intensity in the current train.
    """
    def __init__(self, *, parent=None):
        """Initialization."""
        super().__init__(parent=parent, show_indicator=True)

        self.setTitle("Pulse intensities (SA3)")
        self.setLabel('left', "Intensity (arb.)")
        self.setLabel('bottom', "Pulse index")

        self._plot = self.plotCurve(pen=FColor.mkPen("g"))

    def updateF(self, data):
        """Override."""
        self._plot.setData(data['xgm_intensity'])


class XasTimDigitizerPulsePlot(PlotWidgetF):
    """XasTimDigitizerPulsePlot class.

    Visualize pulse integral of each channel of the digitizer
    in the current train.
    """
    def __init__(self, *, parent=None):
        """Initialization."""
        super().__init__(parent=parent, show_indicator=True)

        self.setTitle("Digitizer pulse integrals")
        self.setLabel('left', "Pulse integral (arb.)")
        self.setLabel('bottom', "Pulse index")
        self.addLegend(offset=(-40, 60))

        self._plots = []
        for ch, c in zip(_DIGITIZER_CHANNEL_NAMES, _DIGITIZER_CHANNEL_COLORS):
            self._plots.append(self.plotCurve(
                name=f"Digitizer channel {ch}", pen=FColor.mkPen(c)))

    def updateF(self, data):
        """Override."""
        for p, apd in zip(self._plots, data['digitizer_apds']):
            p.setData(apd)


class XasTimMonoScanPlot(TimedPlotWidgetF):
    """XasTimMonoScanPlot class.

    Visualize path of soft mono energy scan.
    """
    def __init__(self, *, parent=None):
        """Initialization."""
        super().__init__(parent=parent, show_indicator=True)

        self.setTitle("Softmono energy scan")
        self.setLabel('left', "Energy (eV)")
        self.setLabel('bottom', "Train ID")

        self._plot = self.plotCurve(pen=FColor.mkPen('b'))

    def refresh(self):
        """Override."""
        self._plot.setData(*self._data['energy_scan'])


class XasTimCorrelationPlot(TimedPlotWidgetF):
    """XasTimCorrelationPlot class.

    Visualize correlation between I0 and I1 for single channel.
    """
    def __init__(self, idx, *, parent=None):
        """Initialization.

        :param int idx: channel index.
        """
        super().__init__(parent=parent, show_indicator=True)

        self.setLabel('left', "I1 (arb.)")
        self.setLabel('bottom', "I0 (micro J)")
        self.setTitle(f"MCP{idx+1} correlation")
        self._idx = idx

        self._plot = self.plotScatter(
            brush=FColor.mkBrush(_DIGITIZER_CHANNEL_COLORS[idx], alpha=150))

    def refresh(self):
        """Override."""
        data = self._data
        i1 = data['i1'][self._idx]
        if i1 is None:
            self._plot.setData([], [])
        else:
            s = data['correlation_length']
            self._plot.setData(data['i0'][-s:], i1[-s:])


class XasTimSpectraPlot(TimedPlotWidgetF):
    """XasTimSpectraPlot class.

    Visualize spectrum for all MCPs.
    """
    def __init__(self, *, parent=None):
        """Initialization."""
        super().__init__(parent=parent, show_indicator=True)

        self.setTitle("MCP spectra")
        self.setLabel('left', "Absorption (arb.)")
        self.setLabel('right', "Count")
        self.setLabel('bottom', "Energy (eV)")
        self.addLegend(offset=(-40, 20))

        self._displayed = [False] * 4

        self._plots = []
        for i, c in enumerate(_DIGITIZER_CHANNEL_COLORS):
            self._plots.append(
                self.plotCurve(name=f"MCP{i+1}", pen=FColor.mkPen(c)))
        self._count = self.plotBar(
            y2=True, brush=FColor.mkBrush('i', alpha=70))

    def refresh(self):
        """Override."""
        stats, centers, counts = self._data["spectra"]
        for i, p in enumerate(self._plots):
            v = stats[i]
            if v is not None and self._displayed[i]:
                p.setData(centers, v)
            else:
                p.setData([], [])
        self._count.setData(centers, counts)

    def onSpectraDisplayedChanged(self, index: int, value: bool):
        self._displayed[index] = value


class XasTimXgmSpectrumPlot(TimedPlotWidgetF):
    """XasTimXgmSpectrumPlot class.

    Visualize spectrum of I0 (XGM).
    """
    def __init__(self, *, parent=None):
        """Initialization."""
        super().__init__(parent=parent, show_indicator=True)

        self.setTitle("XGM spectrum")
        self.setLabel('left', "I0 (arb.)")
        self.setLabel('right', "Count")
        self.setLabel('bottom', "Energy (eV)")

        self._plot = self.plotScatter(brush=FColor.mkBrush("w"))
        self._count = self.plotBar(
            y2=True, brush=FColor.mkBrush('i', alpha=70))

    def refresh(self):
        """Override."""
        stats, centers, counts = self._data["spectra"]
        self._plot.setData(centers, stats[4])
        self._count.setData(centers, counts)


@create_special(XasTimCtrlWidget, XasTimProcessor, QThreadKbClient)
class XasTimWindow(_SpecialAnalysisBase):
    """Main GUI for XAS-TIM analysis."""

    icon = "xas_tim.png"
    _title = "XAS-TIM"
    _long_title = "X-ray Absorption Spectroscopy with transmission " \
                  "intensity monitor"

    def __init__(self, topic):
        super().__init__(topic, with_dark=False, with_levels=False)

        self._xgm = XasTimXgmPulsePlot(parent=self)
        self._digitizer = XasTimDigitizerPulsePlot(parent=self)
        self._mono = XasTimMonoScanPlot(parent=self)

        self._correlations = [XasTimCorrelationPlot(i, parent=self)
                              for i in range(4)]
        self._spectra = XasTimSpectraPlot(parent=self)
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
        right_panel1.addWidget(self._mono)

        right_panel2 = QSplitter(Qt.Horizontal)
        correlation_panel = QSplitter(Qt.Vertical)
        for w in self._correlations:
            correlation_panel.addWidget(w)
        spectra_panel = QSplitter(Qt.Vertical)
        spectra_panel.addWidget(self._spectra)
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

        for i, cb in enumerate(self._ctrl_widget_st.spectra_displayed.buttons()):
            cb.toggled.connect(
                partial(self._spectra.onSpectraDisplayedChanged, i))
            cb.toggled.emit(cb.isChecked())
