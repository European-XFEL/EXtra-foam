"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QFrame, QVBoxLayout, QSplitter

from .base_window import _AbstractPlotWindow
from ..ctrl_widgets import PumpProbeCtrlWidget
from ..plot_widgets import PlotWidgetF, TimedPlotWidgetF
from ..misc_widgets import FColor
from ...config import config, AnalysisType, plot_labels


class PumpProbeVFomPlot(PlotWidgetF):
    """PumpProbeVFomPlot class.

    Widget for displaying the pump and probe signal or their difference.
    """

    def __init__(self, diff=False, *, parent=None):
        """Initialization.

        :param bool diff: True for displaying on-off while False for
            displaying on and off
        """
        super().__init__(parent=parent)

        self._analysis_type = AnalysisType.UNDEFINED
        x_label, y_label = plot_labels[self._analysis_type]
        self.setTitle('VFOM')
        self.setLabel('bottom', x_label)
        self.setLabel('left', y_label)
        self.addLegend(offset=(-40, 20))

        self._is_diff = diff
        if diff:
            self._on_off_pulse = self.plotCurve(name="On - Off", pen=FColor.mkPen("p"))
        else:
            self._on_pulse = self.plotCurve(name="On", pen=FColor.mkPen("r"))
            self._off_pulse = self.plotCurve(name="Off", pen=FColor.mkPen("b"))

    def updateF(self, data):
        """Override."""
        pp = data["processed"].pp
        x, y = pp.x, pp.y

        if self._analysis_type != pp.analysis_type:
            x_label, y_label = plot_labels[pp.analysis_type]
            self.setLabel('bottom', x_label)
            self.setLabel('left', y_label)
            self._analysis_type = pp.analysis_type
            self.reset()

        if y is None:
            return

        if self._is_diff:
            self._on_off_pulse.setData(x, y)
        else:
            y_on, y_off = pp.y_on, pp.y_off
            self._on_pulse.setData(x, y_on)
            self._off_pulse.setData(x, y_off)


class PumpProbeFomPlot(TimedPlotWidgetF):
    """PumpProbeFomPlot class.

    Widget for displaying the evolution of FOM in pump-probe analysis.
    """

    def __init__(self, *, parent=None):
        """Initialization."""
        super().__init__(parent=parent)

        self.setLabel('bottom', "Train ID")
        self.setLabel('left', "FOM (arb. u.)")
        self.setTitle('FOM correlation')

        self._plot = self.plotScatter()

    def refresh(self):
        """Override."""
        pp = self._data["processed"].corr.pp
        x, y = pp.x, pp.y
        self._plot.setData(x, y)


class PumpProbeWindow(_AbstractPlotWindow):
    """PumpProbeWindow class."""
    _title = "Pump-probe"

    _TOTAL_W, _TOTAL_H = config['GUI_PLOT_WINDOW_SIZE']

    def __init__(self, *args, **kwargs):
        """Initialization."""
        super().__init__(*args, **kwargs)

        self._ctrl_widget = self.createCtrlWidget(PumpProbeCtrlWidget)

        self._pp_fom = PumpProbeFomPlot(parent=self)

        self._pp_onoff = PumpProbeVFomPlot(parent=self)
        self._pp_diff = PumpProbeVFomPlot(diff=True, parent=self)

        self.initUI()
        self.initConnections()
        self.loadMetaData()
        self.updateMetaData()

        self.resize(self._TOTAL_W, self._TOTAL_H)
        self.setMinimumSize(int(0.6 * self._TOTAL_W), int(0.6 * self._TOTAL_H))

        self.update()

    def initUI(self):
        """Override."""
        plots = QSplitter()
        right_panel = QSplitter(Qt.Vertical)
        right_panel.addWidget(self._pp_onoff)
        right_panel.addWidget(self._pp_diff)
        plots.addWidget(self._pp_fom)
        plots.addWidget(right_panel)

        self._cw = QFrame()
        layout = QVBoxLayout()
        layout.addWidget(plots)
        layout.addWidget(self._ctrl_widget)
        self._ctrl_widget.setFixedHeight(
            self._ctrl_widget.minimumSizeHint().height())
        self._cw.setLayout(layout)
        self.setCentralWidget(self._cw)

    def initConnections(self):
        """Override."""
        pass

    def closeEvent(self, QCloseEvent):
        self._ctrl_widget.resetAnalysisType()
        super().closeEvent(QCloseEvent)
