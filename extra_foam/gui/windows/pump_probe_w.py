"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import numpy as np

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QSplitter

from .base_window import _AbstractPlotWindow
from ..plot_widgets import ImageViewF, PlotWidgetF
from ..misc_widgets import make_brush, make_pen
from ...config import config, AnalysisType, plot_labels


class PumpProbeImageView(ImageViewF):
    """PumpProbeImageView class.

    Widget for displaying the on or off image in the pump-probe analysis.
    """
    def __init__(self, on=True, *, parent=None):
        """Initialization.

        :param bool on: True for display the on image while False for
            displaying the off image.
        """
        super().__init__(parent=parent)

        self._on = on

        flag = "On" if on else "Off"
        self.setTitle(f"{flag} (averaged over train)")

    def updateF(self, data):
        """Override."""
        if self._on:
            img = data.pp.image_on
        else:
            img = data.pp.image_off

        if img is None:
            return

        self.setImage(img, auto_range=False, auto_levels=(not self._is_initialized))
        self.updateROI(data)

        if not self._is_initialized:
            self._is_initialized = True


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
            self._on_off_pulse = self.plotCurve(name="On - Off", pen=make_pen("p"))
        else:
            self._on_pulse = self.plotCurve(name="On", pen=make_pen("r"))
            self._off_pulse = self.plotCurve(name="Off", pen=make_pen("b"))

    def updateF(self, data):
        """Override."""
        pp = data.pp
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


class PumpProbeFomPlot(PlotWidgetF):
    """PumpProbeFomPlot class.

    Widget for displaying the evolution of FOM in pump-probe analysis.
    """

    def __init__(self, *, parent=None):
        """Initialization."""
        super().__init__(parent=parent)

        self.setLabel('bottom', "Train ID")
        self.setLabel('left', "FOM (arb. u.)")
        self.setTitle('FOM correlation')

        self._plot = self.plotScatter(brush=make_brush('g'))

    def updateF(self, data):
        """Override."""
        pp = data.corr.pp
        x, y = pp.x, pp.y
        self._plot.setData(x, y)


class PumpProbeWindow(_AbstractPlotWindow):
    """PumpProbeWindow class."""
    _title = "Pump-probe"

    _TOTAL_W, _TOTAL_H = config['GUI']['PLOT_WINDOW_SIZE']

    def __init__(self, *args, **kwargs):
        """Initialization."""
        super().__init__(*args, **kwargs)

        self._on_image = PumpProbeImageView(on=True, parent=self)
        self._off_image = PumpProbeImageView(on=False, parent=self)

        self._pp_fom = PumpProbeFomPlot(parent=self)

        self._pp_onoff = PumpProbeVFomPlot(parent=self)
        self._pp_diff = PumpProbeVFomPlot(diff=True, parent=self)

        self.initUI()

        self.resize(self._TOTAL_W, self._TOTAL_H)
        self.setMinimumSize(0.6*self._TOTAL_W, 0.6*self._TOTAL_H)

        self.update()

    def initUI(self):
        """Override."""
        self._cw = QSplitter()
        left_panel = QSplitter(Qt.Vertical)
        right_panel = QSplitter(Qt.Vertical)
        self._cw.addWidget(left_panel)
        self._cw.addWidget(right_panel)
        self._cw.setSizes([1, 1])
        self.setCentralWidget(self._cw)

        # TODO: determine the split based on image shape
        view = QSplitter(Qt.Vertical)
        view.addWidget(self._on_image)
        view.addWidget(self._off_image)

        left_panel.addWidget(view)
        left_panel.addWidget(self._pp_fom)
        left_panel.setSizes([self._TOTAL_H / 2, self._TOTAL_H / 2])

        right_panel.addWidget(self._pp_onoff)
        right_panel.addWidget(self._pp_diff)

    def initConnections(self):
        """Override."""
        pass
