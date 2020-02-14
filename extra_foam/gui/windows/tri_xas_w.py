"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QSplitter

from .base_window import _AbstractSpecialAnalysisWindow
from ..plot_widgets import RoiImageView, TimedImageViewF, TimedPlotWidgetF
from ..misc_widgets import make_pen
from ...config import config


class TrXasAbsorptionPlot(TimedPlotWidgetF):
    """TrXasAbsorptionPlot class.

    Display absorption(s) binned by time delay.
    """
    def __init__(self, diff=False, *, parent=None):
        """Initialization."""
        super().__init__(parent=parent)

        self._diff = diff

        self.setLabel('left', "Absorption (arb. u.)")
        self.setLabel('bottom', "Delay (arb. u.)")
        self.addLegend(offset=(-40, 20))

        if diff:
            self._a21 = self.plotCurve(name="ROI2/ROI1", pen=make_pen("g"))
        else:
            c = config['GUI_ROI_COLORS']
            self._a13 = self.plotCurve(name="ROI1/ROI3", pen=make_pen(c[0]))
            self._a23 = self.plotCurve(name="ROI2/ROI3", pen=make_pen(c[1]))

    def refresh(self):
        """Override."""
        xas = self._data.trxas

        delay = xas.delay_bin_centers
        if delay is None:
            return

        if self._diff:
            self._a21.setData(delay, xas.a21_stats)
        else:
            self._a13.setData(delay, xas.a13_stats)
            self._a23.setData(delay, xas.a23_stats)


class TrXasHeatmap(TimedImageViewF):
    """TrXasHeatmap class.

    Display absorption binned by delay and energy.
    """
    def __init__(self, *, parent=None):
        """Initialization."""
        super().__init__(has_roi=False, hide_axis=False, parent=parent)
        self.invertY(False)
        self.setAspectLocked(False)

        self.setLabel('bottom', 'Energy (eV)')
        self.setLabel('left', 'Delay (arb. u.)')
        self.setTitle("Absorption (ROI2/ROI1)")

    def refresh(self):
        """Override."""
        xas = self._data.trxas

        energy = xas.energy_bin_centers
        delay = xas.delay_bin_centers
        heat = xas.a21_heat

        if energy is None or delay is None:
            return

        # do not update if FOM is None
        if heat is not None:
            self.setImage(heat,
                          auto_levels=True,
                          auto_range=True,
                          pos=[energy[0], delay[0]],
                          scale=[(energy[-1] - energy[0])/len(energy),
                                 (delay[-1] - delay[0])/len(delay)])


class TrXasWindow(_AbstractSpecialAnalysisWindow):
    """TrXasWindow class."""
    _title = "Tr-XAS"

    _TOTAL_W, _TOTAL_H = config['GUI_PLOT_WINDOW_SIZE']

    def __init__(self, *args, **kwargs):
        """Initialization."""
        super().__init__(*args, **kwargs)

        self._roi1_image = RoiImageView(1, parent=self)
        self._roi2_image = RoiImageView(2, parent=self)
        self._roi3_image = RoiImageView(3, parent=self)

        self._a13_a23 = TrXasAbsorptionPlot(parent=self)
        self._a21 = TrXasAbsorptionPlot(True, parent=self)
        self._a21_heatmap = TrXasHeatmap(parent=self)

        self.initUI()
        self.initConnections()

        self.resize(self._TOTAL_W, self._TOTAL_H)

        self.update()

    def initUI(self):
        """Override."""
        left_panel = QSplitter(Qt.Vertical)
        left_panel.addWidget(self._roi1_image)
        left_panel.addWidget(self._roi2_image)
        left_panel.addWidget(self._roi3_image)

        right_panel = QSplitter(Qt.Vertical)
        right_panel.addWidget(self._a13_a23)
        right_panel.addWidget(self._a21)
        right_panel.addWidget(self._a21_heatmap)
        right_panel.setSizes([self._TOTAL_H / 3.0] * 3)

        self._cw.addWidget(left_panel)
        self._cw.addWidget(right_panel)
        self._cw.setSizes([self._TOTAL_W / 3., self._TOTAL_W / 2.])

    def initConnections(self):
        """Override."""
        pass
