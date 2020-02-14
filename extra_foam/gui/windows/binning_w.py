"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from PyQt5.QtCore import Qt, pyqtSlot
from PyQt5.QtWidgets import QSplitter

from .base_window import _AbstractPlotWindow
from ..plot_widgets import TimedPlotWidgetF, TimedImageViewF
from ..misc_widgets import make_brush, make_pen
from ...config import config


class Bin1dHist(TimedPlotWidgetF):
    """Bin1dHist class.

    Widget for visualizing histogram of count for 1D-binning.
    """
    def __init__(self, *, count=False, parent=None):
        """Initialization.

        :param bool count: True for count plot and False for FOM plot.
        """
        super().__init__(parent=parent)

        self._count = count

        self._default_x_label = "Bin center (arb. u.)"
        if count:
            self._default_y_label = "Count"
            self.setTitle('1D binning (count)')
            self._plot = self.plotBar(pen=make_pen('g'), brush=make_brush('b'))
        else:
            self._default_y_label = "FOM (arb. u.)"
            self.setTitle('1D binning (FOM)')
            self._plot = self.plotSta(brush=make_brush('p'))

        self._source = ""

        self.updateLabel()

    def refresh(self):
        """Override."""
        item = self._data.bin[0]

        src = item.source
        if src != self._source:
            self._source = src
            self.updateLabel()

        if self._count:
            hist = item.counts
        else:
            hist = item.stats

        self._plot.setData(item.centers, hist)

    def updateLabel(self):
        src = self._source
        if src:
            new_label = f"{src} (arb. u.)"
        else:
            new_label = self._default_x_label
        self.setLabel('bottom', new_label)

        self.setLabel('left', self._default_y_label)


class Bin1dHeatmap(TimedImageViewF):
    """Bin1dHeatmap class.

    Widget for visualizing the heatmap of 1D binning.
    """

    def __init__(self, *, parent=None):
        """Initialization."""
        super().__init__(has_roi=False, hide_axis=False, parent=parent)

        self.invertY(False)
        self.setAspectLocked(False)

        self._auto_level = True

        self._default_x_label = 'Bin center (arb. u.)'
        self._default_y_label = 'VFOM (arb. u.)'
        self.setTitle('1D binning (VFOM)')

        self._source = ""

        self.updateLabel()

    def refresh(self):
        """Override."""
        item = self._data.bin[0]

        src = item.source
        if src != self._source:
            self._source = src
            self.updateLabel()

        heatmap = item.heat
        if heatmap is not None:
            h, w = heatmap.shape
            w_range = item.centers
            h_range = item.x

            self.setImage(heatmap,
                          auto_levels=self._auto_level,
                          auto_range=False,
                          pos=[w_range[0], h_range[0]],
                          scale=[(w_range[-1] - w_range[0])/w,
                                 (h_range[-1] - h_range[0])/h])
        else:
            self.clear()

        self._auto_level = False

    def updateLabel(self):
        src = self._source
        if src:
            new_label = f"{src} (arb. u.)"
        else:
            new_label = self._default_x_label
        self.setLabel('bottom', new_label)

        self.setLabel('left', self._default_y_label)

    @pyqtSlot()
    def onAutoLevel(self):
        self._auto_level = True


class Bin2dHeatmap(TimedImageViewF):
    """Bin2dHeatmap class.

    Widget for visualizing the heatmap of 2D binning.
    """
    def __init__(self, *, count=False, parent=None):
        """Initialization.

        :param bool count: True for count plot and False for value plot.
        """
        super().__init__(has_roi=False, hide_axis=False, parent=parent)

        self._count = count

        self.invertY(False)
        self.setAspectLocked(False)

        self._auto_level = True

        self._default_x_label = 'Bin center (arb. u.)'
        self._default_y_label = 'Bin center (arb. u.)'
        if count:
            self.setTitle("2D binning (Count)")
        else:
            self.setTitle("2D binning (FOM)")

        self._source_x = ""
        self._source_y = ""

        self.updateXLabel()
        self.updateYLabel()

    def refresh(self):
        """Override."""
        bin = self._data.bin

        src_x = bin[0].source
        if src_x != self._source_x:
            self._source_x = src_x
            self.updateXLabel()

        src_y = bin[1].source
        if src_y != self._source_y:
            self._source_y = src_y
            self.updateYLabel()

        if self._count:
            heatmap = bin.heat_count
        else:
            heatmap = bin.heat

        # do not update if FOM is None
        if heatmap is not None:
            h, w = heatmap.shape
            w_range = bin[0].centers
            h_range = bin[1].centers

            self.setImage(heatmap,
                          auto_levels=self._auto_level,
                          auto_range=False,
                          pos=[w_range[0], h_range[0]],
                          scale=[(w_range[-1] - w_range[0])/w,
                                 (h_range[-1] - h_range[0])/h])
        else:
            self.clear()

        self._auto_level = False

    def updateXLabel(self):
        src = self._source_x
        if src:
            new_label = f"{src} (arb. u.)"
        else:
            new_label = self._default_x_label
        self.setLabel('bottom', new_label)

    def updateYLabel(self):
        src = self._source_y
        if src:
            new_label = f"{src} (arb. u.)"
        else:
            new_label = self._default_y_label
        self.setLabel('left', new_label)

    @pyqtSlot()
    def onAutoLevel(self):
        self._auto_level = True


class BinningWindow(_AbstractPlotWindow):
    """BinningWindow class.

    Plot data in selected bins.
    """
    _title = "Binning 1D"

    _TOTAL_W, _TOTAL_H = config['GUI_PLOT_WINDOW_SIZE']

    def __init__(self, *args, **kwargs):
        """Initialization."""
        super().__init__(*args, **kwargs)

        self._bin1d_vfom = Bin1dHeatmap(parent=self)
        self._bin1d_fom = Bin1dHist(parent=self)
        self._bin1d_count = Bin1dHist(count=True, parent=self)

        self._bin2d_value = Bin2dHeatmap(count=False, parent=self)
        self._bin2d_count = Bin2dHeatmap(count=True, parent=self)

        self.initUI()
        self.initConnections()

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
        self.setCentralWidget(self._cw)

        left_panel.addWidget(self._bin1d_vfom)
        left_panel.addWidget(self._bin1d_fom)
        left_panel.addWidget(self._bin1d_count)
        # A value smaller than the minimal size hint of the respective
        # widget will be replaced by the value of the hint.
        left_panel.setSizes([self._TOTAL_H/2, self._TOTAL_H/3, self._TOTAL_H/6])

        right_panel.addWidget(self._bin2d_value)
        right_panel.addWidget(self._bin2d_count)
        right_panel.setSizes([1, 1])

    def initConnections(self):
        """Override."""
        mediator = self._mediator

        mediator.bin_heatmap_autolevel_sgn.connect(
            self._bin1d_vfom.onAutoLevel)
        mediator.bin_heatmap_autolevel_sgn.connect(
            self._bin2d_value.onAutoLevel)
        mediator.bin_heatmap_autolevel_sgn.connect(
            self._bin2d_count.onAutoLevel)
