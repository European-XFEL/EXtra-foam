"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from PyQt5.QtCore import Qt
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

        self.setTitle('')
        self._default_x_label = "Bin center (arb. u.)"
        if count:
            self._default_y_label = "Count"
            self._plot = self.plotBar(pen=make_pen('g'), brush=make_brush('b'))
        else:
            self._default_y_label = "FOM (arb. u.)"
            self._plot = self.plotScatter(brush=make_brush('p'))

        self._device_id = ""
        self._ppt = ""

        self.updateLabel(self._device_id, self._ppt)

    def refresh(self):
        """Override."""
        item = self._data.bin[0]

        device_id = item.device_id
        ppt = item.property
        if device_id != self._device_id or ppt != self._ppt:
            self._device_id = device_id
            self._ppt = ppt
            self.updateLabel(device_id, ppt)

        if self._count:
            hist = item.counts
        else:
            hist = item.stats

        self._plot.setData(item.centers, hist)

    def updateLabel(self, device_id, ppt):
        if device_id and ppt:
            new_label = f"{device_id + ' | ' + ppt} (arb. u.)"
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

        self._default_x_label = 'Bin center (arb. u.)'
        self._default_y_label = 'VFOM (arb. u.)'
        self.setTitle('')

        self._device_id = ""
        self._ppt = ""

        self.updateLabel(self._device_id, self._ppt)

    def refresh(self):
        """Override."""
        item = self._data.bin[0]

        device_id = item.device_id
        ppt = item.property
        if device_id != self._device_id or ppt != self._ppt:
            self._device_id = device_id
            self._ppt = ppt
            self.updateLabel(device_id, ppt)

        heatmap = item.heat
        if heatmap is not None:
            h, w = heatmap.shape
            w_range = item.centers
            h_range = item.x

            self.setImage(heatmap,
                          auto_levels=True,
                          auto_range=True,
                          pos=[w_range[0], h_range[0]],
                          scale=[(w_range[-1] - w_range[0])/w,
                                 (h_range[-1] - h_range[0])/h])
        else:
            self.clear()

    def updateLabel(self, device_id, ppt):
        if device_id and ppt:
            new_label = f"{device_id + ' | ' + ppt} (arb. u.)"
        else:
            new_label = self._default_x_label
        self.setLabel('bottom', new_label)

        self.setLabel('left', self._default_y_label)


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

        self._default_x_label = 'Bin center (arb. u.)'
        self._default_y_label = 'Bin center (arb. u.)'
        if count:
            self.setTitle("2D binning (Count)")
        else:
            self.setTitle("2D binning (FOM)")

        self._device_ids = ["", ""]
        self._ppts = ["", ""]

        for dev, ppt, pos in zip(
                self._device_ids, self._ppts, ['bottom', 'left']):
            self.updateLabel(dev, ppt, pos)

    def refresh(self):
        """Override."""
        bin = self._data.bin

        for i in range(2):
            device_id = bin[i].device_id
            ppt = bin[i].property
            if device_id != self._device_ids[i] or ppt != self._ppts[i]:
                self._device_ids[i] = device_id
                self._ppts[i] = ppt
                self.updateLabel(device_id, ppt,
                                 'bottom' if i == 0 else 'left')

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
                          auto_levels=True,
                          auto_range=True,
                          pos=[w_range[0], h_range[0]],
                          scale=[(w_range[-1] - w_range[0])/w,
                                 (h_range[-1] - h_range[0])/h])
        else:
            self.clear()

    def updateLabel(self, device_id, ppt, pos):
        if device_id and ppt:
            new_label = f"{device_id + ' | ' + ppt} (arb. u.)"
        else:
            new_label = self._default_x_label \
                if pos == 'bottom' else self._default_y_label
        self.setLabel(pos, new_label)


class BinningWindow(_AbstractPlotWindow):
    """BinningWindow class.

    Plot data in selected bins.
    """
    _title = "Binning 1D"

    _TOTAL_W, _TOTAL_H = config['GUI']['PLOT_WINDOW_SIZE']

    def __init__(self, *args, **kwargs):
        """Initialization."""
        super().__init__(*args, **kwargs)

        self._bin1d_vfom = Bin1dHeatmap(parent=self)
        self._bin1d_fom = Bin1dHist(parent=self)
        self._bin1d_count = Bin1dHist(count=True, parent=self)

        self._bin2d_value = Bin2dHeatmap(count=False, parent=self)
        self._bin2d_count = Bin2dHeatmap(count=True, parent=self)

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
        pass
