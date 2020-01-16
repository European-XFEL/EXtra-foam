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
from ..plot_widgets import ImageViewF, TimedPlotWidgetF
from ...config import config


class PoiImageView(ImageViewF):
    """PoiImageView class.

    Widget for displaying the assembled image of pulse-of-interest.
    """
    def __init__(self, idx, *, parent=None):
        """Initialization."""
        super().__init__(parent=parent)

        self._index = idx

    def updateF(self, data):
        """Override."""
        try:
            img = data.image.images[self._index]
            self.setImage(img, auto_levels=(not self._is_initialized))
        except (IndexError, TypeError):
            self.clear()
            return

        self.updateROI(data)

        if not self._is_initialized:
            self._is_initialized = True

    def setPulseIndex(self, idx):
        self._index = idx
        self.setTitle(f"Pulse-of-interest {idx}")


class PoiHist(TimedPlotWidgetF):
    """PoiHist class.

    A widget which monitors the histogram of the FOM of the POI pulse.
    """
    def __init__(self, idx, *, parent=None):
        """Initialization."""
        super().__init__(parent=parent)

        self._index = idx
        self._plot = self.plotBar()

        self.setTitle("FOM Histogram")
        self.setLabel('left', 'Counts')
        self.setLabel('bottom', 'FOM')

    def refresh(self):
        """Override."""
        try:
            hist, bin_centers = self._data.hist[self._index]
        except KeyError:
            self.reset()
            return

        self._plot.setData(bin_centers, hist)

    def setPulseIndex(self, idx):
        self._index = idx


class PulseOfInterestWindow(_AbstractPlotWindow):
    """PulseOfInterestWindow class."""
    _title = "Pulse-of-interest"

    _TOTAL_W, _TOTAL_H = config['GUI']['PLOT_WINDOW_SIZE']

    def __init__(self, *args, **kwargs):
        """Initialization."""
        super().__init__(*args, **kwargs)

        self._poi_imgs = [PoiImageView(0, parent=self),
                          PoiImageView(0, parent=self)]
        self._poi_hists = [PoiHist(0, parent=self),
                           PoiHist(0, parent=self)]

        self.initUI()
        self.initConnections()

        self.resize(self._TOTAL_W, self._TOTAL_H)

        self.setMinimumSize(0.6*self._TOTAL_W, 0.6*self._TOTAL_H)

        self.update()

    def initUI(self):
        """Override."""
        self._cw = QSplitter(Qt.Vertical)

        for img, hist in zip(self._poi_imgs, self._poi_hists):
            w = QSplitter()
            w.addWidget(img)
            w.addWidget(hist)
            w.setSizes([self._TOTAL_W / 2, self._TOTAL_W / 2])
            self._cw.addWidget(w)
        self.setCentralWidget(self._cw)
        self._cw.setHandleWidth(self._SPLITTER_HANDLE_WIDTH)

    def initConnections(self):
        """Override."""
        self._mediator.poi_index_change_sgn.connect(self._updatePoiIndex)
        self._mediator.poi_window_initialized_sgn.emit()

    def _updatePoiIndex(self, poi_idx, pulse_idx):
        self._poi_imgs[poi_idx].setPulseIndex(pulse_idx)
        self._poi_hists[poi_idx].setPulseIndex(pulse_idx)
