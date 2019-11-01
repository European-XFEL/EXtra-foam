"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from PyQt5 import QtGui, QtCore, QtWidgets

from .base_window import PlotWindow
from ..plot_widgets import ImageViewF, PlotWidgetF
from ...config import config


class SinglePulseImageView(ImageViewF):
    """SinglePulseImageView class.

    Widget for displaying the assembled image of a single pulse.
    """
    def __init__(self, idx, *, parent=None):
        """Initialization."""
        super().__init__(parent=parent)

        self.pulse_index = idx

    def update(self, data):
        """Override."""
        images = data.image.images

        try:
            self.setImage(images[self.pulse_index],
                          auto_levels=(not self._is_initialized))
        except IndexError:
            self.clear()
            return

        self.updateROI(data)

        if not self._is_initialized:
            self._is_initialized = True


class PoiStatisticsWidget(PlotWidgetF):
    """PoiStatisticsWidget class.

    A widget which allows users to monitor the statistics of the FOM of
    the POI pulse.
    """
    def __init__(self, idx, *, parent=None):
        """Initialization."""
        super().__init__(parent=parent)

        self.pulse_index = idx
        self._plot = self.plotBar()

        self.setTitle("FOM Histogram")
        self.setLabel('left', 'Counts')
        self.setLabel('bottom', 'FOM')

    def update(self, data):
        """Override."""
        bin_centers = data.st.poi_fom_bin_center
        if bin_centers is None:
            return

        center = bin_centers[self.pulse_index]
        counts = data.st.poi_fom_count[self.pulse_index]
        if center is None:
            self.reset()
            return
        self._plot.setData(center, counts)


class PulseOfInterestWindow(PlotWindow):
    """PulseOfInterestWindow class."""
    title = "pulse-of-interest"

    _TOTAL_W, _TOTAL_H = config['GUI']['PLOT_WINDOW_SIZE']

    def __init__(self, *args, **kwargs):
        """Initialization."""
        super().__init__(*args, **kwargs)

        self._poi1_img = SinglePulseImageView(0, parent=self)
        self._poi2_img = SinglePulseImageView(0, parent=self)

        self._poi1_statistics = PoiStatisticsWidget(0, parent=self)
        self._poi2_statistics = PoiStatisticsWidget(0, parent=self)

        self.initUI()

        self.resize(self._TOTAL_W, self._TOTAL_H)

        self.setMinimumSize(0.6*self._TOTAL_W, 0.6*self._TOTAL_H)

        self.initConnections()
        self.update()

    def initUI(self):
        """Override."""
        self._cw = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        left_panel = QtWidgets.QSplitter()
        right_panel = QtWidgets.QSplitter()
        self._cw.addWidget(left_panel)
        self._cw.addWidget(right_panel)
        self.setCentralWidget(self._cw)

        left_panel.addWidget(self._poi1_img)
        left_panel.addWidget(self._poi1_statistics)
        # A value smaller than the minimal size hint of the respective
        # widget will be replaced by the value of the hint.
        left_panel.setSizes([self._TOTAL_W, self._TOTAL_W])

        right_panel.addWidget(self._poi2_img)
        right_panel.addWidget(self._poi2_statistics)
        right_panel.setSizes([self._TOTAL_W, self._TOTAL_W])

    def initConnections(self):
        """Override."""
        if self._pulse_resolved:
            mediator = self._mediator
            mediator.poi_index1_sgn.connect(self.onPulseID1Updated)
            mediator.poi_index2_sgn.connect(self.onPulseID2Updated)
            mediator.poi_indices_connected_sgn.emit()

    @QtCore.pyqtSlot(int)
    def onPulseID1Updated(self, value):
        self._poi1_img.pulse_index = value
        self._poi1_statistics.pulse_index = value

    @QtCore.pyqtSlot(int)
    def onPulseID2Updated(self, value):
        self._poi2_img.pulse_index = value
        self._poi2_statistics.pulse_index = value
