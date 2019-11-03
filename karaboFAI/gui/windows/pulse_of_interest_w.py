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
from ..ctrl_widgets import SmartLineEdit
from ...config import config


class PoiImageView(ImageViewF):
    """PoiImageView class.

    Widget for displaying the assembled image of pulse-of-interest.
    """
    def __init__(self, idx, *, parent=None):
        """Initialization."""
        super().__init__(parent=parent)

        self._index = idx

    def updateWidgetF(self, data):
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

    def setPulseIndex(self, value):
        self._index = value


class PoiStatisticsWidget(PlotWidgetF):
    """PoiStatisticsWidget class.

    A widget which allows users to monitor the statistics of the FOM of
    the POI pulse.
    """
    def __init__(self, idx, *, parent=None):
        """Initialization."""
        super().__init__(parent=parent)

        self._index = idx
        self._plot = self.plotBar()

        self.setTitle("FOM Histogram")
        self.setLabel('left', 'Counts')
        self.setLabel('bottom', 'FOM')

    def updateWidgetF(self, data):
        """Override."""
        bin_centers = data.st.poi_fom_bin_center
        if bin_centers is None:
            return

        center = bin_centers[self._index]
        counts = data.st.poi_fom_count[self._index]
        if center is None:
            self.reset()
            return
        self._plot.setData(center, counts)

    def setPulseIndex(self, value):
        self._index = value


class PoiWidget(QtWidgets.QWidget):
    """PoiWidget class."""
    _index_validator = QtGui.QIntValidator(
        0, config["MAX_N_PULSES_PER_TRAIN"] - 1)

    pulse_index_sgn = QtCore.pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent=parent)

        index = 0
        self._index_le = SmartLineEdit(str(index))
        self._index_le.setValidator(self._index_validator)
        # register in the parent PlotWindow
        self._poi_img = PoiImageView(index, parent=parent)
        self._poi_statistics = PoiStatisticsWidget(index, parent=parent)

        self.initUI()
        self.initConnections()

    def initUI(self):
        ctrl_layout = QtGui.QHBoxLayout()
        ctrl_layout.addWidget(QtWidgets.QLabel("Pulse index: "))
        ctrl_layout.addWidget(self._index_le)
        ctrl_layout.addStretch(1)

        layout = QtGui.QGridLayout()
        layout.addLayout(ctrl_layout, 0, 0, 1, 2)
        layout.addWidget(self._poi_img, 1, 0, 1, 1)
        layout.addWidget(self._poi_statistics, 1, 1, 1, 1)
        self.setLayout(layout)

    def initConnections(self):
        self._index_le.returnPressed.connect(
            lambda: self._onPulseIndexUpdate(int(self._index_le.text())))

    def _onPulseIndexUpdate(self, value):
        self._poi_img.setPulseIndex(value)
        self._poi_statistics.setPulseIndex(value)
        self.pulse_index_sgn.emit(value)

    def updatePulseIndex(self):
        self._index_le.returnPressed.emit()


class PulseOfInterestWindow(PlotWindow):
    """PulseOfInterestWindow class."""
    title = "pulse-of-interest"

    _TOTAL_W, _TOTAL_H = config['GUI']['PLOT_WINDOW_SIZE']

    def __init__(self, *args, **kwargs):
        """Initialization."""
        super().__init__(*args, **kwargs)

        self._poi_widget1 = PoiWidget(parent=self)
        self._poi_widget2 = PoiWidget(parent=self)

        self.initUI()
        self.initConnections()

        self.resize(self._TOTAL_W, self._TOTAL_H)

        self.setMinimumSize(0.6*self._TOTAL_W, 0.6*self._TOTAL_H)

        self.update()

    def initUI(self):
        """Override."""
        self._cw = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self._cw.addWidget(self._poi_widget1)
        self._cw.addWidget(self._poi_widget2)
        self.setCentralWidget(self._cw)
        self._cw.setHandleWidth(self._SPLITTER_HANDLE_WIDTH)

    def initConnections(self):
        """Override."""
        self._poi_widget1.pulse_index_sgn.connect(
            lambda x: self._mediator.onPoiIndexChange(1, x))
        self._poi_widget1.updatePulseIndex()

        self._poi_widget2.pulse_index_sgn.connect(
            lambda x: self._mediator.onPoiIndexChange(2, x))
        self._poi_widget2.updatePulseIndex()
