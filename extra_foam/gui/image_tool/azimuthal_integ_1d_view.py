"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import numpy as np

from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QVBoxLayout, QSplitter, QTabWidget

from .base_view import _AbstractImageToolView, create_imagetool_view
from ..ctrl_widgets import AzimuthalIntegCtrlWidget
from ..misc_widgets import FColor
from ..plot_widgets import ImageViewF, PlotWidgetF, Crosshair
from ...algorithms import find_peaks_1d
from ...config import AnalysisType, plot_labels


class AzimuthalInteg1dPlot(PlotWidgetF):
    """AzimuthalInteg1dPlot class.

    Widget for visualizing the line plot of 1D azimuthal integration result.
    """
    def __init__(self, *, parent=None):
        """Initialization."""
        super().__init__(parent=parent)

        x_label, y_label = plot_labels[AnalysisType.AZIMUTHAL_INTEG]
        self.setLabel('bottom', x_label)
        self.setLabel('left', y_label)
        self.setTitle('Azimuthal integration')

        self._plot = self.plotCurve(pen=FColor.mkPen("p"))
        self._peaks = self.plotScatter(
            pen=FColor.mkPen("g"), brush=FColor.mkBrush(None), symbol="o",
            size=18)
        self._fitted = self.plotCurve(pen=FColor.mkPen("g"))
        self._center_of_mass = self.plotScatter(pen=FColor.mkPen("r"), symbol="o", size=18)

    def updateF(self, data):
        """Override."""
        ai = data.ai
        momentum, intensity = ai.x, ai.y

        if intensity is None:
            return

        self._plot.setData(momentum, intensity)

        peaks = ai.peaks
        if peaks is None:
            self._peaks.setData([], [])
            self.setAnnotationList([], [])
        else:
            self._peaks.setData(momentum[peaks], intensity[peaks])
            self.setAnnotationList(momentum[peaks], intensity[peaks])

        if ai.center_of_mass is not None:
            com_x, com_y = ai.center_of_mass

            if com_x == np.nan or com_y == np.nan:
                self._center_of_mass.setData([], [])
            else:
                self._center_of_mass.setData([com_x], [com_y])

    def data(self):
        return self._plot.data()

    def setFitted(self, x, y):
        self._fitted.setData(x, y)


@create_imagetool_view(AzimuthalIntegCtrlWidget)
class AzimuthalInteg1dView(_AbstractImageToolView):
    """AzimuthalInteg1dView class.

    Widget for visualizing the current image as well as the 1D azimuthal
    integration result. A ctrl widget is included to set up the parameters
    for azimuthal integration.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._corrected = ImageViewF(hide_axis=False)
        self._corrected.setTitle("Averaged over train")

        self._q_view = ImageViewF(hide_axis=False)
        self._q_view.setTitle("q-map")
        self._q_view.setMouseHoverValueRoundingDecimals(4)

        self._azimuthal_integ_1d_curve = AzimuthalInteg1dPlot()

        self._crosshair = Crosshair()
        self._crosshair.hide()

        self.initUI()
        self.initConnections()

    def initUI(self):
        """Override."""
        view_tab = QTabWidget()
        view_tab.setTabPosition(QTabWidget.TabPosition.South)
        view_tab.addTab(self._corrected, "Corrected")
        view_tab.addTab(self._q_view, "Momentum transfer (q)")

        view_splitter = QSplitter()
        view_splitter.setChildrenCollapsible(False)
        view_splitter.addWidget(view_tab)
        view_splitter.addWidget(self._azimuthal_integ_1d_curve)
        view_splitter.setSizes([int(1e6), int(1e6)])

        layout = QVBoxLayout()
        layout.addWidget(view_splitter)
        layout.addWidget(self._ctrl_widget)
        layout.setContentsMargins(1, 1, 1, 1)
        self.setLayout(layout)

        self._corrected._plot_widget.addItem(self._crosshair)

    def initConnections(self):
        """Override."""
        self._ctrl_widget.cx_changed_sgn.connect(lambda x: self.onBeamCenterChanged(x, None))
        self._ctrl_widget.cy_changed_sgn.connect(lambda y: self.onBeamCenterChanged(None, y))
        self._ctrl_widget.fit_curve_sgn.connect(self._onCurveFit)
        self._ctrl_widget.clear_fitting_sgn.connect(self._onClearFitting)

    def onBeamCenterChanged(self, x, y):
        if x is None:
            x = float(self._ctrl_widget._cx_le.text())
        if y is None:
            y = float(self._ctrl_widget._cy_le.text())

        self._crosshair.setPos(x, y)

    @pyqtSlot()
    def _onCurveFit(self):
        x, y = self._ctrl_widget.fitCurve(*self._azimuthal_integ_1d_curve.data())
        self._azimuthal_integ_1d_curve.setFitted(x, y)

    @pyqtSlot()
    def _onClearFitting(self):
        self._azimuthal_integ_1d_curve.setFitted([], [])

    def updateF(self, data, auto_update):
        """Override."""
        if auto_update or self._corrected.image is None:
            self._corrected.setImage(data.image.masked_mean)
            self._q_view.setImage(data.ai.q_map, auto_levels=True)
            self._azimuthal_integ_1d_curve.updateF(data)

            # The crosshair is hidden on start-up, but once an image is
            # processed we set reasonable defaults for the beam center and
            # enable it.
            if not self._crosshair.isVisible() and data.image.masked_mean is not None:
                height, width = data.image.masked_mean.shape
                self._ctrl_widget._cx_le.setText(str(width / 2))
                self._ctrl_widget._cy_le.setText(str(height / 2))
                self._crosshair.show()

    def onActivated(self):
        """Override."""
        self._mediator.registerAnalysis(AnalysisType.AZIMUTHAL_INTEG)
        self._mediator.registerAnalysis(AnalysisType.AZIMUTHAL_INTEG_COM)

    def onDeactivated(self):
        """Override."""
        self._mediator.unregisterAnalysis(AnalysisType.AZIMUTHAL_INTEG)
        self._mediator.unregisterAnalysis(AnalysisType.AZIMUTHAL_INTEG_COM)
