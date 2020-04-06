"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from PyQt5.QtWidgets import QVBoxLayout, QSplitter, QTabWidget

from .simple_image_data import _SimpleImageData
from .base_view import _AbstractImageToolView, create_imagetool_view
from ..ctrl_widgets import AzimuthalIntegCtrlWidget
from ..misc_widgets import FColor
from ..plot_widgets import ImageAnalysis, ImageViewF, PlotWidgetF
from ...config import AnalysisType, plot_labels


class AzimuthalInteg1dPlot(PlotWidgetF):
    """AzimuthalInteg1dPlot class.

    Widget for visualizing the line plot of 1D azimuthal integration result.
    """
    def __init__(self, *, parent=None):
        """Initialization."""
        super().__init__(parent=parent, show_indicator=True)

        x_label, y_label = plot_labels[AnalysisType.AZIMUTHAL_INTEG]
        self.setLabel('bottom', x_label)
        self.setLabel('left', y_label)
        self.setTitle(' ')

        self._plot = self.plotCurve(pen=FColor.mkPen("p"))

    def updateF(self, data):
        """Override."""
        momentum, intensity = data.ai.x, data.ai.y

        if intensity is None:
            return

        self._plot.setData(momentum, intensity)


@create_imagetool_view(AzimuthalIntegCtrlWidget)
class AzimuthalInteg1dView(_AbstractImageToolView):
    """AzimuthalInteg1dView class.

    Widget for visualizing the current image as well as the 1D azimuthal
    integration result. A ctrl widget is included to set up the parameters
    for azimuthal integration.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._corrected = ImageAnalysis(hide_axis=False)
        self._q_view = ImageViewF(hide_axis=False)
        self._azimuthal_integ_1d_curve = AzimuthalInteg1dPlot()

        self.initUI()

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

        layout = QVBoxLayout()
        layout.addWidget(view_splitter)
        layout.addWidget(self._ctrl_widget)
        self.setLayout(layout)

    def initConnections(self):
        """Override."""
        pass

    def updateF(self, data, auto_update):
        """Override."""
        if auto_update or self._corrected.image is None:
            self._corrected.setImageData(_SimpleImageData(data.image))
            self._q_view.setImage(data.ai.q_map, auto_range=True, auto_levels=True)
            self._azimuthal_integ_1d_curve.updateF(data)

    def onActivated(self):
        """Override."""
        self._mediator.registerAnalysis(AnalysisType.AZIMUTHAL_INTEG)

    def onDeactivated(self):
        """Override."""
        self._mediator.unregisterAnalysis(AnalysisType.AZIMUTHAL_INTEG)
