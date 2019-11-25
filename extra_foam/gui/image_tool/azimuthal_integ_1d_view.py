"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from PyQt5.QtWidgets import QVBoxLayout, QSplitter

from .simple_image_data import _SimpleImageData
from .base_view import _AbstractImageToolView
from ..ctrl_widgets import AzimuthalIntegCtrlWidget
from ..misc_widgets import make_pen
from ..plot_widgets import ImageAnalysis, PlotWidgetF


class AzimuthalInteg1dPlotWidget(PlotWidgetF):
    """AzimuthalInteg1dPlotWidget class.

    Widget for visualizing the line plot of 1D azimuthal integration result.
    """
    def __init__(self, *, parent=None):
        """Initialization."""
        super().__init__(parent=parent)

        self.setLabel('bottom', "Momentum transfer (1/A)")
        self.setLabel('left', "Scattering signal (arb. u.)")
        self.setTitle(' ')

        self._plot = self.plotCurve(pen=make_pen("p"))

    def updateF(self, data):
        """Override."""
        momentum = data.ai.x
        intensity = data.ai.vfom

        if intensity is None:
            return

        self._plot.setData(momentum, intensity)


class AzimuthalInteg1dView(_AbstractImageToolView):
    """AzimuthalInteg1dView class.

    Widget for visualizing the current image as well as the 1D azimuthal
    integration result. A ctrl widget is included to set up the parameters
    for azimuthal integration.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._image_view = ImageAnalysis(hide_axis=False)
        self._azimuthal_integ_1d_curve = AzimuthalInteg1dPlotWidget()
        self._ctrl_widget = self.parent().createCtrlWidget(
            AzimuthalIntegCtrlWidget)

        self.initUI()

    def initUI(self):
        """Override."""
        view_splitter = QSplitter()
        view_splitter.setHandleWidth(9)
        view_splitter.setChildrenCollapsible(False)
        view_splitter.addWidget(self._image_view)
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
        if auto_update or self._image_view.image is None:
            self._image_view.setImageData(_SimpleImageData(data.image))
            self._azimuthal_integ_1d_curve.updateF(data)
