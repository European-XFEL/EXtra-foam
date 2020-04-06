"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from string import Template

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIntValidator
from PyQt5.QtWidgets import QGridLayout, QSplitter

from .module_scan_proc import ModuleScanProcessor
from .special_analysis_base import (
    create_special, QThreadFoamClient, _BaseAnalysisCtrlWidgetS,
    _SpecialAnalysisBase
)
from ..gui.plot_widgets import (
    HistMixin, ImageViewF, PlotWidgetF
)
from ..gui.misc_widgets import FColor
from ..gui.ctrl_widgets.smart_widgets import (
    SmartBoundaryLineEdit, SmartLineEdit, SmartSliceLineEdit
)


class ModuleScanCtrlWidget(_BaseAnalysisCtrlWidgetS):
    """Detector module scan control widget."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._non_reconfigurable_widgets = [
        ]

        self.initUI()
        self.initConnections()

    def initUI(self):
        """Override."""
        layout = QGridLayout()

        self.setLayout(layout)

    def initConnections(self):
        """Override."""
        pass


class ModuleScanRoiFomPlot(PlotWidgetF):
    """ModuleScanRoiFomPlot class.

    Visualize ROI FOMs of all modules.
    """
    def __init__(self, *, parent=None):
        """Initialization."""
        super().__init__(parent=parent)

        self.setLabel('left', "ROI FOM")
        self.setLabel('bottom', "Pixel")
        self.addLegend(offset=(-40, 20))

        self._mean = self.plotCurve(name="mean", pen=FColor.mkPen("p"))

    def updateF(self, data):
        """Override."""
        self._mean.setData(data['mean'])


@create_special(ModuleScanCtrlWidget, ModuleScanProcessor, QThreadFoamClient)
class ModuleScanWindow(_SpecialAnalysisBase):
    """Main GUI for module scan."""

    _title = "Module scan"
    _long_title = "Area detector module scan analysis"

    def __init__(self, topic):
        super().__init__(topic, with_dark=False)

        self._scan_plot = ModuleScanRoiFomPlot(parent=self)

        self.initUI()
        self.initConnections()
        self.startWorker()

    def initUI(self):
        """Override."""
        right_panel = QSplitter(Qt.Vertical)
        right_panel.addWidget(self._scan_plot)

        self._cw.addWidget(self._left_panel)
        self._cw.addWidget(right_panel)
        self._cw.setSizes(
            [self._TOTAL_W / 4, 3 * self._TOTAL_W / 4])

        self.resize(self._TOTAL_W, self._TOTAL_H)

    def initConnections(self):
        """Override."""
        pass
