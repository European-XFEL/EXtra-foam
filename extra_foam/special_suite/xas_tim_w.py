"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIntValidator
from PyQt5.QtWidgets import QGridLayout, QSplitter

from .xas_tim_proc import XasTimProcessor
from .special_analysis_base import (
    create_special, QThreadKbClient, _BaseAnalysisCtrlWidgetS,
    _SpecialAnalysisBase
)


class XasTimCtrlWidget(_BaseAnalysisCtrlWidgetS):
    """Xas-Tim control widget.

    XAS-TIM stands for X-ray Absorption Spectroscopy with transmission
    intensity monitor.
    """

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


@create_special(XasTimCtrlWidget, XasTimProcessor, QThreadKbClient)
class XasTimWindow(_SpecialAnalysisBase):
    """Main GUI for XAS-TIM analysis."""

    _title = "XAS-TIM"
    _long_title = "X-ray Absorption Spectroscopy with transmission " \
                  "intensity monitor"

    def __init__(self, topic):
        super().__init__(topic)

        self.initUI()
        self.initConnections()

        self.resize(self._TOTAL_W, self._TOTAL_H)

    def initUI(self):
        pass

    def initConnections(self):
        pass
