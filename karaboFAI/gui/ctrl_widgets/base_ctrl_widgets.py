"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

AbstractCtrlWidget.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from ..pyqtgraph import QtGui


class AbstractCtrlWidget(QtGui.QGroupBox):
    GROUP_BOX_STYLE_SHEET = 'QGroupBox:title {'\
                            'color: #8B008B;' \
                            'border: 1px;' \
                            'subcontrol-origin: margin;' \
                            'subcontrol-position: top left;' \
                            'padding-left: 10px;' \
                            'padding-top: 10px;' \
                            'margin-top: 0.0em;}'

    def __init__(self, title, *, pulse_resolved=True, parent=None):
        """Initialization.

        :param bool pulse_resolved: whether the related data is
            pulse-resolved or not.
        """
        super().__init__(title, parent=parent)
        self.setStyleSheet(self.GROUP_BOX_STYLE_SHEET)

        parent = self.parent()
        if parent is not None:
            parent.registerCtrlWidget(self)

        # widgets whose values are not allowed to change after the "run"
        # button is clicked
        self._non_reconfigurable_widgets = []

        # whether the related detector is pulse resolved or not
        self._pulse_resolved = pulse_resolved

    def initUI(self):
        """Initialization of UI."""
        raise NotImplementedError

    def onBridgeStarted(self):
        for widget in self._non_reconfigurable_widgets:
            widget.setEnabled(False)

    def onBridgeStopped(self):
        for widget in self._non_reconfigurable_widgets:
            widget.setEnabled(True)

    def updateSharedParameters(self):
        """Update shared parameters for control widget.

        :return: None if any of the parameters is invalid. Otherwise, a
            string of to be logged information.
        """
        return True
