"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

AbstractCtrlWidget.

Author: Jun Zhu <jun.zhu@xfel.eu>, Ebad Kamil <ebad.kamil@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from ..widgets.pyqtgraph import QtCore, QtGui


class AbstractCtrlWidget(QtGui.QGroupBox):
    GROUP_BOX_STYLE_SHEET = 'QGroupBox:title {' \
                            'border: 1px;' \
                            'subcontrol-origin: margin;' \
                            'subcontrol-position: top left;' \
                            'padding-left: 10px;' \
                            'padding-top: 10px;' \
                            'margin-top: 0.0em;}'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setStyleSheet(self.GROUP_BOX_STYLE_SHEET)

        parent = self.parent()
        if parent is not None:
            parent.registerCtrlWidget(self)

        parent.daq_started_sgn.connect(self.onDaqStarted)
        parent.daq_stopped_sgn.connect(self.onDaqStopped)

        self._disabled_widgets_during_daq = []

    def initUI(self):
        """Initialization of UI."""
        raise NotImplementedError

    @QtCore.pyqtSlot()
    def onDaqStarted(self):
        for widget in self._disabled_widgets_during_daq:
            widget.setEnabled(False)

    @QtCore.pyqtSlot()
    def onDaqStopped(self):
        for widget in self._disabled_widgets_during_daq:
            widget.setEnabled(True)

    def updateSharedParameters(self, log=False):
        """Update shared parameters for control widget.

        :params bool log: True for logging shared parameters and False
            for not.

        Returns bool: True if all shared parameters successfully parsed
            and emitted, otherwise False.

        Return True: If method not implemented in the inherited widget
                     class
        """
        raise NotImplementedError
