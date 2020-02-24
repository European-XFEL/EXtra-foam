"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import abc
from collections import OrderedDict

from PyQt5.QtWidgets import QFrame, QGroupBox

from ..mediator import Mediator
from ...config import Normalizer


class _AbstractCtrlWidgetMixin:
    @abc.abstractmethod
    def initUI(self):
        """Initialization of UI."""
        raise NotImplementedError

    @abc.abstractmethod
    def initConnections(self):
        """Initialization of signal-slot connections."""
        raise NotImplementedError

    @abc.abstractmethod
    def updateMetaData(self):
        """Update metadata belong to this control widget.

        :returns bool: True if all metadata successfully parsed
            and emitted, otherwise False.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def onStart(self):
        raise NotImplementedError

    @abc.abstractmethod
    def onStop(self):
        raise NotImplementedError


class _AbstractCtrlWidget(QFrame, _AbstractCtrlWidgetMixin):

    _available_norms = OrderedDict({
        "": Normalizer.UNDEFINED,
        "AUC": Normalizer.AUC,
        "XGM": Normalizer.XGM,
        "DIGITIZER": Normalizer.DIGITIZER,
        "ROI": Normalizer.ROI,
    })

    def __init__(self, *, pulse_resolved=True, parent=None):
        """Initialization.

        :param bool pulse_resolved: whether the related data is
            pulse-resolved or not.
        """
        super().__init__(parent=parent)

        self._mediator = Mediator()

        # widgets whose values are not allowed to change after the "run"
        # button is clicked
        self._non_reconfigurable_widgets = []

        # whether the related detector is pulse resolved or not
        self._pulse_resolved = pulse_resolved

        self.setFrameStyle(QFrame.StyledPanel)

    def onStart(self):
        for widget in self._non_reconfigurable_widgets:
            widget.setEnabled(False)

    def onStop(self):
        for widget in self._non_reconfigurable_widgets:
            widget.setEnabled(True)


class _AbstractGroupBoxCtrlWidget(QGroupBox, _AbstractCtrlWidgetMixin):
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

        self._mediator = Mediator()

        # widgets whose values are not allowed to change after the "run"
        # button is clicked
        self._non_reconfigurable_widgets = []

        # whether the related detector is pulse resolved or not
        self._pulse_resolved = pulse_resolved

    def onStart(self):
        for widget in self._non_reconfigurable_widgets:
            widget.setEnabled(False)

    def onStop(self):
        for widget in self._non_reconfigurable_widgets:
            widget.setEnabled(True)
