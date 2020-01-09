"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import abc

from PyQt5.QtWidgets import QFrame

from ..mediator import Mediator


class _AbstractImageToolView(QFrame):
    """_AbstractImageToolView class.

    Base class for all the "views" in ImageToolWindow.
    """
    def __init__(self, *, pulse_resolved=True, parent=None):
        """Initialization.

        :param bool pulse_resolved: whether the related data is
            pulse-resolved or not.
        """
        super().__init__(parent=parent)

        self._pulse_resolved = pulse_resolved

        self._mediator = Mediator()

    @abc.abstractmethod
    def initUI(self):
        """Initialization of UI."""
        raise NotImplementedError

    @abc.abstractmethod
    def initConnections(self):
        """Initialization of signal-slot connections."""
        raise NotImplementedError

    def updateF(self, data, auto_update):
        """This method is called by ImageTool."""
        # Views in ImageTool are not registered and updated blindly.
        raise NotImplementedError

    def onActivated(self):
        """Handler when the view is activated."""
        pass

    def onDeactivated(self):
        """Handler when the view is deactivated."""
        pass
