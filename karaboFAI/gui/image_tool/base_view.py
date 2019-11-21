"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import abc

from PyQt5.QtWidgets import QFrame


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

        self._image_view = None
        self._pulse_resolved = pulse_resolved

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

    @property
    def imageView(self):
        return self._image_view
