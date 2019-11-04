"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import abc
from weakref import WeakKeyDictionary

from PyQt5 import QtGui, QtWidgets

from ..mediator import Mediator


class _AbstractWindowMixin:
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
        """Update metadata affected by this window.

        :returns bool: True if all metadata successfully parsed
            and emitted, otherwise False.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def reset(self):
        """Reset data in widgets.

        This method is called by the main GUI.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def updateWidgetsF(self):
        """Update widgets.

        This method is called by the main GUI.
        """
        raise NotImplementedError


class _AbstractWindow(QtGui.QMainWindow, _AbstractWindowMixin):
    """Base class for various stand-alone windows.

    All the stand-alone windows should follow the interface defined
    in this abstract class.
    """
    title = ""

    _SPLITTER_HANDLE_WIDTH = 5

    def __init__(self, data=None, *, pulse_resolved=True, parent=None):
        """Initialization.

        :param Data4Visualization data: the data shared by widgets
            and windows.
        :param bool pulse_resolved: whether the related data is
            pulse-resolved or not.
        """
        super().__init__(parent=parent)
        if parent is not None:
            parent.registerWindow(self)

        self._mediator = Mediator()

        self._data = data
        self._pulse_resolved = pulse_resolved

        try:
            if self.title:
                title = parent.title + " - " + self.title
            else:
                title = parent.title

            self.setWindowTitle(title)
        except AttributeError:
            # for unit test where parent is None
            self.setWindowTitle(self.title)

        self._cw = QtGui.QWidget()
        self.setCentralWidget(self._cw)

        self.show()

    def closeEvent(self, QCloseEvent):
        parent = self.parent()
        if parent is not None:
            parent.unregisterWindow(self)
        super().closeEvent(QCloseEvent)


class _AbstractPlotWindow(_AbstractWindow):
    """Abstract plot window consist of plot widgets."""
    def __init__(self, *args, **kwargs):
        """Initialization."""
        super().__init__(*args, **kwargs)

        self._plot_widgets = WeakKeyDictionary()  # book-keeping plot widgets

    def updateWidgetsF(self):
        """Update widgets.

        This method is called by the main GUI.
        """
        if self._data is None:
            return

        data = self._data.get()
        if data is None:
            return

        for widget in self._plot_widgets:
            widget.updateF(data)

    def reset(self):
        """Reset data in widgets.

        This method is called by the main GUI.
        """
        for widget in self._plot_widgets:
            widget.reset()

    def registerPlotWidget(self, instance):
        self._plot_widgets[instance] = 1

    def unregisterPlotWidget(self, instance):
        del self._plot_widgets[instance]


class _AbstractSatelliteWindow(QtGui.QMainWindow, _AbstractWindowMixin):
    """Base class for Satellite windows.

    A satellite window does not need to access the processed data.
    """
    title = ""

    def __init__(self, parent=None):
        """Initialization."""
        super().__init__(parent=parent)
        if parent is not None:
            parent.registerSatelliteWindow(self)

        self._mediator = Mediator()

        try:
            if self.title:
                title = parent.title + " - " + self.title
            else:
                title = parent.title

            self.setWindowTitle(title)
        except AttributeError:
            # for unit test where parent is None
            self.setWindowTitle(self.title)

    def updateWidgetsF(self):
        """Override."""
        pass

    def reset(self):
        """Override."""
        pass

    def updateMetaData(self):
        """Override."""
        return True

    def closeEvent(self, QCloseEvent):
        parent = self.parent()
        if parent is not None:
            parent.unregisterSatelliteWindow(self)
        super().closeEvent(QCloseEvent)
