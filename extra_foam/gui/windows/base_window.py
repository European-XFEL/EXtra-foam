"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import abc
from weakref import WeakKeyDictionary

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QMainWindow, QSplitter, QWidget

from ..mediator import Mediator
from ... import __version__


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


class _AbstractPlotWindow(QMainWindow, _AbstractWindowMixin):
    """Base class for plot windows.

    Abstract plot window consist of plot widgets.
    """
    _title = ""

    _SPLITTER_HANDLE_WIDTH = 5

    def __init__(self, queue, *, pulse_resolved=True, parent=None):
        """Initialization.

        :param deque queue: data queue.
        :param bool pulse_resolved: whether the related data is
            pulse-resolved or not.
        """
        super().__init__(parent=parent)
        self.setAttribute(Qt.WA_DeleteOnClose, True)
        if parent is not None:
            parent.registerWindow(self)

        self._mediator = Mediator()

        self._queue = queue
        self._pulse_resolved = pulse_resolved

        self._plot_widgets = WeakKeyDictionary()  # book-keeping plot widgets

        try:
            title = parent.title + " - " + self._title
        except AttributeError:
            title = self._title  # for unit test where parent is None
        self.setWindowTitle(title)

        self._cw = QWidget()
        self.setCentralWidget(self._cw)

        self.show()

    def reset(self):
        """Override."""
        for widget in self._plot_widgets:
            widget.reset()

    def updateWidgetsF(self):
        """Override."""
        if len(self._queue) == 0:
            return

        data = self._queue[0]
        for widget in self._plot_widgets:
            widget.updateF(data)

    def registerPlotWidget(self, instance):
        self._plot_widgets[instance] = 1

    def unregisterPlotWidget(self, instance):
        del self._plot_widgets[instance]

    def closeEvent(self, QCloseEvent):
        parent = self.parent()
        if parent is not None:
            parent.unregisterWindow(self)
        super().closeEvent(QCloseEvent)


class _AbstractSatelliteWindow(QMainWindow, _AbstractWindowMixin):
    """Base class for satellite windows.

    A satellite window does not need to access the processed data.
    """
    title = ""

    def __init__(self, parent=None):
        """Initialization."""
        super().__init__(parent=parent)
        self.setAttribute(Qt.WA_DeleteOnClose, True)

        if parent is not None:
            parent.registerSatelliteWindow(self)
            self._mediator = Mediator()
        else:
            self._mediator = None

        try:
            title = parent.title + " - " + self._title
        except AttributeError:
            # for unittest in which parent is None and the case when
            # the window is not opened through the main GUI
            title = f"EXtra-foam {__version__} - " + self._title
        self.setWindowTitle(title)

    def updateWidgetsF(self):
        """Override."""
        # SatelliteWindow should not need 'updateF' method
        raise Exception()

    def reset(self):
        """Override."""
        # SatelliteWindow should not need 'reset' method
        raise Exception()

    def closeEvent(self, QCloseEvent):
        parent = self.parent()
        if parent is not None:
            parent.unregisterSatelliteWindow(self)
        super().closeEvent(QCloseEvent)


class _AbstractSpecialAnalysisWindow(QMainWindow, _AbstractWindowMixin):
    """Base class for special analysis windows."""
    title = ""

    _SPLITTER_HANDLE_WIDTH = 5

    def __init__(self, queue, *, pulse_resolved=True, parent=None):
        """Initialization.

        :param deque queue: data queue.
        :param bool pulse_resolved: whether the related data is
            pulse-resolved or not.
        """
        super().__init__(parent=parent)
        self.setAttribute(Qt.WA_DeleteOnClose, True)
        if parent is not None:
            parent.registerSpecialWindow(self)

        self._mediator = Mediator()

        self._queue = queue
        self._pulse_resolved = pulse_resolved

        try:
            title = parent.title + " - " + self._title
        except AttributeError:
            title = self._title  # for unit test where parent is None
        self.setWindowTitle(title)

        self._plot_widgets = WeakKeyDictionary()  # book-keeping plot widgets

        self._cw = QSplitter()
        self.setCentralWidget(self._cw)

        self.show()

    def reset(self):
        """Override."""
        for widget in self._plot_widgets:
            widget.reset()

    def updateWidgetsF(self):
        """Override."""
        if len(self._queue) == 0:
            return

        data = self._queue[0]
        for widget in self._plot_widgets:
            widget.updateF(data)

    def registerPlotWidget(self, instance):
        self._plot_widgets[instance] = 1

    def unregisterPlotWidget(self, instance):
        del self._plot_widgets[instance]

    def closeEvent(self, QCloseEvent):
        parent = self.parent()
        if parent is not None:
            parent.unregisterSpecialWindow(self)
        super().closeEvent(QCloseEvent)
