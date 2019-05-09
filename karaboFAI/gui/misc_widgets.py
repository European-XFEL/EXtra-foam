"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

Hold classes of miscellaneous widgets.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import logging
import threading

from .pyqtgraph import ColorMap, intColor, mkPen, mkBrush, QtCore, QtGui
from .pyqtgraph.graphicsItems.GradientEditorItem import Gradients

from .gui_helpers import parse_boundary
from ..logger import logger


class Colors:
    def __init__(self, alpha=255):
        self.r = (215, 25, 28, alpha)  # red
        self.o = (253, 174, 97, alpha)  # orange
        self.y = (255, 255, 191, alpha)  # yellow
        self.c = (171, 217, 233, alpha)  # cyan
        self.b = (44, 123, 182, alpha)  # blue
        self.g = (26, 150, 65, alpha)  # green
        self.p = (175, 141, 195, alpha)  # purple
        self.w = (247, 247, 247, alpha)  # white
        self.e = (186, 186, 186, alpha)  # dark grey


def make_pen(color, width=2, alpha=255, **kwargs):
    """Convenient function for making QPen."""
    if color is None:
        return mkPen(None)

    if isinstance(color, int):
        return mkPen(intColor(color, **kwargs), width=width)

    return mkPen(getattr(Colors(alpha=alpha), color[0]), width=width, **kwargs)


def make_brush(color, alpha=255):
    return mkBrush(getattr(Colors(alpha=alpha), color[0]))


# Valid keys: thermal, flame, yellowy, bipolar, spectrum, cyclic, greyclip, grey
colorMapFactory = \
    {name: ColorMap(*zip(*Gradients[name]["ticks"]))
     for name in Gradients.keys()}


lookupTableFactory = {name: cmap.getLookupTable()
                      for name, cmap in colorMapFactory.items()}


class InputDialogWithCheckBox(QtGui.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

    @classmethod
    def getResult(cls, parent, window_title, input_label, checkbox_label):
        dialog = cls(parent)

        dialog.setWindowTitle(window_title)

        label = QtGui.QLabel(input_label)
        text_le = QtGui.QLineEdit()

        ok_cb = QtGui.QCheckBox(checkbox_label)
        ok_cb.setChecked(True)

        buttons = QtGui.QDialogButtonBox(
            QtGui.QDialogButtonBox.Ok | QtGui.QDialogButtonBox.Cancel,
            QtCore.Qt.Horizontal, dialog
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)

        layout = QtGui.QVBoxLayout()
        layout.addWidget(label)
        layout.addWidget(text_le)
        layout.addWidget(ok_cb)
        layout.addWidget(buttons)
        dialog.setLayout(layout)

        result = dialog.exec_()

        return (text_le.text(), ok_cb.isChecked()), \
            result == QtGui.QDialog.Accepted


class GuiLogger(logging.Handler):
    def __init__(self, parent):
        super().__init__(level=logging.INFO)
        self.widget = QtGui.QPlainTextEdit(parent)

        formatter = logging.Formatter('%(levelname)s - %(message)s')
        self.setFormatter(formatter)

        logger_font = QtGui.QFont()
        logger_font.setPointSize(12)
        self.widget.setFont(logger_font)

        self.widget.setReadOnly(True)
        self.widget.setMaximumBlockCount(500)

    def emit(self, record):
        # guard logger from other threads
        if threading.current_thread() is threading.main_thread():
            self.widget.appendPlainText(self.format(record))


class _SmartLineEdit(QtGui.QLineEdit):
    """A smart QLineEdit.

    - It stores the latest valid value and restore the value if the
      current input is invalid.
    - One can get the parsed result via a signal-slot connection.
    """
    # signal emitted when a new valid input is applied
    value_changed_sgn = QtCore.pyqtSignal(object)

    def __init__(self, content, handler=None, parent=None):
        """Initialization

        :param str content: Initial value of the QLineEdit
        :param callable handler: a handler use to validate and parse the
            content. If handler raises, only ValueError will be handled.
        """
        super().__init__(content, parent=parent)

        self._handler = handler
        if self._handler is None:
            raise NotImplementedError
        else:
            # validate the initial value
            self._handler(content)

        self._cached = content

        self.returnPressed.connect(self.onReturnPressed)

    def onReturnPressed(self):
        try:
            content = self.text()
            validated = self._handler(content)
            self._cached = content
            self.value_changed_sgn.emit(*validated)
        except ValueError as e:
            # restore the cached valid value
            self.setText(self._cached)
            logger.error(repr(e))


class SmartBoundaryLineEdit(_SmartLineEdit):
    value_changed_sgn = QtCore.pyqtSignal(float, float)

    def __init__(self, content, parent=None):
        super().__init__(content, parse_boundary, parent=parent)
