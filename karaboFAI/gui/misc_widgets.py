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

from .pyqtgraph import ColorMap, mkPen, mkBrush, QtCore, QtGui
from .pyqtgraph.graphicsItems.GradientEditorItem import Gradients


class Colors:
    def __init__(self, alpha):
        self.r = (255, 0, 0, alpha)  # red
        self.g = (0, 255, 0, alpha)  # green
        self.b = (0, 0, 255, alpha)  # blue
        self.c = (0, 255, 255, alpha)  # cyan
        self.p = (255, 0, 255, alpha)  # purple
        self.y = (255, 255, 0, alpha)  # yellow


def make_pen(color, width=2, alpha=255, **kwargs):
    """Convenient function for making QPen."""
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

        logger_font = QtGui.QFont()
        logger_font.setPointSize(12)
        self.widget.setFont(logger_font)

        self.widget.setReadOnly(True)
        self.widget.setMaximumBlockCount(500)

    def emit(self, record):
        self.widget.appendPlainText(self.format(record))
