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

from PyQt5.QtGui import QValidator

from .pyqtgraph import ColorMap, intColor, mkPen, mkBrush, QtCore, QtGui
from .pyqtgraph.graphicsItems.GradientEditorItem import Gradients

from .gui_helpers import parse_boundary, parse_ids


class Colors:
    def __init__(self, alpha=255):
        self.r = (215, 25, 28, alpha)  # red
        self.o = (253, 174, 97, alpha)  # orange
        self.y = (255, 255, 191, alpha)  # yellow
        self.c = (171, 217, 233, alpha)  # cyan
        self.b = (44, 123, 182, alpha)  # blue
        self.g = (26, 150, 65, alpha)  # green
        self.p = (94, 60, 153, alpha)  # purple
        self.d = (218, 112, 214, alpha)  # orchid
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
        logger_font.setPointSize(11)
        self.widget.setFont(logger_font)

        self.widget.setReadOnly(True)
        self.widget.setMaximumBlockCount(500)

    def emit(self, record):
        # guard logger from other threads
        if threading.current_thread() is threading.main_thread():
            self.widget.appendPlainText(self.format(record))


class SmartLineEdit(QtGui.QLineEdit):
    """A smart QLineEdit.

    - It is highlighted when modified but not confirmed.
    - It restores the previous valid input when ESC is pressed. Still,
      one must press enter to confirm the value.
    """

    value_changed_sgn = QtCore.pyqtSignal(object)

    def __init__(self, *args, **kwargs):
        """Initialization"""
        super().__init__(*args, **kwargs)

        self._text_modified = False
        self._cached = self.text()

        self.textChanged.connect(self.onTextChanged)

        self.returnPressed.connect(self.onTextChangeConfirmed)
        self.returnPressed.connect(self.onReturnPressed)

    def onTextChanged(self):
        if not self._text_modified:
            self.setStyleSheet(
                f"QLineEdit {{ background: rgb{Colors().o[:3]}}}")
            self._text_modified = True

    def onTextChangeConfirmed(self):
        self.setStyleSheet("QLineEdit { background: rgb(255, 255, 255)}")
        self._text_modified = False

    def setText(self, text):
        """'Press enter' after setText.

        This will remove the background color used when modified.
        """
        super().setText(text)
        self.returnPressed.emit()

    def setTextWithoutSignal(self, text):
        """SetText without signaling returnPressed."""
        super().setText(text)

    def keyPressEvent(self, event):
        """Press ESC to return to the previous valid value."""
        key = event.key()
        if key == QtCore.Qt.Key_Escape:
            self.setTextWithoutSignal(self._cached)
        else:
            return super().keyPressEvent(event)

    def onReturnPressed(self):
        self._cached = self.text()
        if hasattr(self, "Validator"):
            self.value_changed_sgn.emit(self.Validator.parse(self.text()))
        else:
            self.value_changed_sgn.emit(self.text())

    def value(self):
        if hasattr(self, "Validator"):
            return self.Validator.parse(self.text())
        else:
            return self.text()


class SmartBoundaryLineEdit(SmartLineEdit):

    class Validator(QValidator):
        def __init__(self, parent=None):
            super().__init__(parent)

        def validate(self, s, pos):
            try:
                self.parse(s)
                return QValidator.Acceptable, s, pos
            except ValueError:
                return QValidator.Intermediate, s, pos

        @staticmethod
        def parse(s):
            return parse_boundary(s)

    def __init__(self, content, parent=None):
        super().__init__(content, parent=parent)

        try:
            self.Validator.parse(content)
        except ValueError:
            raise

        self._cached = self.text()

        self.setValidator(self.Validator())


class SmartRangeLineEdit(SmartLineEdit):

    value_changed_sgn = QtCore.pyqtSignal(object)

    class Validator(QValidator):
        def __init__(self, parent=None):
            super().__init__(parent)

        def validate(self, s, pos):
            try:
                self.parse(s)
                return QValidator.Acceptable, s, pos
            except ValueError:
                return QValidator.Intermediate, s, pos

        @staticmethod
        def parse(s):
            return parse_ids(s)

    def __init__(self, content, parent=None):
        super().__init__(content, parent=parent)

        try:
            self.Validator.parse(content)
        except ValueError:
            raise

        self._cached = self.text()

        self.setValidator(self.Validator())
