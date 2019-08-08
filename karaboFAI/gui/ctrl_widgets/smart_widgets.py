"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

SmartLineEdit

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from PyQt5 import QtCore, QtGui

from ..misc_widgets import Colors
from ..gui_helpers import parse_boundary, parse_ids, parse_slice


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

    class Validator(QtGui.QValidator):
        def __init__(self, parent=None):
            super().__init__(parent)

        def validate(self, s, pos):
            try:
                self.parse(s)
                return QtGui.QValidator.Acceptable, s, pos
            except ValueError:
                return QtGui.QValidator.Intermediate, s, pos

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

    class Validator(QtGui.QValidator):
        def __init__(self, parent=None):
            super().__init__(parent)

        def validate(self, s, pos):
            try:
                self.parse(s)
                return QtGui.QValidator.Acceptable, s, pos
            except ValueError:
                return QtGui.QValidator.Intermediate, s, pos

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


class SmartSliceLineEdit(SmartLineEdit):

    value_changed_sgn = QtCore.pyqtSignal(object)

    # TODO: make a base class for this
    class Validator(QtGui.QValidator):
        def __init__(self, parent=None):
            super().__init__(parent)

        def validate(self, s, pos):
            try:
                self.parse(s)
                return QtGui.QValidator.Acceptable, s, pos
            except ValueError:
                return QtGui.QValidator.Intermediate, s, pos

        @staticmethod
        def parse(s):
            return parse_slice(s)

    def __init__(self, content, parent=None):
        super().__init__(content, parent=parent)

        try:
            self.Validator.parse(content)
        except ValueError:
            raise

        self._cached = self.text()

        self.setValidator(self.Validator())
