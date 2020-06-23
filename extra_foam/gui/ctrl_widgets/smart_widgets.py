"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from PyQt5.QtCore import pyqtSignal, QRegExp, Qt
from PyQt5.QtGui import QRegExpValidator, QValidator
from PyQt5.QtWidgets import QLineEdit

from ..misc_widgets import FColor
from ..gui_helpers import parse_boundary, parse_id, parse_slice


class SmartLineEdit(QLineEdit):
    """A smart QLineEdit.

    - It is highlighted when modified but not confirmed.
    - It restores the previous valid input when ESC is pressed. Still,
      one must press enter to confirm the value.
    """

    value_changed_sgn = pyqtSignal(object)

    def __init__(self, *args, **kwargs):
        """Initialization"""
        super().__init__(*args, **kwargs)

        self._text_modified = False
        self._cached = self.text()

        self.textChanged.connect(self.onTextChanged)

        self.returnPressed.connect(self.onReturnPressed)

    def onTextChanged(self):
        if not self._text_modified:
            self.setStyleSheet(
                f"QLineEdit {{ background: rgb{FColor.o}}}")
            self._text_modified = True

    def setText(self, text):
        """'Press enter after setText.

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
        if key == Qt.Key_Escape:
            self.setTextWithoutSignal(self._cached)
        else:
            return super().keyPressEvent(event)

    def onReturnPressed(self):
        self._cached = self.text()
        if hasattr(self, "Validator"):
            self.value_changed_sgn.emit(self.Validator.parse(self.text()))
        else:
            self.value_changed_sgn.emit(self.text())

        self.setStyleSheet("QLineEdit { background: rgb(255, 255, 255)}")
        self._text_modified = False

    def value(self):
        if hasattr(self, "Validator"):
            return self.Validator.parse(self.text())
        else:
            return self.text()


class SmartStringLineEdit(SmartLineEdit):
    """SmartStringLineEdit class.

    Prevent from entering empty string.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # match any string that contains at least one non-space character.
        #
        # ^ anchors the search at the start of the string.
        #
        # (?!\s*$), a so-called negative lookahead, asserts that it's
        # impossible to match only whitespace characters until the end
        # of the string.
        #
        # .+ will then actually do the match. It will match anything
        # (except newline) up to the end of the string.
        self.setValidator(QRegExpValidator(QRegExp('^(?!\s*$).+')))


class SmartBoundaryLineEdit(SmartLineEdit):

    class Validator(QValidator):
        def __init__(self, parent=None):
            super().__init__(parent=parent)

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


class SmartIdLineEdit(SmartLineEdit):

    value_changed_sgn = pyqtSignal(object)

    class Validator(QValidator):
        def __init__(self, parent=None):
            super().__init__(parent=parent)

        def validate(self, s, pos):
            try:
                self.parse(s)
                return QValidator.Acceptable, s, pos
            except ValueError:
                return QValidator.Intermediate, s, pos

        @staticmethod
        def parse(s):
            return parse_id(s)

    def __init__(self, content, parent=None):
        super().__init__(content, parent=parent)

        try:
            self.Validator.parse(content)
        except ValueError:
            raise

        self._cached = self.text()

        self.setValidator(self.Validator())


class SmartSliceLineEdit(SmartLineEdit):

    value_changed_sgn = pyqtSignal(object)

    # TODO: make a base class for this
    class Validator(QValidator):
        def __init__(self, parent=None):
            super().__init__(parent=parent)

        def validate(self, s, pos):
            try:
                self.parse(s)
                return QValidator.Acceptable, s, pos
            except ValueError:
                return QValidator.Intermediate, s, pos

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
