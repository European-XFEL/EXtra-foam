"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

ImageCtrlWidget.

Author: Jun Zhu <jun.zhu@xfel.eu>, Ebad Kamil <ebad.kamil@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from ..pyqtgraph import QtCore, QtGui


class ImageCtrlWidget(QtGui.QWidget):
    """Widget inside the action bar for masking image."""

    moving_avg_window_sgn = QtCore.pyqtSignal(int)

    def __init__(self, parent=None):
        """Initialization"""
        super().__init__(parent=parent)

        self._moving_avg_le = QtGui.QLineEdit(str(1))
        self._moving_avg_le.setValidator(QtGui.QIntValidator(1, 1000000))
        self._moving_avg_le.setMinimumWidth(60)
        self._moving_avg_le.returnPressed.connect(lambda:
            self.moving_avg_window_sgn.emit(int(self._moving_avg_le.text())))

        self.initUI()

        self.setFixedSize(self.minimumSizeHint())

    def initUI(self):
        layout = QtGui.QHBoxLayout()
        layout.addWidget(QtGui.QLabel("Moving average: "))
        layout.addWidget(self._moving_avg_le)

        self.setLayout(layout)
        self.layout().setContentsMargins(2, 1, 2, 1)