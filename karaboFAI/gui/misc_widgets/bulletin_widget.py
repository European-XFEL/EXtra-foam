"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

BulletinWidget.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from ..pyqtgraph import QtGui

from ..misc_widgets import Colors


class BulletinWidget(QtGui.QWidget):
    """BulletinWidget class."""
    class DescriptionLabel(QtGui.QLabel):
        def __init__(self, text, parent=None):
            super().__init__(text, parent=parent)
            self.setFont(QtGui.QFont("Times", 16))

    class NumberLabel(QtGui.QLabel):
        def __init__(self, text, parent=None):
            super().__init__(text, parent=parent)
            self.setFont(QtGui.QFont("Times", 20, QtGui.QFont.Bold))
            self.setStyleSheet(f"color: rgb{Colors().p[:3]};")

    def __init__(self, *, pulse_resolved=True, vertical=False, parent=None):
        """Initialization.

        :param bool pulse_resolved: whether the related data is
            pulse-resolved or not.
        :param bool vertical: False for horizontal layout and True for
            vertical layout.
        """
        super().__init__(parent=parent)
        parent.registerPlotWidget(self)

        self._pulse_resolved = pulse_resolved

        self._trainid_lb = self.DescriptionLabel("Train ID: ")
        self._npulses_lb = self.DescriptionLabel("Number of images per train: ")
        self._moving_average_lb = self.DescriptionLabel("Moving average count: ")

        self._trainid_no = self.NumberLabel(" ")
        self._npulses_no = self.NumberLabel(" ")
        self._moving_average_no = self.NumberLabel(" ")

        if not vertical:
            layout = QtGui.QHBoxLayout()
            layout.addWidget(self._trainid_lb, 1)
            layout.addWidget(self._trainid_no, 2)
            layout.addWidget(self._npulses_lb, 2)
            layout.addWidget(self._npulses_no, 1)
            layout.addWidget(self._moving_average_lb, 2)
            layout.addWidget(self._moving_average_no, 1)
        else:
            layout = QtGui.QGridLayout()
            layout.addWidget(self._trainid_lb, 0, 0)
            layout.addWidget(self._trainid_no, 0, 1)
            layout.addWidget(self._npulses_lb, 1, 0)
            layout.addWidget(self._npulses_no, 1, 1)
            layout.addWidget(self._moving_average_lb, 2, 0)
            layout.addWidget(self._moving_average_no, 2, 1)
        self.setLayout(layout)

        self.reset()

    def reset(self):
        self._set_text()

    def update(self, data):
        """Override."""
        self._set_text(data.tid, data.image.n_images, data.image.ma_count)

    def _set_text(self, tid="", n_pulses="", ma_count=""):
        self._trainid_no.setText(f"{tid}")
        self._npulses_no.setText(f"{n_pulses}")
        self._moving_average_no.setText(f"{ma_count}")
