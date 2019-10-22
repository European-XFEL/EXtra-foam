"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

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
        self._n_total_pulses_lb = self.DescriptionLabel(
            "Total # of pulses/train: ")
        self._n_filtered_pulses_lb = self.DescriptionLabel(
            "# of filtered pulses/train: ")

        self._trainid_no = self.NumberLabel(" ")
        self._n_total_pulses_no = self.NumberLabel(" ")
        self._n_filtered_pulses_no = self.NumberLabel(" ")

        if not vertical:
            layout = QtGui.QHBoxLayout()
            layout.addWidget(self._trainid_lb, 1)
            layout.addWidget(self._trainid_no, 2)
            layout.addWidget(self._n_total_pulses_lb, 2)
            layout.addWidget(self._n_total_pulses_no, 1)
            layout.addWidget(self._n_filtered_pulses_lb, 2)
            layout.addWidget(self._n_filtered_pulses_no, 1)
        else:
            layout = QtGui.QGridLayout()
            layout.addWidget(self._trainid_lb, 0, 0)
            layout.addWidget(self._trainid_no, 0, 1)
            layout.addWidget(self._n_total_pulses_lb, 1, 0)
            layout.addWidget(self._n_total_pulses_no, 1, 1)
            layout.addWidget(self._n_filtered_pulses_lb, 2, 0)
            layout.addWidget(self._n_filtered_pulses_no, 2, 1)
        self.setLayout(layout)

        self.reset()

    def reset(self):
        self._set_text()

    def update(self, data):
        """Override."""
        n_total = data.n_pulses
        n_dropped = data.pidx.n_dropped(n_total)
        self._set_text(data.tid, n_total, n_total - n_dropped)

    def _set_text(self, tid="", n_total="", n_filtered=""):
        self._trainid_no.setText(f"{tid}")
        self._n_total_pulses_no.setText(f"{n_total}")
        self._n_filtered_pulses_no.setText(f"{n_filtered}")
