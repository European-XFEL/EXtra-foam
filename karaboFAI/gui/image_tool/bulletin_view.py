"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import QGridLayout, QLabel, QLCDNumber

from .base_view import _AbstractImageToolView


class BulletinView(_AbstractImageToolView):
    """BulletinView class.

    Widget used to display the basic information of the current image data.
    """
    _LCD_DIGITS = 12

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # default number is 0
        self._latest_tid = QLCDNumber(self._LCD_DIGITS)
        self._n_total_pulses = QLCDNumber(self._LCD_DIGITS)
        self._n_kept_pulses = QLCDNumber(self._LCD_DIGITS)
        self._dark_train_counter = QLCDNumber(self._LCD_DIGITS)
        self._n_dark_pulses = QLCDNumber(self._LCD_DIGITS)

        self.initUI()

    def initUI(self):
        """Override."""
        self._setLcdStyle(self._latest_tid)
        self._setLcdStyle(self._n_total_pulses)
        self._setLcdStyle(self._n_kept_pulses)
        self._setLcdStyle(self._dark_train_counter)
        self._setLcdStyle(self._n_dark_pulses)

        layout = QGridLayout()
        AR = Qt.AlignRight
        layout.addWidget(QLabel("Latest train ID: "), 0, 0, AR)
        layout.addWidget(self._latest_tid, 0, 1)
        layout.addWidget(QLabel("Total # of pulses/train: "), 1, 0, AR)
        layout.addWidget(self._n_total_pulses, 1, 1)
        layout.addWidget(QLabel("# of kept pulses/train: "), 2, 0, AR)
        layout.addWidget(self._n_kept_pulses, 2, 1)
        layout.addWidget(QLabel("# of dark trains: "), 3, 0, AR)
        layout.addWidget(self._dark_train_counter, 3, 1)
        layout.addWidget(QLabel("# of dark pulses/train: "), 4, 0, AR)
        layout.addWidget(self._n_dark_pulses, 4, 1)
        self.setLayout(layout)

    def initConnections(self):
        """Override."""
        pass

    def _setLcdStyle(self, lcd):
        lcd.setLineWidth(0)
        lcd.setSegmentStyle(QLCDNumber.Flat)
        palette = lcd.palette()
        palette.setColor(palette.WindowText, QColor(85, 85, 255))
        lcd.setPalette(palette)

    def updateF(self, data, auto_update):
        """Override."""
        # always update automatically
        self._latest_tid.display(data.tid)
        self._n_total_pulses.display(data.n_pulses)
        self._n_kept_pulses.display(data.pidx.n_kept(data.n_pulses))
        self._dark_train_counter.display(data.image.dark_count)
        self._n_dark_pulses.display(data.image.n_dark_pulses)
