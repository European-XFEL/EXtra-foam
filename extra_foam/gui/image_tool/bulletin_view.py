"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import QGridLayout, QLabel, QLCDNumber, QPushButton

from .base_view import _AbstractImageToolView
from ...database import MonProxy
from ...config import config


class BulletinView(_AbstractImageToolView):
    """BulletinView class.

    Widget used to display the basic information of the current image data.
    """
    _LCD_DIGITS = 12

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # default number is 0
        self._displayed_tid = QLCDNumber(self._LCD_DIGITS)
        self._displayed_tid.display(None)
        self._n_total_pulses = QLCDNumber(self._LCD_DIGITS)
        self._n_kept_pulses = QLCDNumber(self._LCD_DIGITS)
        self._dark_train_counter = QLCDNumber(self._LCD_DIGITS)
        self._n_dark_pulses = QLCDNumber(self._LCD_DIGITS)
        self._last_processed_tid = QLCDNumber(self._LCD_DIGITS)
        self._n_processed_trains = QLCDNumber(self._LCD_DIGITS)
        self._n_dropped_trains = QLCDNumber(self._LCD_DIGITS)
        # Note: it is difficult to count number of dropped pulses since
        #       the processing code takes a shortcut if a train is dropped.
        #       Also, when the pipeline decides to drop a train, it may still
        #       not know how many pulses are there in a train. The number of
        #       pulses in a train is not only decided by the data received,
        #       but also depends on the pulse slicer.
        self._n_processed_pulses = QLCDNumber(self._LCD_DIGITS)

        self._reset_process_count_btn = QPushButton("Reset process count")

        self._mon = MonProxy()
        self._resetProcessCount()  # initialization

        self._timer = QTimer()
        self._timer.timeout.connect(self._updateProcessCount)
        self._timer.start(config["GUI_PLOT_WITH_STATE_UPDATE_TIMER"])

        self.initUI()
        self.initConnections()

    def initUI(self):
        """Override."""
        self._setLcdStyle(self._displayed_tid)
        self._setLcdStyle(self._n_total_pulses)
        self._setLcdStyle(self._n_kept_pulses)
        self._setLcdStyle(self._dark_train_counter)
        self._setLcdStyle(self._n_dark_pulses)
        self._setLcdStyle(self._last_processed_tid)
        self._setLcdStyle(self._n_processed_trains)
        self._setLcdStyle(self._n_dropped_trains)
        self._setLcdStyle(self._n_processed_pulses)

        layout = QGridLayout()
        AR = Qt.AlignRight
        layout.addWidget(QLabel("Displayed train ID: "), 0, 0, AR)
        layout.addWidget(self._displayed_tid, 0, 1)
        layout.addWidget(QLabel("Total # of pulses/train: "), 1, 0, AR)
        layout.addWidget(self._n_total_pulses, 1, 1)
        layout.addWidget(QLabel("# of kept pulses/train: "), 2, 0, AR)
        layout.addWidget(self._n_kept_pulses, 2, 1)
        layout.addWidget(QLabel("# of dark trains: "), 3, 0, AR)
        layout.addWidget(self._dark_train_counter, 3, 1)
        layout.addWidget(QLabel("# of dark pulses/train: "), 4, 0, AR)
        layout.addWidget(self._n_dark_pulses, 4, 1)
        layout.addWidget(QLabel("Last processed train ID: "), 5, 0, AR)
        layout.addWidget(self._last_processed_tid, 5, 1)
        layout.addWidget(QLabel("# of processed trains: "), 6, 0, AR)
        layout.addWidget(self._n_processed_trains, 6, 1)
        layout.addWidget(QLabel("# of dropped trains: "), 7, 0, AR)
        layout.addWidget(self._n_dropped_trains, 7, 1)
        layout.addWidget(QLabel("# of processed pulses: "), 8, 0, AR)
        layout.addWidget(self._n_processed_pulses, 8, 1)
        layout.addWidget(self._reset_process_count_btn, 9, 1)
        self.setLayout(layout)

    def initConnections(self):
        """Override."""
        self._reset_process_count_btn.clicked.connect(self._resetProcessCount)

    def _setLcdStyle(self, lcd):
        lcd.setLineWidth(0)
        lcd.setSegmentStyle(QLCDNumber.Flat)
        palette = lcd.palette()
        palette.setColor(palette.WindowText, QColor(85, 85, 255))
        lcd.setPalette(palette)

    def updateF(self, data, auto_update):
        """Override."""
        # Always update automatically. Note that we convert the trainId to a
        # float because otherwise it gets cast to an int by PyQt and may
        # overflow. This way we hit the double overload of QLCDNumber.display().
        self._displayed_tid.display(data.tid if data.tid is None else float(data.tid))
        self._n_total_pulses.display(data.n_pulses)
        self._n_kept_pulses.display(data.pidx.n_kept(data.n_pulses))
        self._dark_train_counter.display(data.image.dark_count)
        self._n_dark_pulses.display(data.image.n_dark_pulses)

    def _updateProcessCount(self):
        tid, n_processed, n_dropped, n_processed_pulses = \
            self._mon.get_process_count()
        self._last_processed_tid.display(tid if tid is None else float(tid))
        self._n_processed_trains.display(n_processed)
        self._n_dropped_trains.display(n_dropped)
        self._n_processed_pulses.display(n_processed_pulses)

    def _resetProcessCount(self):
        self._mon.reset_process_count()
