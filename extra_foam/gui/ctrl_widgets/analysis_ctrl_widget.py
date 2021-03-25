"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>, Ebad Kamil <ebad.kamil@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIntValidator
from PyQt5.QtWidgets import QGridLayout, QLabel, QPushButton, QWidget

from .base_ctrl_widgets import _AbstractGroupBoxCtrlWidget
from .smart_widgets import SmartLineEdit
from ...config import config, session
from ...database import Metadata as mt


class AnalysisCtrlWidget(_AbstractGroupBoxCtrlWidget):
    """Widget for setting up general analysis parameters."""

    def __init__(self, *args, **kwargs):
        super().__init__("Global setup", *args, **kwargs)

        index_validator = QIntValidator(
            0, config["MAX_N_PULSES_PER_TRAIN"] - 1)
        self._poi_index_les = [SmartLineEdit(str(0)), SmartLineEdit(str(0))]
        for w in self._poi_index_les:
            w.setValidator(index_validator)
            if not self._pulse_resolved:
                w.setEnabled(False)

        self._ma_window_le = SmartLineEdit("1")
        validator = QIntValidator()
        validator.setBottom(1)
        self._ma_window_le.setValidator(validator)

        self._reset_all_btn = QPushButton("Reset all")
        self._reset_ma_btn = QPushButton("Reset moving average")
        self._reset_pp_btn = QPushButton("Reset pump-probe")
        self._reset_correlation_btn = QPushButton("Reset correlation")
        self._reset_binning_btn = QPushButton("Reset binning")
        self._reset_histogram_btn = QPushButton("Reset histogram")

        self._restore_session_btn = QPushButton("Restore last session")
        self._restore_session_btn.setToolTip("Note: this currently only restores ROIs")
        self._restore_session_btn.setEnabled(session.can_restore())

        self.initUI()
        self.initConnections()

        self.setFixedHeight(self.minimumSizeHint().height())

    def initUI(self):
        """Overload."""
        layout = QGridLayout()
        AR = Qt.AlignRight

        row = 0
        layout.addWidget(QLabel("POI indices: "), row, 0, 1, 1, AR)
        layout.addWidget(self._poi_index_les[0], row, 1, 1, 1)
        layout.addWidget(self._poi_index_les[1], row, 2, 1, 1)

        row += 1
        layout.addWidget(QLabel("Moving average window: "), row, 0, AR)
        layout.addWidget(self._ma_window_le, row, 1)

        row += 1
        layout.addWidget(self._reset_all_btn, row, 0,)
        layout.addWidget(self._reset_ma_btn, row, 1)
        layout.addWidget(self._reset_pp_btn, row + 1, 1)
        layout.addWidget(self._reset_correlation_btn, row, 2)
        layout.addWidget(self._reset_histogram_btn, row + 1, 2)
        layout.addWidget(self._reset_binning_btn, row + 2, 2)

        row += 3
        layout.addWidget(self._restore_session_btn, row, 0)

        self.setLayout(layout)

    def initConnections(self):
        """Overload."""
        mediator = self._mediator

        # this cannot be done using a 'for' loop
        self._poi_index_les[0].value_changed_sgn.connect(
            lambda x: mediator.onPoiIndexChange(0, int(x)))
        self._poi_index_les[1].value_changed_sgn.connect(
            lambda x: mediator.onPoiIndexChange(1, int(x)))
        mediator.poi_window_initialized_sgn.connect(self.updateMetaData)

        self._ma_window_le.value_changed_sgn.connect(
            lambda x: mediator.onMaWindowChange(int(x)))

        self._reset_all_btn.clicked.connect(mediator.onResetAll)

        self._reset_ma_btn.clicked.connect(mediator.onResetMa)
        self._reset_pp_btn.clicked.connect(mediator.onPpReset)

        self._reset_correlation_btn.clicked.connect(mediator.onCorrelationReset)
        self._reset_binning_btn.clicked.connect(mediator.onBinReset)
        self._reset_histogram_btn.clicked.connect(mediator.onHistReset)

        self._restore_session_btn.clicked.connect(session.trigger_restore)
        self._restore_session_btn.clicked.connect(lambda: self._restore_session_btn.setEnabled(False))

    def updateMetaData(self):
        """Override"""
        for w in self._poi_index_les:
            w.returnPressed.emit()
        self._ma_window_le.returnPressed.emit()
        return True

    def loadMetaData(self):
        """Override."""
        cfg = self._meta.hget_all(mt.GLOBAL_PROC)
        self._ma_window_le.setText(cfg["ma_window"])
        if self._pulse_resolved:
            self._poi_index_les[0].setText(cfg["poi1_index"])
            self._poi_index_les[1].setText(cfg["poi2_index"])
