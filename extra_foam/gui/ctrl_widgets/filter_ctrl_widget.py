"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from collections import OrderedDict

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QCheckBox, QComboBox, QGridLayout, QLabel

from .base_ctrl_widgets import _AbstractGroupBoxCtrlWidget
from .smart_widgets import SmartBoundaryLineEdit
from ..gui_helpers import invert_dict
from ...config import AnalysisType
from ...database import Metadata as mt


class FomFilterCtrlWidget(_AbstractGroupBoxCtrlWidget):
    """Widget for setting up pulse-resolved filter parameters."""

    _analysis_types = OrderedDict({
        "": AnalysisType.UNDEFINED,
        "ROI FOM": AnalysisType.ROI_FOM,
    })
    _analysis_types_inv = invert_dict(_analysis_types)

    _N_FILTERS = 2
    _N_TRAIN_FILTERS = 1

    def __init__(self, *args, **kwargs):
        super().__init__("FOM filter setup", *args, **kwargs)

        self._analysis_type_cbs = []
        self._fom_range_les = []
        for _ in range(self._N_FILTERS):
            cb = QComboBox()
            cb.addItems(self._analysis_types.keys())
            self._analysis_type_cbs.append(cb)
            self._fom_range_les.append(SmartBoundaryLineEdit("-Inf, Inf"))

        if not self._pulse_resolved:
            pass

        self.initUI()
        self.initConnections()

        self.setFixedHeight(self.minimumSizeHint().height())

    def initUI(self):
        """Overload."""
        layout = QGridLayout()
        AR = Qt.AlignRight

        for i, (cb, le) in enumerate(zip(self._analysis_type_cbs,
                                         self._fom_range_les)):
            if i < self._N_TRAIN_FILTERS:
                layout.addWidget(QLabel(f"By train  - "), i, 0, AR)
            else:
                layout.addWidget(QLabel(f"By pulse  - "), i, 0, AR)
            layout.addWidget(QLabel("Analysis type: "), i, 1, AR)
            layout.addWidget(cb, i, 2)
            layout.addWidget(QLabel("Fom range: "), i, 3, AR)
            layout.addWidget(le, i, 4)

        self.setLayout(layout)

    def initConnections(self):
        """Overload."""
        mediator = self._mediator

        self._analysis_type_cb.currentTextChanged.connect(
            lambda x: mediator.onFomFilterAnalysisTypeChange(
                self._analysis_types[x]))
        self._fom_range_le.value_changed_sgn.connect(
            mediator.onFomFilterRangeChange)

    def updateMetaData(self):
        """Overload."""
        # self._analysis_type_cb.currentTextChanged.emit(
        #     self._analysis_type_cb.currentText())
        # self._fom_range_le.returnPressed.emit()
        # self._pulse_resolved_cb.toggled.emit(
        #     self._pulse_resolved_cb.isChecked())
        # return True
        pass

    def loadMetaData(self):
        """Override."""
        # cfg = self._meta.hget_all(mt.FOM_FILTER_PROC)
        # self._analysis_type_cb.setCurrentText(
        #     self._analysis_types_inv[int(cfg["analysis_type"])])
        # self._fom_range_le.setText(cfg["fom_range"][1:-1])
        # if self._pulse_resolved:
        #     self._pulse_resolved_cb.setChecked(cfg["pulse_resolved"] == 'True')
        pass