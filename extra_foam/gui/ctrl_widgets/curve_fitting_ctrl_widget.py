"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>, Ebad Kamil <ebad.kamil@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from collections import OrderedDict

import numpy as np

from PyQt5.QtWidgets import (
    QCheckBox, QComboBox, QPlainTextEdit, QPushButton, QWidget
)

from .smart_widgets import SmartLineEdit
from ...algorithms import CurveFitting, FittingType


class _BaseFittingCtrlWidget(QWidget):

    _available_types = OrderedDict({
        "": FittingType.UNDEFINED,
        "Linear": FittingType.LINEAR,
        "Cubic": FittingType.CUBIC,
        "Gaussian": FittingType.GAUSSIAN,
        "Lorentzian": FittingType.LORENTZIAN,
        "Error function": FittingType.ERF
    })

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.fit_type_cb = QComboBox()
        for name in self._available_types:
            self.fit_type_cb.addItem(name)

        self._algo = None

        self.fit_btn = QPushButton("Fit")
        self.clear_btn = QPushButton("Clear")
        self.corr1_cb = QCheckBox("Correlation 1")
        self.corr1_cb.setChecked(True)
        self.corr2_cb = QCheckBox("Correlation 2")

        self._params = []
        for i in range(6):
            widget = SmartLineEdit("0")
            widget.setMinimumWidth(120)
            self._params.append(widget)

        self._output = QPlainTextEdit()
        self._output.setOverwriteMode(True)
        self._output.setReadOnly(True)
        self._output.setMaximumBlockCount(10)

    def initConnections(self):
        self.fit_type_cb.currentTextChanged.connect(self._onFitTypeChanged)
        self.fit_type_cb.currentTextChanged.emit(
            self.fit_type_cb.currentText())

        self.corr1_cb.toggled.connect(
            lambda x: self.corr2_cb.setChecked(not x))
        self.corr2_cb.toggled.connect(
            lambda x: self.corr1_cb.setChecked(not x))

    def fit(self, x, y):
        if self._algo is None:
            return None, None

        if len(x) <= 1:
            self._output.setPlainText(f"Not enough data")
            return None, None

        try:
            popt = self._algo.fit(x, y, p0=self._p0())
            self._output.setPlainText(f"Optimized parameters: {popt}")

            # TODO: do we need 'num' to be configurable?
            new_x = np.linspace(np.min(x), np.max(x), num=100)
            new_y = self._algo(new_x, *popt)
            self._output.setPlainText(self._algo.format(*popt))

            return new_x, new_y

        except Exception as e:
            self._output.setPlainText(repr(e))
            return None, None

    def _onFitTypeChanged(self, name):
        self._algo = CurveFitting.create(self._available_types[name])
        self._output.clear()
        n = 0 if self._algo is None else self._algo.n
        for i, item in enumerate(self._params):
            if i < n:
                item.setText('1')
                item.setEnabled(True)
            else:
                item.setText('0')
                item.setEnabled(False)

    def _p0(self):
        n = 0 if self._algo is None else self._algo.n
        p0 = []
        for i, item in enumerate(self._params):
            if i < n:
                p0.append(float(item.text()))

        return p0
