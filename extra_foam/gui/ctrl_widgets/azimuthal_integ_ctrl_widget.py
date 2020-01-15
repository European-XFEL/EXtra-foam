"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>, Ebad Kamil <ebad.kamil@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from collections import OrderedDict

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QDoubleValidator, QIntValidator
from PyQt5.QtWidgets import QComboBox, QGridLayout, QLabel

from .base_ctrl_widgets import _AbstractCtrlWidget
from .smart_widgets import SmartBoundaryLineEdit, SmartLineEdit
from ...algorithms import compute_q
from ...config import Normalizer, config


_DEFAULT_AZIMUTHAL_INTEG_POINTS = 512


def _estimate_q_range():
    # TODO: Improve!
    max_x = 1500 * config["PIXEL_SIZE"]
    max_q = 1e-10 * compute_q(
        config["SAMPLE_DISTANCE"], max_x, 1000 * config["PHOTON_ENERGY"])
    return f'0, {max_q:.4f}'


class AzimuthalIntegCtrlWidget(_AbstractCtrlWidget):
    """Widget for setting up azimuthal integration parameters."""

    _available_norms = OrderedDict({
        "": Normalizer.UNDEFINED,
        "AUC": Normalizer.AUC,
        "XGM": Normalizer.XGM,
        "ROI": Normalizer.ROI,
    })

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._photon_energy_le = SmartLineEdit(str(config["PHOTON_ENERGY"]))
        self._photon_energy_le.setValidator(QDoubleValidator(0.001, 100, 6))

        self._sample_dist_le = SmartLineEdit(str(config["SAMPLE_DISTANCE"]))
        self._sample_dist_le.setValidator(QDoubleValidator(0.001, 100, 6))

        # coordinate of the point of normal incidence along the detector's
        # second dimension, in pixels, PONI2 in pyFAI
        self._cx_le = SmartLineEdit(str(0))
        self._cx_le.setValidator(QIntValidator())
        # coordinate of the point of normal incidence along the detector's
        # first dimension, in pixels, PONI1 in pyFAI
        self._cy_le = SmartLineEdit(str(0))
        self._cy_le.setValidator(QIntValidator())

        self._px_le = SmartLineEdit(str(config["PIXEL_SIZE"]))
        self._py_le = SmartLineEdit(str(config["PIXEL_SIZE"]))
        self._px_le.setEnabled(False)
        self._py_le.setEnabled(False)

        self._rx_le = SmartLineEdit("0.0")
        self._ry_le = SmartLineEdit("0.0")
        self._rz_le = SmartLineEdit("0.0")
        self._rx_le.setEnabled(False)
        self._ry_le.setEnabled(False)
        self._rz_le.setEnabled(False)

        self._integ_method_cb = QComboBox()
        for method in config["AZIMUTHAL_INTEG_METHODS"]:
            self._integ_method_cb.addItem(method)

        q_range = _estimate_q_range()
        self._integ_range_le = SmartBoundaryLineEdit(q_range)
        self._integ_pts_le = SmartLineEdit(
            str(_DEFAULT_AZIMUTHAL_INTEG_POINTS))
        self._integ_pts_le.setValidator(QIntValidator(1, 8192))

        self._norm_cb = QComboBox()
        for v in self._available_norms:
            self._norm_cb.addItem(v)

        self._auc_range_le = SmartBoundaryLineEdit("0, Inf")
        self._fom_integ_range_le = SmartBoundaryLineEdit("0, Inf")

        self.initUI()
        self.initConnections()

        self.setFixedHeight(self.minimumSizeHint().height())

    def initUI(self):
        """Override."""
        layout = QGridLayout()
        AR = Qt.AlignRight

        row = 0
        layout.addWidget(QLabel("Cx (pixel): "), row, 0, AR)
        layout.addWidget(self._cx_le, row, 1)
        layout.addWidget(QLabel("Cy (pixel): "), row, 2, AR)
        layout.addWidget(self._cy_le, row, 3)
        layout.addWidget(QLabel("Pixel x (m): "), row, 4, AR)
        layout.addWidget(self._px_le, row, 5)
        layout.addWidget(QLabel("Pixel y (m): "), row, 6, AR)
        layout.addWidget(self._py_le, row, 7)

        row += 1
        layout.addWidget(QLabel("Sample distance (m): "), row, 0, AR)
        layout.addWidget(self._sample_dist_le, row, 1)
        layout.addWidget(QLabel("Rotation x (rad): "), row, 2, AR)
        layout.addWidget(self._rx_le, row, 3)
        layout.addWidget(QLabel("Rotation y (rad): "), row, 4, AR)
        layout.addWidget(self._ry_le, row, 5)
        layout.addWidget(QLabel("Rotation z (rad): "), row, 6, AR)
        layout.addWidget(self._rz_le, row, 7)

        row += 1
        layout.addWidget(QLabel("Photon energy (keV): "), row, 0, AR)
        layout.addWidget(self._photon_energy_le, row, 1)
        layout.addWidget(QLabel("Integ method: "), row, 2, AR)
        layout.addWidget(self._integ_method_cb, row, 3)
        layout.addWidget(QLabel("Integ points: "), row, 4, AR)
        layout.addWidget(self._integ_pts_le, row, 5)
        layout.addWidget(QLabel("Integ range (1/A): "), row, 6, AR)
        layout.addWidget(self._integ_range_le, row, 7)

        row += 1
        layout.addWidget(QLabel("Norm: "), row, 0, AR)
        layout.addWidget(self._norm_cb, row, 1)
        layout.addWidget(QLabel("AUC range (1/A): "), row, 2, AR)
        layout.addWidget(self._auc_range_le, row, 3)
        layout.addWidget(QLabel("FOM range (1/A): "), row, 4, AR)
        layout.addWidget(self._fom_integ_range_le, row, 5)

        self.setLayout(layout)

    def initConnections(self):
        """Override."""
        mediator = self._mediator

        self._photon_energy_le.value_changed_sgn.connect(
            lambda x: mediator.onPhotonEnergyChange(float(x)))

        self._sample_dist_le.value_changed_sgn.connect(
            lambda x: mediator.onSampleDistanceChange(float(x)))

        self._px_le.value_changed_sgn.connect(
            lambda x: mediator.onAiPixelSizeXChange(float(x)))
        self._py_le.value_changed_sgn.connect(
            lambda x: mediator.onAiPixelSizeYChange(float(x)))

        self._cx_le.value_changed_sgn.connect(
            lambda x: mediator.onAiIntegCenterXChange(int(x)))
        self._cy_le.value_changed_sgn.connect(
            lambda x: mediator.onAiIntegCenterYChange(int(x)))

        self._integ_method_cb.currentTextChanged.connect(
            mediator.onAiIntegMethodChange)

        self._norm_cb.currentTextChanged.connect(
            lambda x: mediator.onCurveNormalizerChange(
                self._available_norms[x]))

        self._integ_range_le.value_changed_sgn.connect(
            mediator.onAiIntegRangeChange)

        self._integ_pts_le.value_changed_sgn.connect(
            lambda x: mediator.onAiIntegPointsChange(int(x)))

        self._auc_range_le.value_changed_sgn.connect(
            mediator.onAiAucChangeChange)

        self._fom_integ_range_le.value_changed_sgn.connect(
            mediator.onAiFomIntegRangeChange)

    def updateMetaData(self):
        """Override."""
        self._photon_energy_le.returnPressed.emit()
        self._sample_dist_le.returnPressed.emit()

        self._px_le.returnPressed.emit()
        self._py_le.returnPressed.emit()

        self._cx_le.returnPressed.emit()
        self._cy_le.returnPressed.emit()

        self._integ_method_cb.currentTextChanged.emit(
            self._integ_method_cb.currentText())

        self._norm_cb.currentTextChanged.emit(
            self._norm_cb.currentText())

        self._integ_range_le.returnPressed.emit()

        self._integ_pts_le.returnPressed.emit()

        self._auc_range_le.returnPressed.emit()

        self._fom_integ_range_le.returnPressed.emit()

        return True
