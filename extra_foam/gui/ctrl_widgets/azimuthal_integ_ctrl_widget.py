"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>, Ebad Kamil <ebad.kamil@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from collections import OrderedDict

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QDoubleValidator, QIntValidator
from PyQt5.QtWidgets import (
    QCheckBox, QComboBox, QFrame, QGridLayout, QHBoxLayout, QLabel
)

from .base_ctrl_widgets import _AbstractCtrlWidget
from .curve_fitting_ctrl_widget import _BaseFittingCtrlWidget
from .smart_widgets import (
    SmartBoundaryLineEdit, SmartSliceLineEdit, SmartLineEdit
)
from ..gui_helpers import invert_dict
from ...algorithms import compute_q
from ...config import config, list_azimuthal_integ_methods, Normalizer
from ...database import Metadata as mt

_DEFAULT_AZIMUTHAL_INTEG_POINTS = 512
_DEFAULT_PEAK_PROMINENCE = 100


def _estimate_q_range():
    # TODO: Improve!
    max_x = 1500 * config["PIXEL_SIZE"]
    max_q = 1e-10 * compute_q(
        config["SAMPLE_DISTANCE"], max_x, 1000 * config["PHOTON_ENERGY"])
    return f'0, {max_q:.4f}'


class _AzimuthalIntegFittingCtrlWidget(_BaseFittingCtrlWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.initUI()
        self.initConnections()

    def initUI(self):
        layout = QGridLayout()

        layout.addWidget(QLabel("Fit type: "), 0, 0)
        layout.addWidget(self.fit_type_cb, 0, 1)

        layout.addWidget(self.fit_btn, 0, 2, 1, 1)
        layout.addWidget(self.clear_btn, 0, 3, 1, 1)

        layout.addWidget(QLabel("Param a0 = "), 1, 0)
        layout.addWidget(self._params[0], 1, 1)
        layout.addWidget(QLabel("Param b0 = "), 1, 2)
        layout.addWidget(self._params[1], 1, 3)
        layout.addWidget(QLabel("Param c0 = "), 2, 0)
        layout.addWidget(self._params[2], 2, 1)
        layout.addWidget(QLabel("Param d0 = "), 2, 2)
        layout.addWidget(self._params[3], 2, 3)
        layout.addWidget(self._output, 3, 0, 1, 4)

        self.setLayout(layout)


class AzimuthalIntegCtrlWidget(_AbstractCtrlWidget):
    """Widget for setting up azimuthal integration parameters."""

    _available_norms = OrderedDict({
        "": Normalizer.UNDEFINED,
        "AUC": Normalizer.AUC,
        "XGM": Normalizer.XGM,
        "DIGITIZER": Normalizer.DIGITIZER,
        "ROI": Normalizer.ROI,
    })
    _available_norms_inv = invert_dict(_available_norms)

    cx_changed_sgn = pyqtSignal(float)
    cy_changed_sgn = pyqtSignal(float)
    fit_curve_sgn = pyqtSignal()
    clear_fitting_sgn = pyqtSignal()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._photon_energy_le = SmartLineEdit(str(config["PHOTON_ENERGY"]))
        self._photon_energy_le.setValidator(QDoubleValidator(0.001, 100, 6))

        self._sample_dist_le = SmartLineEdit(str(config["SAMPLE_DISTANCE"]))
        self._sample_dist_le.setValidator(QDoubleValidator(0.001, 100, 6))

        # coordinate of the point of normal incidence along the detector's
        # second dimension, in pixels, PONI2 in pyFAI
        self._cx_le = SmartLineEdit(str(0))
        self._cx_le.setValidator(QDoubleValidator())
        # coordinate of the point of normal incidence along the detector's
        # first dimension, in pixels, PONI1 in pyFAI
        self._cy_le = SmartLineEdit(str(0))
        self._cy_le.setValidator(QDoubleValidator())

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
        for method in list_azimuthal_integ_methods(config["DETECTOR"]):
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

        self._peak_finding_cb = QCheckBox("Peak finding")
        self._peak_finding_cb.setChecked(True)
        self._peak_prominence_le = SmartLineEdit(str(_DEFAULT_PEAK_PROMINENCE))
        self._peak_prominence_le.setValidator(QDoubleValidator())
        self._peak_slicer_le = SmartSliceLineEdit(":")

        self._reference_cb = QCheckBox("Use Reference")
        self._reference_cb.setChecked(False)

        self._non_reconfigurable_widgets = [
        ]

        self.initUI()
        self.initConnections()

        self.setFixedHeight(self.minimumSizeHint().height())

    def initUI(self):
        """Override."""
        layout = QHBoxLayout()
        AR = Qt.AlignRight

        param_widget = QFrame()
        param_layout = QGridLayout()
        row = 0
        param_layout.addWidget(QLabel("Cx (pixel): "), row, 0, AR)
        param_layout.addWidget(self._cx_le, row, 1)
        param_layout.addWidget(QLabel("Cy (pixel): "), row, 2, AR)
        param_layout.addWidget(self._cy_le, row, 3)
        param_layout.addWidget(QLabel("Pixel x (m): "), row, 4, AR)
        param_layout.addWidget(self._px_le, row, 5)
        param_layout.addWidget(QLabel("Pixel y (m): "), row, 6, AR)
        param_layout.addWidget(self._py_le, row, 7)

        row += 1
        param_layout.addWidget(QLabel("Sample distance (m): "), row, 0, AR)
        param_layout.addWidget(self._sample_dist_le, row, 1)
        param_layout.addWidget(QLabel("Rotation x (rad): "), row, 2, AR)
        param_layout.addWidget(self._rx_le, row, 3)
        param_layout.addWidget(QLabel("Rotation y (rad): "), row, 4, AR)
        param_layout.addWidget(self._ry_le, row, 5)
        param_layout.addWidget(QLabel("Rotation z (rad): "), row, 6, AR)
        param_layout.addWidget(self._rz_le, row, 7)

        row += 1
        param_layout.addWidget(QLabel("Photon energy (keV): "), row, 0, AR)
        param_layout.addWidget(self._photon_energy_le, row, 1)
        param_layout.addWidget(QLabel("Integ method: "), row, 2, AR)
        param_layout.addWidget(self._integ_method_cb, row, 3)
        param_layout.addWidget(QLabel("Integ points: "), row, 4, AR)
        param_layout.addWidget(self._integ_pts_le, row, 5)
        param_layout.addWidget(QLabel("Integ range (1/A): "), row, 6, AR)
        param_layout.addWidget(self._integ_range_le, row, 7)

        row += 1
        param_layout.addWidget(QLabel("Norm: "), row, 0, AR)
        param_layout.addWidget(self._norm_cb, row, 1)
        param_layout.addWidget(QLabel("AUC range (1/A): "), row, 2, AR)
        param_layout.addWidget(self._auc_range_le, row, 3)
        param_layout.addWidget(QLabel("FOM range (1/A): "), row, 4, AR)
        param_layout.addWidget(self._fom_integ_range_le, row, 5)
        param_layout.addWidget(self._reference_cb, row, 6, AR)

        param_widget.setLayout(param_layout)

        algo_widget = QFrame()
        algo_layout = QGridLayout()
        algo_layout.addWidget(self._peak_finding_cb, 0, 0, 1, 2)
        algo_layout.addWidget(QLabel("Peak prominence: "), 1, 0, AR)
        algo_layout.addWidget(self._peak_prominence_le, 1, 1)
        algo_layout.addWidget(QLabel("Peak slicer: "), 2, 0, AR)
        algo_layout.addWidget(self._peak_slicer_le, 2, 1)
        algo_widget.setLayout(algo_layout)

        fitting_widget = QFrame()
        fitting_layout = QHBoxLayout()
        self._fitter = _AzimuthalIntegFittingCtrlWidget()
        fitting_layout.addWidget(self._fitter)
        fitting_widget.setLayout(fitting_layout)

        layout.addWidget(param_widget)
        layout.addWidget(algo_widget)
        layout.addWidget(fitting_widget)
        layout.setContentsMargins(1, 1, 1, 1)
        self.setLayout(layout)

        self.setFrameStyle(QFrame.NoFrame)
        param_widget.setFrameStyle(QFrame.StyledPanel)
        algo_widget.setFrameStyle(QFrame.StyledPanel)
        fitting_widget.setFrameStyle(QFrame.StyledPanel)

    def initConnections(self):
        """Override."""
        mediator = self._mediator

        self._fitter.fit_btn.clicked.connect(self.fit_curve_sgn)
        self._fitter.clear_btn.clicked.connect(self.clear_fitting_sgn)

        self._reference_cb.toggled.connect(mediator.onUseReferenceChange)

        self._photon_energy_le.value_changed_sgn.connect(
            lambda x: mediator.onPhotonEnergyChange(float(x)))

        self._sample_dist_le.value_changed_sgn.connect(
            lambda x: mediator.onSampleDistanceChange(float(x)))

        self._px_le.value_changed_sgn.connect(
            lambda x: mediator.onAiPixelSizeXChange(float(x)))
        self._py_le.value_changed_sgn.connect(
            lambda x: mediator.onAiPixelSizeYChange(float(x)))

        self._cx_le.value_changed_sgn.connect(
            lambda x: mediator.onAiIntegCenterXChange(float(x)))
        self._cy_le.value_changed_sgn.connect(
            lambda x: mediator.onAiIntegCenterYChange(float(x)))
        self._cx_le.value_changed_sgn.connect(lambda x: self.cx_changed_sgn.emit(float(x)))
        self._cy_le.value_changed_sgn.connect(lambda y: self.cy_changed_sgn.emit(float(y)))

        self._integ_method_cb.currentTextChanged.connect(
            mediator.onAiIntegMethodChange)

        self._norm_cb.currentTextChanged.connect(
            lambda x: mediator.onAiNormChange(self._available_norms[x]))

        self._integ_range_le.value_changed_sgn.connect(
            mediator.onAiIntegRangeChange)

        self._integ_pts_le.value_changed_sgn.connect(
            lambda x: mediator.onAiIntegPointsChange(int(x)))

        self._auc_range_le.value_changed_sgn.connect(
            mediator.onAiAucRangeChange)

        self._fom_integ_range_le.value_changed_sgn.connect(
            mediator.onAiFomIntegRangeChange)

        self._peak_finding_cb.toggled.connect(
            mediator.onAiPeakFindingChange)
        
        self._peak_prominence_le.value_changed_sgn.connect(
            mediator.onAiPeakProminenceChange)

        self._peak_slicer_le.value_changed_sgn.connect(
            mediator.onAiPeakSlicerChange)

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

        self._peak_finding_cb.toggled.emit(self._peak_finding_cb.isChecked())
        self._peak_prominence_le.returnPressed.emit()
        self._peak_slicer_le.returnPressed.emit()
        self._reference_cb.toggled.emit(self._reference_cb.isChecked())

        return True

    def loadMetaData(self):
        """Override."""
        cfg = self._meta.hget_all(mt.GLOBAL_PROC)
        rfg = self._meta.hget_all(mt.REFERENCE_IMAGE_PROC)
        self._photon_energy_le.setText(cfg["photon_energy"])
        self._sample_dist_le.setText(cfg["sample_distance"])

        cfg = self._meta.hget_all(mt.AZIMUTHAL_INTEG_PROC)
        self._px_le.setText(cfg['pixel_size_x'])
        self._py_le.setText(cfg['pixel_size_y'])
        self._cx_le.setText(cfg['integ_center_x'])
        self._cy_le.setText(cfg['integ_center_y'])
        self._integ_method_cb.setCurrentText(cfg['integ_method'])
        self._integ_range_le.setText(cfg['integ_range'][1:-1])
        self._integ_pts_le.setText(cfg['integ_points'])
        self._norm_cb.setCurrentText(
            self._available_norms_inv[int(cfg['normalizer'])])
        self._auc_range_le.setText(cfg['auc_range'][1:-1])
        self._fom_integ_range_le.setText(cfg['fom_integ_range'][1:-1])

        self._updateWidgetValue(self._peak_finding_cb, cfg, "peak_finding")
        self._updateWidgetValue(
            self._peak_prominence_le, cfg, "peak_prominence")
        self._updateWidgetValue(self._peak_slicer_le, cfg, "peak_slicer")
        self._updateWidgetValue(self._reference_cb, rfg, "reference_used")

    def fitCurve(self, x, y):
        return self._fitter.fit(x, y)
