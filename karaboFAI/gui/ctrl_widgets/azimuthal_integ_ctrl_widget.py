"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

AzimuthalIntegCtrlWidget.

Author: Jun Zhu <jun.zhu@xfel.eu>, Ebad Kamil <ebad.kamil@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from collections import OrderedDict

from ..pyqtgraph import QtCore, QtGui

from .base_ctrl_widgets import AbstractCtrlWidget
from ..misc_widgets import SmartBoundaryLineEdit, SmartLineEdit
from ...config import AiNormalizer, config


class AzimuthalIntegCtrlWidget(AbstractCtrlWidget):
    """Widget for setting up the azimuthal integration parameters."""

    _available_normalizers = OrderedDict({
        "AUC": AiNormalizer.AUC,
        "ROI1 - ROI2": AiNormalizer.ROI_SUB,
        "ROI1": AiNormalizer.ROI1,
        "ROI2": AiNormalizer.ROI2,
        "ROI1 + ROI2": AiNormalizer.ROI_SUM,
    })

    def __init__(self, *args, **kwargs):
        super().__init__("Azimuthal integration setup", *args, **kwargs)

        # default state is unchecked
        self._pulsed_integ_cb = QtGui.QCheckBox("Pulsed azimuthal integ")

        self._cx_le = SmartLineEdit(str(config["CENTER_X"]))
        self._cx_le.setValidator(QtGui.QIntValidator())
        self._cy_le = SmartLineEdit(str(config["CENTER_Y"]))
        self._cy_le.setValidator(QtGui.QIntValidator())
        self._itgt_method_cb = QtGui.QComboBox()
        for method in config["AZIMUTHAL_INTEG_METHODS"]:
            self._itgt_method_cb.addItem(method)
        self._integ_range_le = SmartBoundaryLineEdit(
            ', '.join([str(v) for v in config["AZIMUTHAL_INTEG_RANGE"]]))
        self._integ_pts_le = SmartLineEdit(
            str(config["AZIMUTHAL_INTEG_POINTS"]))
        self._integ_pts_le.setValidator(QtGui.QIntValidator(1, 8192))

        self._normalizers_cb = QtGui.QComboBox()
        for v in self._available_normalizers:
            self._normalizers_cb.addItem(v)

        self._auc_range_le = SmartBoundaryLineEdit(
            ', '.join([str(v) for v in config["AZIMUTHAL_INTEG_RANGE"]]))

        self._fom_integ_range_le = SmartBoundaryLineEdit(
            ', '.join([str(v) for v in config["AZIMUTHAL_INTEG_RANGE"]]))

        self._non_reconfigurable_widgets = [
            self._pulsed_integ_cb,
        ]

        self.initUI()
        self.initConnections()

        self.setFixedHeight(self.minimumSizeHint().height())

    def initUI(self):
        """Override."""
        layout = QtGui.QGridLayout()
        AR = QtCore.Qt.AlignRight

        layout.addWidget(QtGui.QLabel("Cx (pixel): "), 0, 0, AR)
        layout.addWidget(self._cx_le, 0, 1)
        layout.addWidget(QtGui.QLabel("Cy (pixel): "), 0, 2, AR)
        layout.addWidget(self._cy_le, 0, 3)
        layout.addWidget(QtGui.QLabel("Integ method: "), 1, 0, AR)
        layout.addWidget(self._itgt_method_cb, 1, 1)
        layout.addWidget(QtGui.QLabel("Integ points: "), 1, 2, AR)
        layout.addWidget(self._integ_pts_le, 1, 3)
        layout.addWidget(QtGui.QLabel("Integ range (1/A): "), 2, 0, AR)
        layout.addWidget(self._integ_range_le, 2, 1)
        layout.addWidget(QtGui.QLabel("Normalized by: "), 2, 2, AR)
        layout.addWidget(self._normalizers_cb, 2, 3)
        layout.addWidget(QtGui.QLabel("AUC range: "), 3, 0, AR)
        layout.addWidget(self._auc_range_le, 3, 1)
        layout.addWidget(QtGui.QLabel("FOM range: "), 3, 2, AR)
        layout.addWidget(self._fom_integ_range_le, 3, 3)
        layout.addWidget(self._pulsed_integ_cb, 4, 0, 1, 4, AR)

        self.setLayout(layout)

    def initConnections(self):
        mediator = self._mediator

        self._cx_le.returnPressed.connect(
            lambda: mediator.onAiIntegCenterXChange(
                int(self._cx_le.text())))

        self._cy_le.returnPressed.connect(
            lambda: mediator.onAiIntegCenterYChange(
                int(self._cy_le.text())))

        self._itgt_method_cb.currentTextChanged.connect(
            mediator.onAiIntegMethodChange)

        self._normalizers_cb.currentTextChanged.connect(
            lambda x: mediator.onAiNormalizerChange(
                self._available_normalizers[x]))

        self._pulsed_integ_cb.toggled.connect(
            mediator.onAiPulsedIntegStateChange)

        self._integ_range_le.value_changed_sgn.connect(
            mediator.onAiIntegRangeChange)

        self._integ_pts_le.returnPressed.connect(
            lambda: mediator.onAiIntegPointsChange(
                int(self._integ_pts_le.text())))

        self._auc_range_le.value_changed_sgn.connect(
            mediator.onAiAucChangeChange)

        self._fom_integ_range_le.value_changed_sgn.connect(
            mediator.onAiFomIntegRangeChange)

    def updateSharedParameters(self):
        self._cx_le.returnPressed.emit()

        self._cy_le.returnPressed.emit()

        self._itgt_method_cb.currentTextChanged.emit(
            self._itgt_method_cb.currentText())

        self._normalizers_cb.currentTextChanged.emit(
            self._normalizers_cb.currentText())

        self._pulsed_integ_cb.toggled.emit(self._pulsed_integ_cb.isChecked())

        self._integ_range_le.returnPressed.emit()

        self._integ_pts_le.returnPressed.emit()

        self._auc_range_le.returnPressed.emit()

        self._fom_integ_range_le.returnPressed.emit()

        return True
