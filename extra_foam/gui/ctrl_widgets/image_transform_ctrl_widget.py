"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIntValidator
from PyQt5.QtWidgets import (
    QCheckBox, QDoubleSpinBox, QFrame, QGridLayout, QHBoxLayout, QLabel,
    QLineEdit, QPushButton, QSpinBox, QTabWidget, QWidget
)

from .smart_widgets import SmartLineEdit, SmartBoundaryLineEdit
from .base_ctrl_widgets import _AbstractCtrlWidget
from ...config import ImageTransformType
from ...database import Metadata as mt

from extra_foam.algorithms import (
    edge_detect, fourier_transform_2d
)


class _FourierTransformCtrlWidget(QWidget):
    """Fourier transform control widget."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.logrithmic_cb = QCheckBox("Logrithmic scale")
        self.logrithmic_cb.setChecked(True)

        self.initUI()

    def initUI(self):
        layout = QGridLayout()
        layout.addWidget(self.logrithmic_cb)
        self.setLayout(layout)


class _EdgeDetectionCtrlWidget(QWidget):
    """Edge detection control widget."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.kernel_size_sp = QSpinBox()
        self.kernel_size_sp.setRange(1, 11)
        self.kernel_size_sp.setSingleStep(2)
        self.kernel_size_sp.setValue(5)

        self.sigma_sp = QDoubleSpinBox()
        # double the default min and max values
        self.sigma_sp.setRange(0.4, 4)
        self.sigma_sp.setSingleStep(0.1)
        self.sigma_sp.setValue(1.1)

        self.threshold_le = SmartBoundaryLineEdit("50, 100")

        self.initUI()

    def initUI(self):
        AR = Qt.AlignRight

        layout = QGridLayout()
        layout.addWidget(QLabel("Kernel size: "), 0, 0, AR)
        layout.addWidget(self.kernel_size_sp, 0, 1)
        layout.addWidget(QLabel("Sigma: "), 0, 2, AR)
        layout.addWidget(self.sigma_sp, 0, 3)
        layout.addWidget(QLabel("Threshold: "), 0, 4, AR)
        layout.addWidget(self.threshold_le, 0, 5)

        self.setLayout(layout)


class ImageTransformCtrlWidget(_AbstractCtrlWidget):
    """Control widget for image transform in the ImageTool."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._ma_window_le = SmartLineEdit("1")
        validator = QIntValidator()
        validator.setBottom(1)
        self._ma_window_le.setValidator(validator)

        self._fourier_transform = _FourierTransformCtrlWidget()
        self._edge_detection = _EdgeDetectionCtrlWidget()

        self._opt_index_map = dict()
        self._opt_tab = QTabWidget()
        self._opt_tab.addTab(self._fourier_transform, "Fourier transform")
        idx = 0
        self._opt_index_map[idx] = ImageTransformType.FOURIER_TRANSFORM
        self._opt_tab.addTab(self._edge_detection, "Edge detection")
        idx += 1
        self._opt_index_map[idx] = ImageTransformType.EDGE_DETECTION

        self._non_reconfigurable_widgets = [
        ]

        self.initUI()
        self.initConnections()

    def initUI(self):
        """Override."""
        common = QFrame()
        common_layout = QGridLayout()
        common_layout.addWidget(QLabel("Moving average window: "), 0, 0)
        common_layout.addWidget(self._ma_window_le, 0, 1)
        common.setLayout(common_layout)

        layout = QHBoxLayout()
        layout.addWidget(common)
        layout.addWidget(self._opt_tab)
        self.setLayout(layout)

        self.setFixedHeight(self.minimumSizeHint().height())

    def initConnections(self):
        """Override."""
        mediator = self._mediator

        self._ma_window_le.value_changed_sgn.connect(
            mediator.onItMaWindowChange)

        self._opt_tab.currentChanged.connect(
            lambda x: mediator.onItTransformTypeChange(
                self._opt_index_map[x]))
        # initialize here since updateMetaData does not initialize it
        self._mediator.onItTransformTypeChange(
            ImageTransformType.UNDEFINED)

        fft = self._fourier_transform
        fft.logrithmic_cb.toggled.connect(mediator.onItFftLogrithmicScaleChange)

        ed = self._edge_detection
        ed.kernel_size_sp.valueChanged.connect(
            mediator.onItEdKernelSizeChange)
        ed.sigma_sp.valueChanged.connect(mediator.onItEdSigmaChange)
        ed.threshold_le.value_changed_sgn.connect(
            mediator.onItEdThresholdChange)

    def updateMetaData(self):
        """Override."""
        self._ma_window_le.returnPressed.emit()

        # do not update transform type since it is not an "input"

        fft = self._fourier_transform
        fft.logrithmic_cb.toggled.emit(fft.logrithmic_cb.isChecked())

        ed = self._edge_detection
        ed.kernel_size_sp.valueChanged.emit(ed.kernel_size_sp.value())
        ed.sigma_sp.valueChanged.emit(ed.sigma_sp.value())
        ed.threshold_le.returnPressed.emit()

        return True

    def loadMetaData(self):
        """Override."""
        cfg = self._meta.hget_all(mt.IMAGE_TRANSFORM_PROC)

        self._ma_window_le.setText(str(cfg["ma_window"]))

        # do not load transform type since it is not an "input"

        fft = self._fourier_transform
        fft.logrithmic_cb.setChecked(cfg["fft:logrithmic"] == 'True')

        ed = self._edge_detection
        ed.kernel_size_sp.setValue(int(cfg["ed:kernel_size"]))
        ed.sigma_sp.setValue(float(cfg["ed:sigma"]))
        ed.threshold_le.setText(cfg['ed:threshold'][1:-1])

    def registerTransformType(self):
        self._mediator.onItTransformTypeChange(
            self._opt_index_map[self._opt_tab.currentIndex()])

    def unregisterTransformType(self):
        self._mediator.onItTransformTypeChange(
            ImageTransformType.UNDEFINED)
