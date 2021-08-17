"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import abc

from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QDoubleValidator, QIntValidator
from PyQt5.QtWidgets import (
    QCheckBox, QDoubleSpinBox, QFrame, QGridLayout, QHBoxLayout, QVBoxLayout,
    QLabel, QPushButton, QSpinBox, QTabWidget, QWidget, QComboBox
)

from .smart_widgets import SmartLineEdit, SmartBoundaryLineEdit
from .base_ctrl_widgets import _AbstractCtrlWidget
from ...config import config, ImageTransformType
from ...database import Metadata as mt

from extra_foam.algorithms import (
    ConcentricRingsFinder, find_peaks_1d
)


class _FeatureExtractionMixIn:
    @abc.abstractmethod
    def extractFeature(self, img):
        pass


class _ConcentricRingsCtrlWidget(_FeatureExtractionMixIn, QWidget):
    """Concentric rings detection control widget."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._cx = 0
        self._cy = 0
        self._prominence = 200
        self._distance = 10
        self._min_count = 500

        self.cx_le = SmartLineEdit(str(self._cx))
        self.cx_le.setValidator(QDoubleValidator())
        self.cy_le = SmartLineEdit(str(self._cy))
        self.cy_le.setValidator(QDoubleValidator())
        self.prominence_le = SmartLineEdit(str(self._prominence))
        validator = QDoubleValidator()
        validator.setBottom(0)
        self.prominence_le.setValidator(validator)
        self.distance_le = SmartLineEdit(str(self._distance))
        self.distance_le.setValidator(QIntValidator(1, 99999))
        self.min_count_le = SmartLineEdit(str(self._min_count))
        self.min_count_le.setValidator(QIntValidator(1, 99999))
        self.detect_btn = QPushButton("Detect")

        pixel1 = config["PIXEL_SIZE"]  # y
        pixel2 = pixel1  # x
        self._finder = ConcentricRingsFinder(pixel2, pixel1)

        self.initUI()
        self.initConnections()

    def initUI(self):
        layout = QHBoxLayout()
        layout.addWidget(QLabel("Cx: "))
        layout.addWidget(self.cx_le)
        layout.addWidget(QLabel("Cy: "))
        layout.addWidget(self.cy_le)
        layout.addWidget(QLabel("Prominence: "))
        layout.addWidget(self.prominence_le)
        layout.addWidget(QLabel("Distance: "))
        layout.addWidget(self.distance_le)
        layout.addWidget(QLabel("Min. count: "))
        layout.addWidget(self.min_count_le)
        layout.addWidget(self.detect_btn)
        self.setLayout(layout)

    def initConnections(self):
        self.cx_le.value_changed_sgn.connect(self._onCxChanged)
        self.cy_le.value_changed_sgn.connect(self._onCyChanged)
        self.prominence_le.value_changed_sgn.connect(self._onProminenceChanged)
        self.distance_le.value_changed_sgn.connect(self._onDistanceChanged)
        self.min_count_le.value_changed_sgn.connect(self._onMinCountChanged)

    def _onCxChanged(self, v):
        self._cx = float(v)

    def _onCyChanged(self, v):
        self._cy = float(v)

    def _onProminenceChanged(self, v):
        self._prominence = float(v)

    def _onDistanceChanged(self, v):
        self._distance = int(v)

    def _onMinCountChanged(self, v):
        self._min_count = int(v)

    def extractFeature(self, img):
        """Override."""
        cx, cy = self._finder.search(
            img, self._cx, self._cy, min_count=self._min_count)
        s, q = self._finder.integrate(img, cx, cy, min_count=self._min_count)
        radials = s[find_peaks_1d(q,
                                  prominence=self._prominence,
                                  distance=self._distance)[0]]

        self.cx_le.setText(str(cx))
        self.cy_le.setText(str(cy))

        return cx, cy, radials


class _FourierTransformCtrlWidget(_FeatureExtractionMixIn, QWidget):
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


class _EdgeDetectionCtrlWidget(_FeatureExtractionMixIn, QWidget):
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


class _BraggPeakCtrlWidget(_FeatureExtractionMixIn, QWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.add_roi_btn = QPushButton("Add ROI")
        self.window_size_le = SmartLineEdit("10")
        self.window_size_le.setValidator(QIntValidator())
        self.delete_roi_btn = QPushButton("Delete ROI")
        self._roi_cb = QComboBox()

        self.initUI()
        self.initConnections()

    def initUI(self):
        hbox_layout = QHBoxLayout()
        layout = QVBoxLayout()

        window_size_layout = QHBoxLayout()
        window_size_label = QLabel("Rolling window size:")
        window_size_layout.addWidget(window_size_label)
        window_size_layout.addWidget(self.window_size_le)

        delete_roi_layout = QHBoxLayout()
        delete_roi_layout.addWidget(self.delete_roi_btn)
        delete_roi_layout.addWidget(self._roi_cb)
        self._roi_cb.addItem("")
        self.delete_roi_btn.setEnabled(False)

        layout.addWidget(self.add_roi_btn)
        layout.addLayout(window_size_layout)
        layout.addLayout(delete_roi_layout)
        layout.addStretch()

        hbox_layout.addLayout(layout)
        hbox_layout.addStretch(10)
        self.setLayout(hbox_layout)

    def initConnections(self):
        self._roi_cb.currentTextChanged.connect(
            self._onSelectedRoiChanged
        )

    @pyqtSlot(str)
    def _onSelectedRoiChanged(self, label):
        self.delete_roi_btn.setEnabled(label != "")

    @pyqtSlot(object)
    def roiAdded(self, roi):
        self._roi_cb.addItem(roi.label())

    @property
    def selectedRoi(self):
        return self._roi_cb.currentText()

    def onRoiDeleted(self, label):
        index = self._roi_cb.findText(label)
        self._roi_cb.removeItem(index)

class ImageTransformCtrlWidget(_AbstractCtrlWidget):
    """Control widget for image transform in the ImageTool."""

    extract_concentric_rings_sgn = pyqtSignal()

    transform_type_changed_sgn = pyqtSignal(int)

    roi_requested_sgn = pyqtSignal()
    delete_roi_sgn = pyqtSignal(str)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._concentric_rings = _ConcentricRingsCtrlWidget()
        self._fourier_transform = _FourierTransformCtrlWidget()
        self._edge_detection = _EdgeDetectionCtrlWidget()
        self._bragg_peak = _BraggPeakCtrlWidget()

        self._opt_tab = QTabWidget()
        self._opt_tab.addTab(self._concentric_rings, "Concentric rings")
        self._opt_tab.addTab(self._fourier_transform, "Fourier transform")
        self._opt_tab.addTab(self._edge_detection, "Edge detection")
        self._opt_tab.addTab(self._bragg_peak, "Bragg peak analysis")

        self._non_reconfigurable_widgets = [
            self._concentric_rings.detect_btn,
        ]

        self.initUI()
        self.initConnections()

    def initUI(self):
        """Override."""
        layout = QHBoxLayout()
        layout.addWidget(self._opt_tab)
        self.setLayout(layout)

        self.setFixedHeight(self.minimumSizeHint().height())

    def initConnections(self):
        """Override."""
        mediator = self._mediator

        self._opt_tab.currentChanged.connect(mediator.onItTransformTypeChange)
        self._opt_tab.currentChanged.connect(self.transform_type_changed_sgn)

        cr = self._concentric_rings
        cr.detect_btn.clicked.connect(self.extract_concentric_rings_sgn)
        cr.cx_le.value_changed_sgn.connect(mediator.onItCrCxChange)
        cr.cy_le.value_changed_sgn.connect(mediator.onItCrCyChange)
        cr.prominence_le.value_changed_sgn.connect(
            mediator.onItCrProminenceChange)
        cr.distance_le.value_changed_sgn.connect(
            mediator.onItCrDistanceChange)
        cr.min_count_le.value_changed_sgn.connect(
            mediator.onItCrMinCountChange)

        fft = self._fourier_transform
        fft.logrithmic_cb.toggled.connect(mediator.onItFftLogrithmicScaleChange)

        ed = self._edge_detection
        ed.kernel_size_sp.valueChanged.connect(
            mediator.onItEdKernelSizeChange)
        ed.sigma_sp.valueChanged.connect(mediator.onItEdSigmaChange)
        ed.threshold_le.value_changed_sgn.connect(
            mediator.onItEdThresholdChange)

        self._bragg_peak.add_roi_btn.clicked.connect(self.roi_requested_sgn)
        self._bragg_peak.delete_roi_btn.clicked.connect(
            lambda: self.delete_roi_sgn.emit(self._bragg_peak.selectedRoi)
        )
        self._bragg_peak.window_size_le.value_changed_sgn.connect(
            mediator.onItBraggPeakWindowSizeChange)

    @pyqtSlot(object)
    def onRoiAdded(self, roi):
        self._bragg_peak.roiAdded(roi)

    def updateMetaData(self):
        """Override."""
        if self.isVisible():
            self.registerTransformType()
        else:
            self.unregisterTransformType()

        cr = self._concentric_rings
        cr.cx_le.returnPressed.emit()
        cr.cy_le.returnPressed.emit()
        cr.prominence_le.returnPressed.emit()
        cr.distance_le.returnPressed.emit()
        cr.min_count_le.returnPressed.emit()

        fft = self._fourier_transform
        fft.logrithmic_cb.toggled.emit(fft.logrithmic_cb.isChecked())

        ed = self._edge_detection
        ed.kernel_size_sp.valueChanged.emit(ed.kernel_size_sp.value())
        ed.sigma_sp.valueChanged.emit(ed.sigma_sp.value())
        ed.threshold_le.returnPressed.emit()

        bp = self._bragg_peak
        bp.window_size_le.value_changed_sgn.emit(bp.window_size_le.value())

        return True

    def loadMetaData(self):
        """Override."""
        cfg = self._meta.hget_all(mt.IMAGE_TRANSFORM_PROC)

        # do not load transform type since it is not an "input"

        cr = self._concentric_rings
        self._updateWidgetValue(cr.cx_le, cfg, "cr:cx")
        self._updateWidgetValue(cr.cy_le, cfg, "cr:cy")
        self._updateWidgetValue(cr.prominence_le, cfg, "cr:prominence")
        self._updateWidgetValue(cr.distance_le, cfg, "cr:distance")
        self._updateWidgetValue(cr.min_count_le, cfg, "cr:min_count")

        fft = self._fourier_transform
        self._updateWidgetValue(fft.logrithmic_cb, cfg, "fft:logrithmic")

        ed = self._edge_detection
        self._updateWidgetValue(
            ed.kernel_size_sp, cfg, "ed:kernel_size", cast=int)
        self._updateWidgetValue(ed.sigma_sp, cfg, "ed:sigma", cast=float)
        self._updateWidgetValue(ed.threshold_le, cfg, "ed:threshold")

    def registerTransformType(self):
        self._opt_tab.currentChanged.emit(self._opt_tab.currentIndex())

    def unregisterTransformType(self):
        self._mediator.onItTransformTypeChange(
            int(ImageTransformType.UNDEFINED))

    def extractFeature(self, img):
        return self._opt_tab.currentWidget().extractFeature(img)
