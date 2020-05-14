"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIntValidator
from PyQt5.QtWidgets import QPushButton, QSplitter

from extra_foam.gui.ctrl_widgets.smart_widgets import (
    SmartBoundaryLineEdit, SmartLineEdit, SmartStringLineEdit
)
from extra_foam.gui.misc_widgets import FColor
from extra_foam.gui.plot_widgets import (
    ImageViewF, TimedImageViewF, TimedPlotWidgetF
)

from .special_analysis_base import (
    create_special, QThreadFoamClient, _BaseAnalysisCtrlWidgetS,
    _SpecialAnalysisBase
)
from .trxas_proc import (
    TrXasProcessor, _DEFAULT_N_BINS, _DEFAULT_BIN_RANGE
)

_MAX_N_BINS = 999


class TrXasCtrlWidget(_BaseAnalysisCtrlWidgetS):
    """tr-XAS analysis control widget.

    tr-XAS stands for Time-resolved X-ray Absorption Spectroscopy.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.device_id1_le = SmartStringLineEdit(
            "SCS_ILH_LAS/MOTOR/LT3")
        self.ppt1_le = SmartStringLineEdit("AActualPosition")
        self.label1_le = SmartStringLineEdit("Delay (arb. u.)")

        self.device_id2_le = SmartStringLineEdit(
            "SA3_XTD10_MONO/MDL/PHOTON_ENERGY")
        self.ppt2_le = SmartStringLineEdit("actualEnergy")
        self.label2_le = SmartStringLineEdit("Energy (eV)")

        self.bin_range1_le = SmartBoundaryLineEdit(_DEFAULT_BIN_RANGE)
        self.n_bins1_le = SmartLineEdit(str(_DEFAULT_N_BINS))
        self.n_bins1_le.setValidator(QIntValidator(1, _MAX_N_BINS))

        self.bin_range2_le = SmartBoundaryLineEdit(_DEFAULT_BIN_RANGE)
        self.n_bins2_le = SmartLineEdit(str(_DEFAULT_N_BINS))
        self.n_bins2_le.setValidator(QIntValidator(1, _MAX_N_BINS))

        self.swap_btn = QPushButton("Swap devices")

        self._non_reconfigurable_widgets.extend([
            self.device_id1_le,
            self.ppt1_le,
            self.device_id2_le,
            self.ppt2_le,
            self.swap_btn
        ])

        self.initUI()
        self.initConnections()

    def initUI(self):
        """Override."""
        layout = self.layout()

        layout.addRow("Device ID 1: ", self.device_id1_le)
        layout.addRow("Property 1: ", self.ppt1_le)
        layout.addRow("Label 1: ", self.label1_le)
        layout.addRow("Bin range 1: ", self.bin_range1_le)
        layout.addRow("# of bins 1: ", self.n_bins1_le)
        layout.addRow("Device ID 2: ", self.device_id2_le)
        layout.addRow("Property 2: ", self.ppt2_le)
        layout.addRow("Label 2: ", self.label2_le)
        layout.addRow("Bin range 2: ", self.bin_range2_le)
        layout.addRow("# of bins 2: ", self.n_bins2_le)
        layout.addRow("", self.swap_btn)

    def initConnections(self):
        """Override."""
        self.swap_btn.clicked.connect(self._swapDataSources)

    def _swapDataSources(self):
        self._swapLineEditContent(self.device_id1_le, self.device_id2_le)
        self._swapLineEditContent(self.ppt1_le, self.ppt2_le)
        self._swapLineEditContent(self.label1_le, self.label2_le)
        self._swapLineEditContent(self.bin_range1_le, self.bin_range2_le)
        self._swapLineEditContent(self.n_bins1_le, self.n_bins2_le)

    def _swapLineEditContent(self, edit1, edit2):
        text1 = edit1.text()
        text2 = edit2.text()
        edit1.setText(text2)
        edit2.setText(text1)


class TrXasRoiImageView(ImageViewF):
    """TrXasRoiImageView class.

    Visualize ROIs.
    """
    def __init__(self, idx, **kwargs):
        """Initialization."""
        super().__init__(**kwargs)

        self._index = idx
        self.setTitle(f"ROI{idx}")

    def updateF(self, data):
        """Override."""
        self.setImage(data[f"roi{self._index}"])


class TrXasSpectraPlot(TimedPlotWidgetF):
    """TrXasSpectraPlot class.

    Visualize 1D binning of absorption(s).
    """
    def __init__(self, diff=False, *, parent=None):
        """Initialization."""
        super().__init__(parent=parent, show_indicator=True)

        self._diff = diff

        self.setTitle("XAS")
        self.setLabel('left', "Absorption (arb. u.)")
        self.setLabel('right', "Count")
        self.addLegend(offset=(-40, 20))

        if diff:
            self._a21 = self.plotCurve(name="ROI2/ROI1", pen=FColor.mkPen("g"))
        else:
            # same color as ROI1
            self._a13 = self.plotCurve(name="ROI1/ROI3", pen=FColor.mkPen("b"))
            # same color as ROI2
            self._a23 = self.plotCurve(name="ROI2/ROI3", pen=FColor.mkPen("r"))

        self._count = self.plotBar(y2=True, brush=FColor.mkBrush('i', alpha=70))

    def refresh(self):
        """Override."""
        data = self._data

        centers1 = data["centers1"]
        if centers1 is None:
            return

        if self._diff:
            self._a21.setData(centers1, data["a21_stats"])
        else:
            self._a13.setData(centers1, data["a13_stats"])
            self._a23.setData(centers1, data["a23_stats"])
        self._count.setData(centers1, data["counts1"])

    def onXLabelChanged(self, label):
        self.setLabel('bottom', label)


class TrXasHeatmap(TimedImageViewF):
    """TrXasHeatmap class.

    Visualize 2D binning of absorption.
    """
    def __init__(self, *, parent=None):
        """Initialization."""
        super().__init__(hide_axis=False, parent=parent)
        self.invertY(False)
        self.setAspectLocked(False)

        self.setTitle("XAS (ROI2/ROI1)")

    def refresh(self):
        """Override."""
        data = self._data

        centers2 = data["centers2"]
        centers1 = data["centers1"]
        heat = data["a21_heat"]

        if centers2 is None or centers1 is None:
            return

        # do not update if FOM is None
        if heat is not None:
            self.setImage(heat,
                          pos=[centers2[0], centers1[0]],
                          scale=[(centers2[-1] - centers2[0])/len(centers2),
                                 (centers1[-1] - centers1[0])/len(centers1)])

    def onXLabelChanged(self, label):
        self.setLabel('bottom', label)

    def onYLabelChanged(self, label):
        self.setLabel('left', label)


@create_special(TrXasCtrlWidget, TrXasProcessor, QThreadFoamClient)
class TrXasWindow(_SpecialAnalysisBase):
    """Main GUI for tr-XAS analysis."""

    icon = "tr_xas.png"
    _title = "tr-XAS"
    _long_title = "Time-resolved X-ray Absorption Spectroscopy"

    def __init__(self, topic):
        """Initialization."""
        super().__init__(topic, with_dark=False)

        self._roi1_image = TrXasRoiImageView(1, parent=self)
        self._roi2_image = TrXasRoiImageView(2, parent=self)
        self._roi3_image = TrXasRoiImageView(3, parent=self)

        self._a13_a23 = TrXasSpectraPlot(parent=self)
        self._a21 = TrXasSpectraPlot(True, parent=self)
        self._a21_heatmap = TrXasHeatmap(parent=self)

        self.initUI()
        self.initConnections()
        self.startWorker()

    def initUI(self):
        """Override."""
        middle_panel = QSplitter(Qt.Vertical)
        middle_panel.addWidget(self._roi1_image)
        middle_panel.addWidget(self._roi2_image)
        middle_panel.addWidget(self._roi3_image)

        right_panel = QSplitter(Qt.Vertical)
        right_panel.addWidget(self._a13_a23)
        right_panel.addWidget(self._a21)
        right_panel.addWidget(self._a21_heatmap)
        right_panel.setSizes([self._TOTAL_H / 3.0] * 3)

        cw = self.centralWidget()
        cw.addWidget(middle_panel)
        cw.addWidget(right_panel)

        self.resize(self._TOTAL_W, self._TOTAL_H)

    def initConnections(self):
        """Override."""
        self._ctrl_widget_st.device_id1_le.value_changed_sgn.connect(
            self._worker_st.onDeviceId1Changed)
        self._ctrl_widget_st.ppt1_le.value_changed_sgn.connect(
            self._worker_st.onProperty1Changed)

        self._ctrl_widget_st.device_id1_le.returnPressed.emit()
        self._ctrl_widget_st.ppt1_le.returnPressed.emit()

        self._ctrl_widget_st.device_id2_le.value_changed_sgn.connect(
            self._worker_st.onDeviceId2Changed)
        self._ctrl_widget_st.ppt2_le.value_changed_sgn.connect(
            self._worker_st.onProperty2Changed)

        self._ctrl_widget_st.device_id2_le.returnPressed.emit()
        self._ctrl_widget_st.ppt2_le.returnPressed.emit()

        self._ctrl_widget_st.n_bins1_le.value_changed_sgn.connect(
            self._worker_st.onNBins1Changed)
        self._ctrl_widget_st.bin_range1_le.value_changed_sgn.connect(
            self._worker_st.onBinRange1Changed)

        self._ctrl_widget_st.n_bins1_le.returnPressed.emit()
        self._ctrl_widget_st.bin_range1_le.returnPressed.emit()

        self._ctrl_widget_st.n_bins2_le.value_changed_sgn.connect(
            self._worker_st.onNBins2Changed)
        self._ctrl_widget_st.bin_range2_le.value_changed_sgn.connect(
            self._worker_st.onBinRange2Changed)

        self._ctrl_widget_st.n_bins2_le.returnPressed.emit()
        self._ctrl_widget_st.bin_range2_le.returnPressed.emit()

        self._ctrl_widget_st.label1_le.value_changed_sgn.connect(
            self._a21.onXLabelChanged)
        self._ctrl_widget_st.label1_le.value_changed_sgn.connect(
            self._a13_a23.onXLabelChanged)
        self._ctrl_widget_st.label1_le.value_changed_sgn.connect(
            self._a21_heatmap.onYLabelChanged)
        self._ctrl_widget_st.label2_le.value_changed_sgn.connect(
            self._a21_heatmap.onXLabelChanged)

        self._ctrl_widget_st.label1_le.returnPressed.emit()
        self._ctrl_widget_st.label2_le.returnPressed.emit()
