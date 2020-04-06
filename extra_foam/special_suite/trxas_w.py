"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIntValidator
from PyQt5.QtWidgets import QGridLayout, QLabel, QPushButton, QSplitter

from .trxas_proc import TrxasProcessor
from .special_analysis_base import (
    create_special, QThreadFoamClient, _BaseAnalysisCtrlWidgetS,
    _SpecialAnalysisBase
)
from ..gui.plot_widgets import ImageViewF, TimedImageViewF, TimedPlotWidgetF
from ..gui.misc_widgets import FColor
from ..gui.ctrl_widgets.smart_widgets import (
    SmartBoundaryLineEdit, SmartLineEdit, SmartStringLineEdit
)
from ..config import config


_DEFAULT_N_BINS = "10"
_DEFAULT_BIN_RANGE = "0, 1e9"
_MAX_N_BINS = 999
_DEFAULT_DEVICE_ID = "META"
_DEFAULT_PROPERTY = "timestamp.tid"


class TrxasCtrlWidget(_BaseAnalysisCtrlWidgetS):
    """tr-XAS analysis control widget.

    tr-XAS stands for Time-resolved X-ray Absorption Spectroscopy.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.delay_device_le = SmartStringLineEdit(_DEFAULT_DEVICE_ID)
        self.delay_ppt_le = SmartStringLineEdit(_DEFAULT_PROPERTY)

        self.energy_device_le = SmartStringLineEdit(_DEFAULT_DEVICE_ID)
        self.energy_ppt_le = SmartStringLineEdit(_DEFAULT_PROPERTY)

        self.delay_range_le = SmartBoundaryLineEdit(_DEFAULT_BIN_RANGE)
        self.n_delay_bins_le = SmartLineEdit(_DEFAULT_N_BINS)
        self.n_delay_bins_le.setValidator(QIntValidator(1, _MAX_N_BINS))

        self.energy_range_le = SmartBoundaryLineEdit(_DEFAULT_BIN_RANGE)
        self.n_energy_bins_le = SmartLineEdit(_DEFAULT_N_BINS)
        self.n_energy_bins_le.setValidator(QIntValidator(1, _MAX_N_BINS))

        self.swap_btn = QPushButton("Swap delay and energy")

        self._non_reconfigurable_widgets.extend([
            self.delay_device_le,
            self.delay_ppt_le,
            self.energy_device_le,
            self.energy_ppt_le,
            self.swap_btn
        ])

        self.initUI()
        self.initConnections()

    def initUI(self):
        """Override."""
        layout = QGridLayout()
        AR = Qt.AlignRight

        i_row = 0
        layout.addWidget(QLabel("Delay device ID: "), i_row, 0, AR)
        layout.addWidget(self.delay_device_le, i_row, 1)

        i_row += 1
        layout.addWidget(QLabel("Delay device property: "), i_row, 0, AR)
        layout.addWidget(self.delay_ppt_le, i_row, 1)

        i_row += 1
        layout.addWidget(QLabel("Mono device ID: "), i_row, 0, AR)
        layout.addWidget(self.energy_device_le, i_row, 1)

        i_row += 1
        layout.addWidget(QLabel("Mono device property: "), i_row, 0, AR)
        layout.addWidget(self.energy_ppt_le, i_row, 1)

        i_row += 1
        layout.addWidget(QLabel("Delay range: "), i_row, 0, AR)
        layout.addWidget(self.delay_range_le, i_row, 1)

        i_row += 1
        layout.addWidget(QLabel("# of delay bins: "), i_row, 0, AR)
        layout.addWidget(self.n_delay_bins_le, i_row, 1)

        i_row += 1
        layout.addWidget(QLabel("Energy range: "), i_row, 0, AR)
        layout.addWidget(self.energy_range_le, i_row, 1)

        i_row += 1
        layout.addWidget(QLabel("# of energy bins: "), i_row, 0, AR)
        layout.addWidget(self.n_energy_bins_le, i_row, 1)

        i_row += 1
        layout.addWidget(self.swap_btn, i_row, 1)

        self.setLayout(layout)

    def initConnections(self):
        """Override."""
        pass

    def _onDelaySourceChange(self):
        device_id = self._delay_device_le.text()
        ppt = self._delay_ppt_le.text()
        src = f"{device_id} {ppt}" if device_id and ppt else ""

    def _onEnergySourceChange(self):
        device_id = self._energy_device_le.text()
        ppt = self._energy_ppt_le.text()
        src = f"{device_id} {ppt}" if device_id and ppt else ""

    def _swapEnergyDelay(self):
        self._swapLineEditContent(self._delay_device_le, self._energy_device_le)
        self._swapLineEditContent(self._delay_ppt_le, self._energy_ppt_le)
        self._swapLineEditContent(self._delay_range_le, self._energy_range_le)
        self._swapLineEditContent(self._n_delay_bins_le, self._n_energy_bins_le)
        self._scan_btn_set.reset_sgn.emit()

    def _swapLineEditContent(self, edit1, edit2):
        text1 = edit1.text()
        text2 = edit2.text()
        edit1.setText(text2)
        edit2.setText(text1)


class TrxasRoiImageView(ImageViewF):
    """TrxasRoiImageView class.

    Visualize ROIs.
    """
    def __init__(self, idx, **kwargs):
        """Initialization."""
        super().__init__(has_roi=False, **kwargs)

        self._index = idx
        self.setTitle(f"ROI{idx}")

    def updateF(self, data):
        """Override."""
        self.setImage(data[f"roi{self._index}"])


class TrxasAbsorptionPlot(TimedPlotWidgetF):
    """TrxasAbsorptionPlot class.

    Visualize absorption(s) binned by time delay.
    """
    def __init__(self, diff=False, *, parent=None):
        """Initialization."""
        super().__init__(parent=parent)

        self._diff = diff

        self.setLabel('left', "Absorption (arb. u.)")
        self.setLabel('bottom', "Delay (arb. u.)")
        self.addLegend(offset=(-40, 20))

        if diff:
            self._a21 = self.plotCurve(name="ROI2/ROI1", pen=FColor.mkPen("g"))
        else:
            c = config['GUI_ROI_COLORS']
            self._a13 = self.plotCurve(name="ROI1/ROI3", pen=FColor.mkPen(c[0]))
            self._a23 = self.plotCurve(name="ROI2/ROI3", pen=FColor.mkPen(c[1]))

    def refresh(self):
        """Override."""
        data = self._data

        delay = data["delay_bin_centers"]
        if delay is None:
            return

        if self._diff:
            self._a21.setData(delay, data["a21_stats"])
        else:
            self._a13.setData(delay, data["a13_stats"])
            self._a23.setData(delay, data["a23_stats"])


class TrxasHeatmap(TimedImageViewF):
    """TrxasHeatmap class.

    Visualize absorption binned by delay and energy.
    """
    def __init__(self, *, parent=None):
        """Initialization."""
        super().__init__(has_roi=False, hide_axis=False, parent=parent)
        self.invertY(False)
        self.setAspectLocked(False)

        self.setLabel('bottom', 'Energy (eV)')
        self.setLabel('left', 'Delay (arb. u.)')
        self.setTitle("Absorption (ROI2/ROI1)")

    def refresh(self):
        """Override."""
        data = self._data

        energy = data["energy_bin_centers"]
        delay = data["delay_bin_centers"]
        heat = data["a21_heat"]

        if energy is None or delay is None:
            return

        # do not update if FOM is None
        if heat is not None:
            self.setImage(heat,
                          pos=[energy[0], delay[0]],
                          scale=[(energy[-1] - energy[0])/len(energy),
                                 (delay[-1] - delay[0])/len(delay)])


@create_special(TrxasCtrlWidget, TrxasProcessor, QThreadFoamClient)
class TrxasWindow(_SpecialAnalysisBase):
    """Main GUI for tr-XAS analysis."""

    _title = "tr-XAS"
    _long_title = "Time-resolved X-ray Absorption Spectroscopy"

    def __init__(self, topic):
        """Initialization."""
        super().__init__(topic, with_dark=False)

        self._roi1_image = TrxasRoiImageView(1, parent=self)
        self._roi2_image = TrxasRoiImageView(2, parent=self)
        self._roi3_image = TrxasRoiImageView(3, parent=self)

        self._a13_a23 = TrxasAbsorptionPlot(parent=self)
        self._a21 = TrxasAbsorptionPlot(True, parent=self)
        self._a21_heatmap = TrxasHeatmap(parent=self)

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

        self._cw.addWidget(self._left_panel)
        self._cw.addWidget(middle_panel)
        self._cw.addWidget(right_panel)

        self.resize(self._TOTAL_W, self._TOTAL_H)

    def initConnections(self):
        """Override."""
        self._ctrl_widget.delay_device_le.value_changed_sgn.connect(
            self._worker.onDelayDeviceChanged)
        self._ctrl_widget.delay_ppt_le.value_changed_sgn.connect(
            self._worker.onDelayPropertyChanged)

        self._ctrl_widget.delay_device_le.returnPressed.emit()
        self._ctrl_widget.delay_ppt_le.returnPressed.emit()

        self._ctrl_widget.energy_device_le.value_changed_sgn.connect(
            self._worker.onEnergyDeviceChanged)
        self._ctrl_widget.energy_ppt_le.value_changed_sgn.connect(
            self._worker.onEnergyPropertyChanged)

        self._ctrl_widget.energy_device_le.returnPressed.emit()
        self._ctrl_widget.energy_ppt_le.returnPressed.emit()

        self._ctrl_widget.n_delay_bins_le.value_changed_sgn.connect(
            self._worker.onNoDelayBinsChanged)
        self._ctrl_widget.delay_range_le.value_changed_sgn.connect(
            self._worker.onDelayRangeChanged)

        self._ctrl_widget.n_delay_bins_le.returnPressed.emit()
        self._ctrl_widget.delay_range_le.returnPressed.emit()

        self._ctrl_widget.n_energy_bins_le.value_changed_sgn.connect(
            self._worker.onNoEnergyBinsChanged)
        self._ctrl_widget.energy_range_le.value_changed_sgn.connect(
            self._worker.onEnergyRangeChanged)

        self._ctrl_widget.n_energy_bins_le.returnPressed.emit()
        self._ctrl_widget.energy_range_le.returnPressed.emit()

        self.reset_sgn.connect(self._worker.onReset)
