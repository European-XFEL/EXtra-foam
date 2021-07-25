"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from enum import Enum
import os
import os.path as osp

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QFileDialog, QFrame, QVBoxLayout, QSplitter

from .base_window import _AbstractPlotWindow
from ..ctrl_widgets import PumpProbeCtrlWidget
from ..plot_widgets import PlotWidgetF, TimedPlotWidgetF
from ..misc_widgets import FColor
from ...config import config, AnalysisType, plot_labels


class PumpProbeVFomPlot(PlotWidgetF):
    """PumpProbeVFomPlot class.

    Widget for displaying the pump and probe signal or their difference.
    """

    def __init__(self, diff=False, *, parent=None):
        """Initialization.

        :param bool diff: True for displaying on-off while False for
            displaying on and off
        """
        super().__init__(parent=parent)

        self._analysis_type = AnalysisType.UNDEFINED
        x_label, y_label = plot_labels[self._analysis_type]
        self.setTitle('VFOM')
        self.setLabel('bottom', x_label)
        self.setLabel('left', y_label)
        self.addLegend(offset=(-40, 20))

        self._is_diff = diff
        if diff:
            self._on_off_pulse = self.plotCurve(name="On - Off", pen=FColor.mkPen("p"))
        else:
            self._on_pulse = self.plotCurve(name="On", pen=FColor.mkPen("r"))
            self._off_pulse = self.plotCurve(name="Off", pen=FColor.mkPen("b"))

    def updateF(self, data):
        """Override."""
        pp = data.pp
        x, y = pp.x, pp.y

        if self._analysis_type != pp.analysis_type:
            x_label, y_label = plot_labels[pp.analysis_type]
            self.setLabel('bottom', x_label)
            self.setLabel('left', y_label)
            self._analysis_type = pp.analysis_type
            self.reset()

        if y is None:
            return

        if self._is_diff:
            self._on_off_pulse.setData(x, y)
        else:
            y_on, y_off = pp.y_on, pp.y_off
            self._on_pulse.setData(x, y_on)
            self._off_pulse.setData(x, y_off)


class PumpProbeFomPlot(TimedPlotWidgetF):
    """PumpProbeFomPlot class.

    Widget for displaying the evolution of FOM in pump-probe analysis.
    """

    def __init__(self, *, parent=None):
        """Initialization."""
        super().__init__(parent=parent)

        self.setLabel('bottom', "Train ID")
        self.setLabel('left', "FOM (arb. u.)")
        self.setTitle('FOM correlation')

        self._plot = self.plotScatter()

    def refresh(self):
        """Override."""
        pp = self._data.corr.pp
        x, y = pp.x, pp.y
        self._plot.setData(x, y)


class SaveFile(Enum):
    NPZ = "NumPy Binary File (*.npz)"
    TXT = "Text File (*.txt)"


class PumpProbeWindow(_AbstractPlotWindow):
    """PumpProbeWindow class."""
    _title = "Pump-probe"

    _TOTAL_W, _TOTAL_H = config['GUI_PLOT_WINDOW_SIZE']

    def __init__(self, *args, **kwargs):
        """Initialization."""
        super().__init__(*args, **kwargs)

        self._ctrl_widget = self.createCtrlWidget(PumpProbeCtrlWidget)

        self._pp_fom = PumpProbeFomPlot(parent=self)

        self._pp_onoff = PumpProbeVFomPlot(parent=self)
        self._pp_diff = PumpProbeVFomPlot(diff=True, parent=self)

        self.initUI()
        self.initConnections()
        self.loadMetaData()
        self.updateMetaData()

        self.resize(self._TOTAL_W, self._TOTAL_H)
        self.setMinimumSize(0.6*self._TOTAL_W, 0.6*self._TOTAL_H)

        self.update()

    def initUI(self):
        """Override."""
        plots = QSplitter()
        right_panel = QSplitter(Qt.Vertical)
        right_panel.addWidget(self._pp_onoff)
        right_panel.addWidget(self._pp_diff)
        plots.addWidget(self._pp_fom)
        plots.addWidget(right_panel)

        self._cw = QFrame()
        layout = QVBoxLayout()
        layout.addWidget(plots)
        layout.addWidget(self._ctrl_widget)
        self._ctrl_widget.setFixedHeight(
            self._ctrl_widget.minimumSizeHint().height())
        self._cw.setLayout(layout)
        self.setCentralWidget(self._cw)

    def initConnections(self):
        """Override."""
        self._ctrl_widget.save_btn.clicked.connect(self.saveToFile)

    def closeEvent(self, QCloseEvent):
        self._ctrl_widget.resetAnalysisType()
        super().closeEvent(QCloseEvent)

    def saveToFile(self):
        # Get the current data
        if not len(self._queue):
            return
        data = self._queue[0]

        # Open file dialog
        filepath, filter = QFileDialog.getSaveFileName(
            caption="Save image",
            directory=osp.expanduser("~"),
            filter=f"{SaveFile.NPZ.value};;{SaveFile.TXT.value}")
        filter = SaveFile(filter)

        # Validate filepath
        if not filepath:
            return
        suffix = f".{filter.name.lower()}"
        if not filepath.lower().endswith(suffix):
            filepath += suffix

        # Copy reference file from tmp folder to desired destination
        os.makedirs(osp.dirname(filepath), exist_ok=True)

        # Save the data
        pp = data.pp
        train_id = data.tid
        position = pp.x
        intensity_on = pp.y_on
        intensity_off = pp.y_off
        intensity_sub = pp.y

        if position is None:
            return

        if filter is SaveFile.NPZ:
            np.savez(filepath,
                     trainId=train_id,
                     position=position,
                     intensity_on=intensity_on,
                     intensity_off=intensity_off,
                     intensity_subtracted=intensity_sub)
        elif filter is SaveFile.TXT:
            # write out conversion to tabular ASCII
            with open(filepath, 'w') as f:
                f.write(f" Train-ID: {train_id}\n\n")
                f.write('       q            I_on     '
                        '       I_off           I_diff\n\n')
                for i, q in enumerate(position):
                    f.write(f"{q:12f}"
                            f"{intensity_on[i]:16f}"
                            f"{intensity_off[i]:16f}"
                            f"{intensity_sub[i]:16f}\n")
