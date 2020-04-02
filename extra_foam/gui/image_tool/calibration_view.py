"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import os.path as osp

from PyQt5.QtCore import pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import QFileDialog, QGridLayout, QVBoxLayout

from .base_view import _AbstractImageToolView, create_imagetool_view
from .simple_image_data import _SimpleImageData
from ..ctrl_widgets import CalibrationCtrlWidget
from ..plot_widgets import ImageViewF, ImageAnalysis


@create_imagetool_view(CalibrationCtrlWidget)
class CalibrationView(_AbstractImageToolView):
    """CalibrationView class.

    Widget for visualizing image calibration.
    """

    gain_const_path_sgn = pyqtSignal(str)
    offset_const_path_sgn = pyqtSignal(str)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._raw = ImageViewF(hide_axis=False)
        self._raw.setTitle("Raw")
        self._corrected = ImageAnalysis(hide_axis=False)
        self._corrected.setTitle("Corrected")
        self._gain = ImageViewF(hide_axis=False)
        self._gain.setTitle("Gain")
        self._offset = ImageViewF(hide_axis=False)
        self._offset.setTitle("Offset")

        self.initUI()
        self.initConnections()

    def initUI(self):
        """Override."""
        view_splitter = QGridLayout()
        view_splitter.addWidget(self._raw, 0, 0)
        view_splitter.addWidget(self._corrected, 1, 0)
        view_splitter.addWidget(self._gain, 0, 1)
        view_splitter.addWidget(self._offset, 1, 1)

        layout = QVBoxLayout()
        layout.addLayout(view_splitter)
        layout.addWidget(self._ctrl_widget)
        self.setLayout(layout)

    def initConnections(self):
        """Override."""
        self._ctrl_widget.load_gain_btn.clicked.connect(self._loadGainConst)
        self._ctrl_widget.load_offset_btn.clicked.connect(
            self._loadOffsetConst)

        self._ctrl_widget.remove_gain_btn.clicked.connect(self._removeGain)
        self._ctrl_widget.remove_offset_btn.clicked.connect(
            self._removeOffset)

        self.gain_const_path_sgn.connect(
            self._ctrl_widget.onGainConstPathChanged)
        self.offset_const_path_sgn.connect(
            self._ctrl_widget.onOffsetConstPathChanged)

    def updateF(self, data, auto_update):
        """Override."""
        if auto_update or self._corrected.image is None:
            self._corrected.setImageData(_SimpleImageData(data.image))
            self._offset.setImage(data.image.offset_mean)
            self._gain.setImage(data.image.gain_mean)

    @pyqtSlot()
    def _loadGainConst(self):
        filepath = QFileDialog.getOpenFileName(
            caption="Load gain", directory=osp.expanduser("~"))[0]

        if filepath:
            self.gain_const_path_sgn.emit(filepath)

    @pyqtSlot()
    def _loadOffsetConst(self):
        filepath = QFileDialog.getOpenFileName(
            caption="Load offset", directory=osp.expanduser("~"))[0]

        if filepath:
            self.offset_const_path_sgn.emit(filepath)

    @pyqtSlot()
    def _removeGain(self):
        self.gain_const_path_sgn.emit("")

    @pyqtSlot()
    def _removeOffset(self):
        self.offset_const_path_sgn.emit("")

    def onDeactivated(self):
        btn = self._ctrl_widget.record_dark_btn
        if btn.isChecked():
            btn.setChecked(False)
