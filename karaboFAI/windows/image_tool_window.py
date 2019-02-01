"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

ImageToolWindow.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from ..widgets.pyqtgraph import QtCore, QtGui

from ..widgets import ImageView
from .base_window import AbstractWindow, SingletonWindow
from ..logger import logger


class ROICtrlWidget(QtGui.QGroupBox):

    roi_region_changed_sgn = QtCore.Signal(float, float)

    def __init__(self, title, *, parent=None):
        """"""
        super().__init__(title, parent=parent)

        self._width_le = QtGui.QLineEdit()
        self._height_le = QtGui.QLineEdit()
        self._width_le.editingFinished.connect(self.roiRegionChangedEvent)
        self._height_le.editingFinished.connect(self.roiRegionChangedEvent)

        self._lock_cb = QtGui.QCheckBox("Lock")
        self.activate_cb = QtGui.QCheckBox("Activate")

        self.initUI()

    def initUI(self):
        wh_layout = QtGui.QHBoxLayout()
        wh_layout.addWidget(QtGui.QLabel("Width: "))
        wh_layout.addWidget(self._width_le)
        wh_layout.addWidget(QtGui.QLabel("Height: "))
        wh_layout.addWidget(self._height_le)

        cb_layout = QtGui.QHBoxLayout()
        cb_layout.addWidget(self.activate_cb)
        cb_layout.addWidget(self._lock_cb)

        layout = QtGui.QVBoxLayout()

        layout.addLayout(cb_layout)
        layout.addLayout(wh_layout)

        self.setLayout(layout)

    def updateParameters(self, pos, size):
        self._width_le.setText(str(size[0]))
        self._height_le.setText(str(size[1]))

    def roiRegionChangedEvent(self):
        w = float(self._width_le.text())
        h = float(self._height_le.text())
        self.roi_region_changed_sgn.emit(w, h)


class MaskCtrlWidget(QtGui.QGroupBox):

    def __init__(self, title, *, parent=None):
        """"""
        super().__init__(title, parent=parent)

        self._lock_cb = QtGui.QCheckBox("Lock")

        self.initUI()

    def initUI(self):

        layout = QtGui.QVBoxLayout()

        layout.addWidget(self._lock_cb)

        self.setLayout(layout)


@SingletonWindow
class ImageToolWindow(AbstractWindow):
    """ImageToolWindow class."""
    title = "image tool"

    def __init__(self, data, *, parent=None):
        super().__init__(data, parent=parent)

        self._image_view = ImageView()
        self._image_view.roi1.sigRegionChangeFinished.connect(
            self.roiRegionChangedEvent)
        self._image_view.roi2.sigRegionChangeFinished.connect(
            self.roiRegionChangedEvent)

        self._roi1_ctrl = ROICtrlWidget("ROI 1")
        self._roi2_ctrl = ROICtrlWidget("ROI 2")
        self._roi1_ctrl.activate_cb.stateChanged.connect(
            self.toggleRoiActivationEvent)
        self._roi2_ctrl.activate_cb.stateChanged.connect(
            self.toggleRoiActivationEvent)
        self._roi1_ctrl.roi_region_changed_sgn.connect(self.onRoiRegionChanged)
        self._roi2_ctrl.roi_region_changed_sgn.connect(self.onRoiRegionChanged)

        self._image_view.roi1.setSize(self._image_view.roi1.size())
        self._image_view.roi2.setSize(self._image_view.roi2.size())

        self._mask_panel = MaskCtrlWidget("Masking tool")

        self._update_image_btn = QtGui.QPushButton("Update image")
        self._update_image_btn.clicked.connect(self.updateImage)

        self.initUI()
        self.resize(800, 800)
        self.updateImage()

        logger.info("Open DrawMaskWindow")

    def initUI(self):
        """Override."""
        tool_layout = QtGui.QGridLayout()
        tool_layout.addWidget(self._roi1_ctrl, 0, 0, 1, 1)
        tool_layout.addWidget(self._roi2_ctrl, 1, 0, 1, 1)

        layout = QtGui.QGridLayout()
        layout.addWidget(self._image_view, 0, 0, 1, 1)
        layout.addLayout(tool_layout, 1, 0, 1, 1)
        layout.addWidget(self._mask_panel, 0, 1, 1, 1)
        layout.addWidget(self._update_image_btn, 1, 1, 1, 1)

        self._cw.setLayout(layout)

        self._activate_roi1()
        self._activate_roi2()

    def updateImage(self):
        """For updating image manually."""
        data = self._data.get()
        if data.empty():
            return

        self._image_view.setImage(data.image_mean)

    def toggleRoiActivationEvent(self, state):
        sender = self.sender()
        if sender is self._roi1_ctrl.activate_cb:
            if state == QtCore.Qt.Checked:
                self._image_view.roi1.show()
            else:
                self._image_view.roi1.hide()
        elif sender is self._roi2_ctrl.activate_cb:
            if state == QtCore.Qt.Checked:
                self._image_view.roi2.show()
            else:
                self._image_view.roi2.hide()

    def _activate_roi1(self):
        self._roi1_ctrl.activate_cb.setChecked(True)

    def _activate_roi2(self):
        self._roi2_ctrl.activate_cb.setChecked(True)

    def roiRegionChangedEvent(self):
        sender = self.sender()
        if sender is self._image_view.roi1:
            self._roi1_ctrl.updateParameters(sender.pos(), sender.size())
        elif sender is self._image_view.roi2:
            self._roi2_ctrl.updateParameters(sender.pos(), sender.size())

    @QtCore.pyqtSlot(float, float)
    def onRoiRegionChanged(self, w, h):
        sender = self.sender()
        if sender is self._roi1_ctrl:
            self._image_view.roi1.setSize((w, h))
        elif sender is self._roi2_ctrl:
            self._image_view.roi2.setSize((w, h))
