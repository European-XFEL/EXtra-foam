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


class ROICtrlWidget(QtGui.QGroupBox):
    """Widget for controlling of ROI."""
    # w, h, cx, cy
    roi_region_changed_sgn = QtCore.Signal(float, float, float, float)

    _pos_validator = QtGui.QDoubleValidator(-10000.0, 10000.0, 1)
    _size_validator = QtGui.QDoubleValidator(0.0, 10000.0, 1)

    def __init__(self, title, *, parent=None):
        """Initialization"""
        super().__init__(title, parent=parent)

        self._width_le = QtGui.QLineEdit()
        self._width_le.setValidator(self._size_validator)
        self._height_le = QtGui.QLineEdit()
        self._height_le.setValidator(self._size_validator)
        self._cx_le = QtGui.QLineEdit()
        self._cx_le.setValidator(self._pos_validator)
        self._cy_le = QtGui.QLineEdit()
        self._cy_le.setValidator(self._pos_validator)
        self._width_le.editingFinished.connect(self.roiRegionChangedEvent)
        self._height_le.editingFinished.connect(self.roiRegionChangedEvent)
        self._cx_le.editingFinished.connect(self.roiRegionChangedEvent)
        self._cy_le.editingFinished.connect(self.roiRegionChangedEvent)

        self.activate_cb = QtGui.QCheckBox("Activate")
        self._lock_cb = QtGui.QCheckBox("Lock")
        self.lock_aspect_cb = QtGui.QCheckBox("Lock aspect ratio")

        self.initUI()

    def initUI(self):
        le_layout = QtGui.QHBoxLayout()
        le_layout.addWidget(QtGui.QLabel("Width: "))
        le_layout.addWidget(self._width_le)
        le_layout.addWidget(QtGui.QLabel("Height: "))
        le_layout.addWidget(self._height_le)
        le_layout.addWidget(QtGui.QLabel("X center: "))
        le_layout.addWidget(self._cx_le)
        le_layout.addWidget(QtGui.QLabel("Y center: "))
        le_layout.addWidget(self._cy_le)

        cb_layout = QtGui.QHBoxLayout()
        cb_layout.addWidget(self.activate_cb)
        cb_layout.addWidget(self._lock_cb)
        cb_layout.addWidget(self.lock_aspect_cb)

        layout = QtGui.QVBoxLayout()

        layout.addLayout(cb_layout)
        layout.addLayout(le_layout)

        self.setLayout(layout)

    def updateParameters(self, size, pos):
        digits = 1
        self._width_le.setText(str(round(size[0], digits)))
        self._height_le.setText(str(round(size[1], digits)))
        self._cx_le.setText(str(round(pos[0], digits)))
        self._cy_le.setText(str(round(pos[1], digits)))

    def roiRegionChangedEvent(self):
        w = float(self._width_le.text())
        h = float(self._height_le.text())
        cx = float(self._cx_le.text())
        cy = float(self._cy_le.text())
        self.roi_region_changed_sgn.emit(w, h, cx, cy)


class MaskCtrlWidget(QtGui.QGroupBox):
    """Widget for masking image."""
    def __init__(self, title, *, parent=None):
        """"""
        super().__init__(title, parent=parent)

        self._lock_cb = QtGui.QCheckBox("Not implemented")

        self.initUI()

    def initUI(self):
        layout = QtGui.QVBoxLayout()

        layout.addWidget(self._lock_cb)

        self.setLayout(layout)


@SingletonWindow
class ImageToolWindow(AbstractWindow):
    """ImageToolWindow class.

    This tool provides selecting of ROI and image masking.
    """
    title = "Image tool"

    def __init__(self, data, *, parent=None):
        super().__init__(data, parent=parent)

        self._image_view = ImageView()
        self._image_view.roi1.sigRegionChangeFinished.connect(
            self.roiRegionChangedEvent)
        self._image_view.roi2.sigRegionChangeFinished.connect(
            self.roiRegionChangedEvent)

        self._roi1_ctrl = ROICtrlWidget(
            "ROI 1 ({})".format(ImageView.roi1_color))
        self._roi2_ctrl = ROICtrlWidget(
            "ROI 2 ({})".format(ImageView.roi2_color))
        self._roi1_ctrl.activate_cb.stateChanged.connect(
            self.toggleRoiActivationEvent)
        self._roi2_ctrl.activate_cb.stateChanged.connect(
            self.toggleRoiActivationEvent)
        self._roi1_ctrl.lock_aspect_cb.stateChanged.connect(
            self.lockAspectEvent
        )
        self._roi2_ctrl.lock_aspect_cb.stateChanged.connect(
            self.lockAspectEvent
        )
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

    def lockAspectEvent(self, state):
        sender = self.sender()
        if sender is self._roi1_ctrl.lock_aspect_cb:
            if state == QtCore.Qt.Checked:
                self._image_view.roi1.lockAspect()
            else:
                self._image_view.roi1.unLockAspect()
        elif sender is self._roi2_ctrl.lock_aspect_cb:
            if state == QtCore.Qt.Checked:
                self._image_view.roi2.lockAspect()
            else:
                self._image_view.roi2.unLockAspect()

    def _activate_roi1(self):
        self._roi1_ctrl.activate_cb.setChecked(True)

    def _activate_roi2(self):
        self._roi2_ctrl.activate_cb.setChecked(True)

    def roiRegionChangedEvent(self):
        sender = self.sender()
        if sender is self._image_view.roi1:
            self._roi1_ctrl.updateParameters(sender.size(), sender.pos())
        elif sender is self._image_view.roi2:
            self._roi2_ctrl.updateParameters(sender.size(), sender.pos())

    @QtCore.pyqtSlot(float, float, float, float)
    def onRoiRegionChanged(self, w, h, cx, cy):
        sender = self.sender()
        if sender is self._roi1_ctrl:
            roi = self._image_view.roi1
        else:
            roi = self._image_view.roi2

        # If 'update' == False, the state change will be remembered
        # but not processed and no signals will be emitted.
        roi.setSize((w, h), update=False)
        roi.setPos((cx, cy), update=False)
