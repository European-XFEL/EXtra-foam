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
from ..config import config


class ROICtrlWidget(QtGui.QGroupBox):
    """Widget for controlling of ROI."""
    # w, h, cx, cy
    roi_region_changed_sgn = QtCore.Signal(int, int, int, int)

    _pos_validator = QtGui.QIntValidator(-10000, 10000)
    _size_validator = QtGui.QIntValidator(0, 10000)

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

        self._line_edits = (self._width_le, self._height_le,
                            self._cx_le, self._cy_le)

        self.activate_cb = QtGui.QCheckBox("Activate")
        self.lock_cb = QtGui.QCheckBox("Lock")
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
        cb_layout.addWidget(self.lock_cb)
        cb_layout.addWidget(self.lock_aspect_cb)

        layout = QtGui.QVBoxLayout()

        layout.addLayout(cb_layout)
        layout.addLayout(le_layout)

        self.setLayout(layout)

    def updateParameters(self, w, h, cx, cy):
        self._width_le.setText(str(w))
        self._height_le.setText(str(h))
        self._cx_le.setText(str(cx))
        self._cy_le.setText(str(cy))

    def roiRegionChangedEvent(self):
        w = self._width_le.text()
        h = self._height_le.text()
        cx = self._cx_le.text()
        cy = self._cy_le.text()
        self.roi_region_changed_sgn.emit(w, h, cx, cy)

    def disableLockEdit(self):
        for w in self._line_edits:
            w.setDisabled(True)
        self.lock_aspect_cb.setDisabled(True)

    def enableLockEdit(self):
        for w in self._line_edits:
            w.setDisabled(False)
        self.lock_aspect_cb.setDisabled(False)

    def disableAllEdit(self):
        self.disableLockEdit()
        self.lock_cb.setDisabled(True)

    def enableAllEdit(self):
        self.lock_cb.setDisabled(False)
        self.enableLockEdit()


class MaskCtrlWidget(QtGui.QGroupBox):
    """Widget for masking image."""

    threshold_mask_sgn = QtCore.pyqtSignal(float, float)

    _double_validator = QtGui.QDoubleValidator()

    def __init__(self, title, *, parent=None):
        """"""
        super().__init__(title, parent=parent)

        self._min_pixel_le = QtGui.QLineEdit(str(config["MASK_RANGE"][0]))
        self._min_pixel_le.setValidator(self._double_validator)
        self._max_pixel_le = QtGui.QLineEdit(str(config["MASK_RANGE"][1]))
        self._max_pixel_le.setValidator(self._double_validator)
        self._min_pixel_le.returnPressed.connect(self.thresholdMaskChangedEvent)
        self._max_pixel_le.returnPressed.connect(self.thresholdMaskChangedEvent)

        self.initUI()

    def initUI(self):
        threshold_layout = QtGui.QHBoxLayout()
        threshold_layout.addWidget(QtGui.QLabel("Min.: "))
        threshold_layout.addWidget(self._min_pixel_le)
        threshold_layout.addWidget(QtGui.QLabel("Max.: "))
        threshold_layout.addWidget(self._max_pixel_le)

        layout = QtGui.QVBoxLayout()

        layout.addLayout(threshold_layout)
        self.setLayout(layout)

    def thresholdMaskChangedEvent(self):
        self.threshold_mask_sgn.emit(float(self._min_pixel_le.text()),
                                     float(self._max_pixel_le.text()))


@SingletonWindow
class ImageToolWindow(AbstractWindow):
    """ImageToolWindow class.

    This tool provides selecting of ROI and image masking.
    """
    title = "Image tool"

    # w, h, cx, cy
    roi1_region_changed_sgn = QtCore.Signal(int, int, int, int)
    roi2_region_changed_sgn = QtCore.Signal(int, int, int, int)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        parent = self.parent()

        self._image_view = ImageView(lock_roi=False)
        self._image_view.roi1.sigRegionChangeFinished.connect(
            self.roiRegionChangedEvent)
        self._image_view.roi2.sigRegionChangeFinished.connect(
            self.roiRegionChangedEvent)

        self._clear_roi_hist_btn = QtGui.QPushButton("Clear ROI history")
        self._clear_roi_hist_btn.clicked.connect(
            parent._proc_worker.onRoiHistCleared)
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
        self._roi1_ctrl.lock_cb.stateChanged.connect(self.lockEvent)
        self._roi2_ctrl.lock_cb.stateChanged.connect(self.lockEvent)
        self._roi1_ctrl.roi_region_changed_sgn.connect(self.onRoiRegionChanged)
        self._roi2_ctrl.roi_region_changed_sgn.connect(self.onRoiRegionChanged)

        self.roi1_region_changed_sgn.connect(parent._proc_worker.onRoi1Changed)
        self.roi2_region_changed_sgn.connect(parent._proc_worker.onRoi2Changed)

        self._image_view.roi1.setSize(self._image_view.roi1.size())
        self._image_view.roi2.setSize(self._image_view.roi2.size())

        self._mask_panel = MaskCtrlWidget("Masking tool")
        self._mask_panel.threshold_mask_sgn.connect(
            parent._proc_worker.onThresholdMaskChanged)

        self._update_image_btn = QtGui.QPushButton("Update image")
        self._update_image_btn.clicked.connect(self.updateImage)

        self.initUI()
        self.resize(800, 800)
        self.updateImage()

    def initUI(self):
        """Override."""
        tool_layout = QtGui.QVBoxLayout()
        tool_layout.addWidget(self._clear_roi_hist_btn)
        tool_layout.addWidget(self._roi1_ctrl)
        tool_layout.addWidget(self._roi2_ctrl)

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
        sender = self.sender().parent()
        if sender is self._roi1_ctrl:
            roi = self._image_view.roi1
        elif sender is self._roi2_ctrl:
            roi = self._image_view.roi2
        else:
            return

        if state == QtCore.Qt.Checked:
            roi.show()
            sender.enableAllEdit()
        else:
            roi.hide()
            sender.disableAllEdit()

    def lockAspectEvent(self, state):
        sender = self.sender().parent()
        if sender is self._roi1_ctrl:
            roi = self._image_view.roi1
        elif sender is self._roi2_ctrl:
            roi = self._image_view.roi2
        else:
            return

        if state == QtCore.Qt.Checked:
            roi.lockAspect()
        else:
            roi.unLockAspect()

    def lockEvent(self, state):
        sender = self.sender().parent()
        if sender is self._roi1_ctrl:
            roi = self._image_view.roi1
        elif sender is self._roi2_ctrl:
            roi = self._image_view.roi2
        else:
            return

        if state == QtCore.Qt.Checked:
            roi.lock()
            sender.disableLockEdit()
        else:
            roi.unLock()
            sender.enableLockEdit()

    def _activate_roi1(self):
        self._roi1_ctrl.activate_cb.setChecked(True)

    def _activate_roi2(self):
        self._roi2_ctrl.activate_cb.setChecked(True)

    def roiRegionChangedEvent(self):
        sender = self.sender()
        w, h = [int(v) for v in sender.size()]
        cx, cy = [int(v) for v in sender.pos()]
        if sender is self._image_view.roi1:
            self._roi1_ctrl.updateParameters(w, h, cx, cy)
            # inform widgets outside this window
            self.roi1_region_changed_sgn.emit(w, h, cx, cy)
        elif sender is self._image_view.roi2:
            self._roi2_ctrl.updateParameters(w, h, cx, cy)
            self.roi2_region_changed_sgn.emit(w, h, cx, cy)

    @QtCore.pyqtSlot(int, int, int, int)
    def onRoiRegionChanged(self, w, h, cx, cy):
        """Connect to the signal from ROICtrlWidget."""
        sender = self.sender()
        if sender is self._roi1_ctrl:
            roi = self._image_view.roi1
            # a relay signal for widgets outside this window
            self.roi1_region_changed_sgn.emit(w, h, cx, cy)
        else:
            roi = self._image_view.roi2
            self.roi2_region_changed_sgn.emit(w, h, cx, cy)

        # If 'update' == False, the state change will be remembered
        # but not processed and no signals will be emitted.
        roi.setSize((w, h), update=False)
        roi.setPos((cx, cy), update=False)
