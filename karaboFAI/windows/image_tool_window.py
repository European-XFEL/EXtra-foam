"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

ImageToolWindow.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import os
from collections import OrderedDict

from .base_window import AbstractWindow, SingletonWindow

from ..widgets.pyqtgraph import QtCore, QtGui
from ..data_processing import RoiValueType
from ..widgets import ImageView
from ..config import config


class RoiCtrlWidget(QtGui.QGroupBox):
    """Widget for controlling of an ROI."""

    GROUP_BOX_STYLE_SHEET = 'QGroupBox:title {' \
                            'border: 0px;' \
                            'subcontrol-origin: margin;' \
                            'subcontrol-position: top left;' \
                            'padding-left: 5px;' \
                            'padding-top: 5px;' \
                            'margin-top: 0.0em;}'

    # activated, w, h, px, py
    roi_region_change_sgn = QtCore.Signal(bool, int, int, int, int)

    _pos_validator = QtGui.QIntValidator(-10000, 10000)
    _size_validator = QtGui.QIntValidator(0, 10000)

    def __init__(self, roi, *, title="ROI control", parent=None):
        """Initialization.

        :param RectROI roi: RectROI object.
        """
        super().__init__(title, parent=parent)
        self.setStyleSheet(self.GROUP_BOX_STYLE_SHEET)

        self._roi = roi

        self._width_le = QtGui.QLineEdit()
        self._width_le.setValidator(self._size_validator)
        self._height_le = QtGui.QLineEdit()
        self._height_le.setValidator(self._size_validator)
        self._px_le = QtGui.QLineEdit()
        self._px_le.setValidator(self._pos_validator)
        self._py_le = QtGui.QLineEdit()
        self._py_le.setValidator(self._pos_validator)
        self._width_le.editingFinished.connect(self.onRoiRegionChanged)
        self._height_le.editingFinished.connect(self.onRoiRegionChanged)
        self._px_le.editingFinished.connect(self.onRoiRegionChanged)
        self._py_le.editingFinished.connect(self.onRoiRegionChanged)

        self._line_edits = (self._width_le, self._height_le,
                            self._px_le, self._py_le)

        self.activate_cb = QtGui.QCheckBox("Activate")
        self.lock_cb = QtGui.QCheckBox("Lock")
        self.lock_aspect_cb = QtGui.QCheckBox("Lock aspect ratio")

        self.initUI()

        roi.sigRegionChangeFinished.connect(self.onRoiRegionChangeFinished)
        roi.sigRegionChangeFinished.emit(roi)  # fill the QLineEdit(s)
        self.activate_cb.stateChanged.connect(self.onToggleRoiActivation)
        self.lock_cb.stateChanged.connect(self.onLock)
        self.lock_aspect_cb.stateChanged.connect(self.onLockAspect)

    def initUI(self):
        le_layout = QtGui.QHBoxLayout()
        le_layout.addWidget(QtGui.QLabel("w: "))
        le_layout.addWidget(self._width_le)
        le_layout.addWidget(QtGui.QLabel("h: "))
        le_layout.addWidget(self._height_le)
        le_layout.addWidget(QtGui.QLabel("x0: "))
        le_layout.addWidget(self._px_le)
        le_layout.addWidget(QtGui.QLabel("y0: "))
        le_layout.addWidget(self._py_le)

        cb_layout = QtGui.QHBoxLayout()
        cb_layout.addWidget(self.activate_cb)
        cb_layout.addWidget(self.lock_cb)
        cb_layout.addWidget(self.lock_aspect_cb)

        layout = QtGui.QVBoxLayout()

        layout.addLayout(cb_layout)
        layout.addLayout(le_layout)

        self.setLayout(layout)
        # left, top, right, bottom
        self.layout().setContentsMargins(2, 1, 2, 1)

    @QtCore.pyqtSlot(object)
    def onRoiRegionChangeFinished(self, roi):
        """Connect to the signal from an ROI object."""
        w, h = [int(v) for v in roi.size()]
        px, py = [int(v) for v in roi.pos()]
        self.updateParameters(w, h, px, py)
        # inform widgets outside this window
        self.roi_region_change_sgn.emit(True, w, h, px, py)

    @QtCore.pyqtSlot(int)
    def onToggleRoiActivation(self, state):
        if state == QtCore.Qt.Checked:
            self._roi.show()
            self.enableAllEdit()
            self.roi_region_change_sgn.emit(
                True, *self._roi.size(), *self._roi.pos())
        else:
            self._roi.hide()
            self.disableAllEdit()
            self.roi_region_change_sgn.emit(
                False, *self._roi.size(), *self._roi.pos())

    @QtCore.pyqtSlot(int)
    def onLock(self, state):
        if state == QtCore.Qt.Checked:
            self._roi.lock()
            self.disableLockEdit()
        else:
            self._roi.unLock()
            self.enableLockEdit()

    @QtCore.pyqtSlot(int)
    def onLockAspect(self, state):
        if state == QtCore.Qt.Checked:
            self._roi.lockAspect()
        else:
            self._roi.unLockAspect()

    def updateParameters(self, w, h, px, py):
        self._width_le.setText(str(w))
        self._height_le.setText(str(h))
        self._px_le.setText(str(px))
        self._py_le.setText(str(py))

    @QtCore.pyqtSlot()
    def onRoiRegionChanged(self):
        w = int(self._width_le.text())
        h = int(self._height_le.text())
        px = int(self._px_le.text())
        py = int(self._py_le.text())

        # If 'update' == False, the state change will be remembered
        # but not processed and no signals will be emitted.
        self._roi.setSize((w, h), update=False)
        self._roi.setPos((px, py), update=False)

        self.roi_region_change_sgn.emit(True, w, h, px, py)

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


class MaskCtrlWidget(QtGui.QWidget):
    """Widget inside the action bar for masking image."""

    threshold_mask_sgn = QtCore.pyqtSignal(float, float)

    _double_validator = QtGui.QDoubleValidator()

    def __init__(self, parent=None):
        """Initialization"""
        super().__init__(parent=parent)

        self._min_pixel_le = QtGui.QLineEdit(str(config["MASK_RANGE"][0]))
        self._min_pixel_le.setValidator(self._double_validator)
        self._max_pixel_le = QtGui.QLineEdit(str(config["MASK_RANGE"][1]))
        self._max_pixel_le.setValidator(self._double_validator)
        self._min_pixel_le.returnPressed.connect(
            self.thresholdMaskChangedEvent)
        self._max_pixel_le.returnPressed.connect(
            self.thresholdMaskChangedEvent)

        self.initUI()

        self.setFixedSize(self.minimumSizeHint())

    def initUI(self):
        layout = QtGui.QHBoxLayout()
        layout.addWidget(QtGui.QLabel("Min. val: "))
        layout.addWidget(self._min_pixel_le)
        layout.addWidget(QtGui.QLabel("Max. val: "))
        layout.addWidget(self._max_pixel_le)

        self.setLayout(layout)
        self.layout().setContentsMargins(2, 1, 2, 1)

    def thresholdMaskChangedEvent(self):
        self.threshold_mask_sgn.emit(float(self._min_pixel_le.text()),
                                     float(self._max_pixel_le.text()))


@SingletonWindow
class ImageToolWindow(AbstractWindow):
    """ImageToolWindow class.

    This tool provides selecting of ROI and image masking.
    """
    title = "Image tool"

    _available_roi_value_types = OrderedDict({
        "integration": RoiValueType.INTEGRATION,
        "mean": RoiValueType.MEAN,
    })

    roi_value_type_sgn = QtCore.pyqtSignal(object)

    _root_dir = os.path.dirname(os.path.abspath(__file__))

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._image_view = ImageView(
            lock_roi=False, hide_axis=False, enable_hover=True)
        self._image_view.crop_area_change_sgn.connect(
            self._mediator.onCropAreaChange)

        self._clear_roi_hist_btn = QtGui.QPushButton("Clear history")
        self._clear_roi_hist_btn.clicked.connect(
            self._mediator.onRoiHistClear)

        self._roi_displayed_range_le = QtGui.QLineEdit(str(600))
        validator = QtGui.QIntValidator()
        validator.setBottom(1)
        self._roi_displayed_range_le.setValidator(validator)
        self._roi_displayed_range_le.editingFinished.connect(
            self._mediator.onRoiDisplayedRangeChange)

        self._roi_value_type_cb = QtGui.QComboBox()
        for v in self._available_roi_value_types:
            self._roi_value_type_cb.addItem(v)
        self._roi_value_type_cb.currentTextChanged.connect(
            lambda x: self.roi_value_type_sgn.emit(
                self._available_roi_value_types[x]))
        self.roi_value_type_sgn.connect(self._mediator.onRoiValueTypeChange)
        self._roi_value_type_cb.currentTextChanged.emit(
            self._roi_value_type_cb.currentText())

        self._bkg_le = QtGui.QLineEdit(str(0))
        self._bkg_le.setValidator(QtGui.QIntValidator())
        self._bkg_le.editingFinished.connect(self._mediator.onBkgChange)

        self._lock_bkg_cb = QtGui.QCheckBox("Lock background")
        self._lock_bkg_cb.stateChanged.connect(
            lambda x: self._bkg_le.setEnabled(x != QtCore.Qt.Checked))

        self._roi1_ctrl = RoiCtrlWidget(
            self._image_view.roi1,
            title="ROI 1 ({})".format(config['ROI_COLORS'][0]))
        self._roi2_ctrl = RoiCtrlWidget(
            self._image_view.roi2,
            title="ROI 2 ({})".format(config['ROI_COLORS'][1]))

        self._roi1_ctrl.roi_region_change_sgn.connect(
            self._mediator.onRoi1Change)
        self._roi2_ctrl.roi_region_change_sgn.connect(
            self._mediator.onRoi2Change)

        #
        # tool bar
        #
        self._tool_bar = self.addToolBar("Control")

        self._crop_at = QtGui.QAction(
            QtGui.QIcon(os.path.join(self._root_dir,
                                     "icons/crop_selection.png")),
            "Crop",
            self)
        self._tool_bar.addAction(self._crop_at)
        self._crop_at.triggered.connect(self._image_view.onCropToggle)

        self._crop_to_selection_at = QtGui.QAction(
            QtGui.QIcon(os.path.join(self._root_dir, "icons/crop.png")),
            "Crop to selection",
            self)
        self._tool_bar.addAction(self._crop_to_selection_at)
        self._crop_to_selection_at.triggered.connect(
            self._image_view.onCropConfirmed)

        self._restore_image_at = QtGui.QAction(
            QtGui.QIcon(os.path.join(self._root_dir, "icons/restore.png")),
            "Restore image",
            self)
        self._tool_bar.addAction(self._restore_image_at)
        self._restore_image_at.triggered.connect(
            self._image_view.onRestoreImage)

        self._mask_ctrl = MaskCtrlWidget()
        self._mask_ctrl.threshold_mask_sgn.connect(
            self._mediator.onThresholdMaskChange)
        self._mask_ctrl.threshold_mask_sgn.connect(
            self._image_view.onImageMaskChange)
        self._mask_at = QtGui.QWidgetAction(self._tool_bar)
        self._mask_at.setDefaultWidget(self._mask_ctrl)
        self._tool_bar.addAction(self._mask_at)

        self._update_image_btn = QtGui.QPushButton("Update image")
        self._update_image_btn.setStyleSheet('background-color: green')
        self._update_image_btn.clicked.connect(self.updateImage)
        self._update_image_at = QtGui.QWidgetAction(self._tool_bar)
        self._update_image_at.setDefaultWidget(self._update_image_btn)
        self._tool_bar.addAction(self._update_image_at)

        self.initUI()
        self.resize(800, 800)
        self.update()

    def initUI(self):
        """Override."""
        roi_ctrl_layout = QtGui.QGridLayout()
        roi_ctrl_layout.addWidget(QtGui.QLabel("ROI value: "), 0, 0)
        roi_ctrl_layout.addWidget(self._roi_value_type_cb, 0, 1)
        roi_ctrl_layout.addWidget(QtGui.QLabel("Displayed range: "), 0, 2)
        roi_ctrl_layout.addWidget(self._roi_displayed_range_le, 0, 3)
        roi_ctrl_layout.addWidget(self._clear_roi_hist_btn, 0, 4)
        roi_ctrl_layout.addWidget(QtGui.QLabel("Background level: "), 1, 0)
        roi_ctrl_layout.addWidget(self._bkg_le, 1, 1)
        roi_ctrl_layout.addWidget(self._lock_bkg_cb, 1, 2)

        tool_layout = QtGui.QVBoxLayout()
        tool_layout.addLayout(roi_ctrl_layout)
        tool_layout.addWidget(self._roi1_ctrl)
        tool_layout.addWidget(self._roi2_ctrl)

        layout = QtGui.QVBoxLayout()
        layout.addWidget(self._image_view)
        layout.addLayout(tool_layout)

        self._cw.setLayout(layout)
        self.layout().setContentsMargins(0, 0, 0, 0)

    def update(self):
        """Update widgets.

        This method is called by the main GUI.
        """
        # Always automatically get an image
        if self._image_view.image is not None:
            return

        self.updateImage()

    def updateImage(self):
        """Update the current image.

        It is used for updating the image manually.
        """
        data = self._data.get()
        if data.image is None:
            return

        self._image_view.setImageData(data.image)
