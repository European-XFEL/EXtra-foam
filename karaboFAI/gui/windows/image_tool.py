"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

ImageToolWindow.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import os.path as osp
from collections import OrderedDict
import functools

from ..pyqtgraph import QtCore, QtGui

from .base_window import AbstractWindow, SingletonWindow
from ..mediator import Mediator
from ..ctrl_widgets import ImageCtrlWidget, MaskCtrlWidget, RoiCtrlWidget
from ..plot_widgets import ImageAnalysis
from ...config import config, RoiValueType, ImageMaskChange, ImageNormalizer


@SingletonWindow
class ImageToolWindow(AbstractWindow):
    """ImageToolWindow class.

    This tool provides selecting of ROI and image masking.
    """
    title = "Image tool"

    _available_roi_value_types = OrderedDict({
        "sum": RoiValueType.SUM,
        "mean": RoiValueType.MEAN,
    })

    _available_img_normalizers = OrderedDict({
        "None": ImageNormalizer.NONE,
        "ROI_SUM": ImageNormalizer.ROI_SUM
    })

    roi_value_type_sgn = QtCore.pyqtSignal(object)

    _root_dir = osp.dirname(osp.abspath(__file__))

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        mediator = Mediator()

        self._image_view = ImageAnalysis(
            lock_roi=False, hide_axis=False, parent=self)

        self._clear_roi_hist_btn = QtGui.QPushButton("Clear history")
        self._clear_roi_hist_btn.clicked.connect(mediator.onRoiHistClear)

        self._roi_displayed_range_le = QtGui.QLineEdit(str(600))
        validator = QtGui.QIntValidator()
        validator.setBottom(1)
        self._roi_displayed_range_le.setValidator(validator)
        self._roi_displayed_range_le.editingFinished.connect(
            mediator.onRoiDisplayedRangeChange)

        self._roi_value_type_cb = QtGui.QComboBox()
        for v in self._available_roi_value_types:
            self._roi_value_type_cb.addItem(v)
        self._roi_value_type_cb.currentTextChanged.connect(
            lambda x: self.roi_value_type_sgn.emit(
                self._available_roi_value_types[x]))
        self.roi_value_type_sgn.connect(mediator.onRoiValueTypeChange)
        self._roi_value_type_cb.currentTextChanged.emit(
            self._roi_value_type_cb.currentText())

        self._bkg_le = QtGui.QLineEdit(str(0.0))
        self._bkg_le.setValidator(QtGui.QDoubleValidator())
        self._bkg_le.editingFinished.connect(self._image_view.onBkgChange)

        self._normalizer_cb = QtGui.QComboBox()
        for v in self._available_img_normalizers:
            self._normalizer_cb.addItem(v)

        self._set_ref_btn = QtGui.QPushButton("Set reference image")
        self._set_ref_btn.clicked.connect(self._image_view.setImageRef)

        self._roi1_ctrl = RoiCtrlWidget(
            self._image_view.roi1,
            title="ROI 1 ({})".format(config['ROI_COLORS'][0]))
        self._roi2_ctrl = RoiCtrlWidget(
            self._image_view.roi2,
            title="ROI 2 ({})".format(config['ROI_COLORS'][1]))

        self._roi1_ctrl.roi_region_change_sgn.connect(mediator.onRoi1Change)
        self._roi2_ctrl.roi_region_change_sgn.connect(mediator.onRoi2Change)

        #
        # image tool bar
        #

        self._tool_bar_image = self.addToolBar("image")

        self._update_image_at = self._addAction(
            self._tool_bar_image, "Update image", "sync.png")
        self._update_image_at.triggered.connect(self.updateImage)

        self._image_ctrl = ImageCtrlWidget()
        self._image_ctrl.moving_avg_window_sgn.connect(
            self._image_view.onMovingAverageWindowChange)
        self._image_ctrl_at = QtGui.QWidgetAction(self._tool_bar_image)
        self._image_ctrl_at.setDefaultWidget(self._image_ctrl)
        self._tool_bar_image.addAction(self._image_ctrl_at)

        #
        # mask tool bar
        #

        self._tool_bar_mask = self.addToolBar("mask")

        mask_at = self._addAction(self._tool_bar_mask, "Mask", "mask.png")
        mask_at.setCheckable(True)
        # Note: the sequence of the following two 'connect'
        mask_at.toggled.connect(self._exclude_actions)
        mask_at.toggled.connect(functools.partial(
            self._image_view.onDrawToggled, ImageMaskChange.MASK))

        unmask_at = self._addAction(
            self._tool_bar_mask, "Unmask", "un_mask.png")
        unmask_at.setCheckable(True)
        # Note: the sequence of the following two 'connect'
        unmask_at.toggled.connect(self._exclude_actions)
        unmask_at.toggled.connect(functools.partial(
            self._image_view.onDrawToggled, ImageMaskChange.UNMASK))

        clear_mask_at = self._addAction(
            self._tool_bar_mask, "Trash mask", "trash_mask.png")
        clear_mask_at.triggered.connect(self._image_view.onClearMask)

        save_img_mask_at = self._addAction(
            self._tool_bar_mask, "Save image mask", "save_mask.png")
        save_img_mask_at.triggered.connect(self._image_view.saveImageMask)

        load_img_mask_at = self._addAction(
            self._tool_bar_mask, "Load image mask", "load_mask.png")
        load_img_mask_at.triggered.connect(self._image_view.loadImageMask)

        self._mask_ctrl = MaskCtrlWidget()
        self._mask_ctrl.threshold_mask_sgn.connect(
            self._image_view.onThresholdMaskChange)

        mask_widget = QtGui.QWidgetAction(self._tool_bar_mask)
        mask_widget.setDefaultWidget(self._mask_ctrl)
        self._tool_bar_mask.addAction(mask_widget)

        self.addToolBarBreak()

        #
        # crop tool bar
        #

        self._tool_bar_crop = self.addToolBar("crop")

        crop_at = self._addAction(
            self._tool_bar_crop, "Crop", "crop_selection.png")
        crop_at.setCheckable(True)
        crop_at.toggled.connect(self._exclude_actions)
        crop_at.toggled.connect(self._image_view.onCropToggle)

        crop_to_selection_at = self._addAction(
            self._tool_bar_crop, "Crop to selection", "crop.png")
        crop_to_selection_at.triggered.connect(
            self._image_view.onCropConfirmed)
        crop_to_selection_at.triggered.connect(functools.partial(
            crop_at.setChecked, False))

        restore_image_at = self._addAction(
            self._tool_bar_crop, "Restore image", "restore.png")
        restore_image_at.triggered.connect(self._image_view.onRestoreImage)

        self._exclusive_actions = {mask_at, unmask_at, crop_at}

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
        roi_ctrl_layout.addWidget(QtGui.QLabel("Normalized by: "), 1, 2)
        roi_ctrl_layout.addWidget(self._normalizer_cb, 1, 3)
        roi_ctrl_layout.addWidget(self._set_ref_btn, 1, 4)

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

        self.updateImage(auto_range=True, auto_levels=True)

    def updateImage(self, **kwargs):
        """Update the current image.

        :param kwargs: forward to ImageView.setImage().

        It is only used for updating the image manually.
        """
        data = self._data.get()
        if data.image is None:
            return

        self._image_view.setImageData(data.image, **kwargs)

    @QtCore.pyqtSlot(bool)
    def _exclude_actions(self, checked):
        if checked:
            for at in self._exclusive_actions:
                if at != self.sender():
                    at.setChecked(False)

    def _addAction(self, tool_bar, description, filename):
        icon = QtGui.QIcon(osp.join(self._root_dir, "../icons/" + filename))
        action = QtGui.QAction(icon, description, self)
        tool_bar.addAction(action)
        return action
