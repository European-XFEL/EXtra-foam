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
    class ImageProcWidget(QtGui.QGroupBox):

        _available_img_normalizers = OrderedDict({
            "None": ImageNormalizer.NONE,
            "ROI_SUM": ImageNormalizer.ROI_SUM
        })

        _pos_validator = QtGui.QIntValidator(-10000, 10000)
        _size_validator = QtGui.QIntValidator(0, 10000)

        image_normalizer_change_sgn = QtCore.pyqtSignal(int)

        def __init__(self, parent=None):
            super().__init__(parent)

            self.bkg_le = QtGui.QLineEdit(str(0.0))
            self.bkg_le.setValidator(QtGui.QDoubleValidator())

            self._normalizer_cb = QtGui.QComboBox()
            for v in self._available_img_normalizers:
                self._normalizer_cb.addItem(v)
            self._normalizer_cb.currentTextChanged.emit(
                self._normalizer_cb.currentText())
            self._normalizer_cb.currentTextChanged.connect(
                lambda x: self.image_normalizer_change_sgn.emit(
                    self._available_img_normalizers[x]))

            self._width_le = QtGui.QLineEdit()
            self._width_le.setValidator(self._size_validator)
            self._height_le = QtGui.QLineEdit()
            self._height_le.setValidator(self._size_validator)
            self._px_le = QtGui.QLineEdit()
            self._px_le.setValidator(self._pos_validator)
            self._py_le = QtGui.QLineEdit()
            self._py_le.setValidator(self._pos_validator)

            self.set_ref_btn = QtGui.QPushButton("Set reference image")

            self.initUI()

        def initUI(self):
            """Override."""
            AR = QtCore.Qt.AlignRight

            layout = QtGui.QGridLayout()
            layout.addWidget(QtGui.QLabel("Subtract bkg: "), 0, 0, 1, 2, AR)
            layout.addWidget(self.bkg_le, 0, 2, 1, 2)
            layout.addWidget(QtGui.QLabel("Normalized by: "), 1, 0, 1, 2, AR)
            layout.addWidget(self._normalizer_cb, 1, 2, 1, 2)
            layout.addWidget(QtGui.QLabel("w: "), 2, 0)
            layout.addWidget(self._width_le, 2, 1)
            layout.addWidget(QtGui.QLabel("h: "), 2, 2)
            layout.addWidget(self._height_le, 2, 3)
            layout.addWidget(QtGui.QLabel("x: "), 3, 0)
            layout.addWidget(self._px_le, 3, 1)
            layout.addWidget(QtGui.QLabel("y: "), 3, 2)
            layout.addWidget(self._py_le, 3, 3)
            layout.addWidget(self.set_ref_btn, 4, 0, 1, 4)
            self.setLayout(layout)

    title = "Image tool"

    _available_roi_value_types = OrderedDict({
        "sum": RoiValueType.SUM,
        "mean": RoiValueType.MEAN,
    })

    roi_value_type_sgn = QtCore.pyqtSignal(object)

    _root_dir = osp.dirname(osp.abspath(__file__))

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        mediator = Mediator()

        self._image_view = ImageAnalysis(hide_axis=False, parent=self)

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

        self._image_proc_widget = self.ImageProcWidget(parent=self)

        self._image_proc_widget.bkg_le.editingFinished.connect(
            self._image_view.onBkgChange)
        self._image_proc_widget.set_ref_btn.clicked.connect(
            self._image_view.setImageRef)
        self._image_proc_widget.image_normalizer_change_sgn.connect(
            self._image_view.onImageNormalizerChange)

        self._roi_ctrls = []
        roi_colors = config['ROI_COLORS']
        for i, color in enumerate(roi_colors, 1):
            widget = RoiCtrlWidget(getattr(self._image_view, f"roi{i}"))
            self._roi_ctrls.append(widget)
            widget.roi_region_change_sgn.connect(mediator.onRoiChange)

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
        for i, roi_ctrl in enumerate(self._roi_ctrls, 1):
            roi_ctrl_layout.addWidget(roi_ctrl, i, 0, 1, 5)

        layout = QtGui.QGridLayout()
        layout.addWidget(self._image_view, 0, 0, 1, 3)
        layout.addLayout(roi_ctrl_layout, 1, 0, 1, 2)
        layout.addWidget(self._image_proc_widget, 1, 2, 1, 1)

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
