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
from ...config import config, RoiValueType, ImageMaskChange


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

    roi_value_type_sgn = QtCore.pyqtSignal(object)

    _root_dir = osp.dirname(osp.abspath(__file__))

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        mediator = Mediator()

        self._image_view = ImageAnalysis(
            lock_roi=False, hide_axis=False, parent=self)
        self._image_view.crop_area_sgn.connect(mediator.onCropAreaChange)

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
        self._bkg_le.editingFinished.connect(mediator.onBkgChange)
        self._bkg_le.editingFinished.connect(self._image_view.onBkgChange)

        self._lock_bkg_cb = QtGui.QCheckBox("Lock background")
        self._lock_bkg_cb.stateChanged.connect(
            lambda x: self._bkg_le.setEnabled(x != QtCore.Qt.Checked))

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

        icon = QtGui.QIcon(osp.join(self._root_dir, "../icons/update_image.png"))
        self._update_image_at = QtGui.QAction(icon, "Update image", self)
        self._tool_bar_image.addAction(self._update_image_at)
        self._update_image_at.triggered.connect(self.updateImage)

        self._image_ctrl = ImageCtrlWidget()
        self._image_ctrl.moving_avg_window_sgn.connect(
            mediator.onMovingAvgWindowChange)
        self._image_ctrl_at = QtGui.QWidgetAction(self._tool_bar_image)
        self._image_ctrl_at.setDefaultWidget(self._image_ctrl)
        self._tool_bar_image.addAction(self._image_ctrl_at)

        #
        # mask tool bar
        #
        self._tool_bar_mask = self.addToolBar("mask")

        icon = QtGui.QIcon(osp.join(self._root_dir, "../icons/mask.png"))
        self._mask_at = QtGui.QAction(icon, "Mask", self)
        self._mask_at.setCheckable(True)
        self._tool_bar_mask.addAction(self._mask_at)
        self._mask_at.triggered.connect(functools.partial(
            self._image_view.onDrawToggled, ImageMaskChange.MASK))
        self._mask_at.triggered.connect(self._exclusive_mask_unmask)

        icon = QtGui.QIcon(osp.join(self._root_dir, "../icons/un_mask.png"))
        self._unmask_at = QtGui.QAction(icon, "Unmask", self)
        self._unmask_at.setCheckable(True)
        self._tool_bar_mask.addAction(self._unmask_at)
        self._unmask_at.triggered.connect(functools.partial(
            self._image_view.onDrawToggled, ImageMaskChange.UNMASK))
        self._unmask_at.triggered.connect(self._exclusive_mask_unmask)

        icon = QtGui.QIcon(osp.join(self._root_dir, "../icons/trash_mask.png"))
        self._clear_mask_at = QtGui.QAction(icon, "Trash mask", self)
        self._tool_bar_mask.addAction(self._clear_mask_at)
        self._clear_mask_at.triggered.connect(self._image_view.clearMask)

        self._mask_ctrl = MaskCtrlWidget()
        self._mask_ctrl.threshold_mask_sgn.connect(
            mediator.onThresholdMaskChange)
        self._mask_ctrl.threshold_mask_sgn.connect(
            self._image_view.onImageMaskChange)
        self._masking_at = QtGui.QWidgetAction(self._tool_bar_mask)
        self._masking_at.setDefaultWidget(self._mask_ctrl)
        self._tool_bar_mask.addAction(self._masking_at)

        self.addToolBarBreak()

        #
        # crop tool bar
        #
        self._tool_bar_crop = self.addToolBar("crop")

        #
        icon = QtGui.QIcon(osp.join(self._root_dir,
                                    "../icons/crop_selection.png"))
        self._crop_at = QtGui.QAction(icon, "Crop", self)
        self._tool_bar_crop.addAction(self._crop_at)
        self._crop_at.triggered.connect(self._image_view.onCropToggle)

        #
        icon = QtGui.QIcon(osp.join(self._root_dir, "../icons/crop.png"))
        self._crop_to_selection_at = QtGui.QAction(
            icon, "Crop to selection", self)
        self._tool_bar_crop.addAction(self._crop_to_selection_at)
        self._crop_to_selection_at.triggered.connect(
            self._image_view.onCropConfirmed)

        #
        icon = QtGui.QIcon(osp.join(self._root_dir, "../icons/restore.png"))
        self._restore_image_at = QtGui.QAction(icon, "Restore image", self)
        self._tool_bar_crop.addAction(self._restore_image_at)
        self._restore_image_at.triggered.connect(
            self._image_view.onRestoreImage)

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

        It is only used for updating the image manually.
        """
        data = self._data.get()
        if data.image is None:
            return

        self._image_view.setImageData(data.image)

    @QtCore.pyqtSlot()
    def _exclusive_mask_unmask(self):
        if self.sender() is self._mask_at:
            self._unmask_at.setChecked(False)
        elif self.sender() is self._unmask_at:
            self._mask_at.setChecked(False)
