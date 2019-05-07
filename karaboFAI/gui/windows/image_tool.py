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
from ..plot_widgets import ImageAnalysis
from ...config import config, RoiFom, ImageMaskChange


class _RoiCtrlWidgetBase(QtGui.QWidget):
    """Base class for RoiCtrlWidget.

    Implemented four QLineEdits (w, h, x, y) and their connection to the
    corresponding ROI.
    """
    # rank, activated, w, h, px, py
    roi_region_change_sgn = QtCore.Signal(int, bool, int, int, int, int)

    _pos_validator = QtGui.QIntValidator(-10000, 10000)
    _size_validator = QtGui.QIntValidator(0, 10000)

    def __init__(self, roi, *, parent=None):
        super().__init__(parent=parent)
        self._roi = roi

        self._width_le = QtGui.QLineEdit()
        self._width_le.setValidator(self._size_validator)
        self._height_le = QtGui.QLineEdit()
        self._height_le.setValidator(self._size_validator)
        self._px_le = QtGui.QLineEdit()
        self._px_le.setValidator(self._pos_validator)
        self._py_le = QtGui.QLineEdit()
        self._py_le.setValidator(self._pos_validator)
        self._width_le.editingFinished.connect(self.onRoiRegionEdited)
        self._height_le.editingFinished.connect(self.onRoiRegionEdited)
        self._px_le.editingFinished.connect(self.onRoiRegionEdited)
        self._py_le.editingFinished.connect(self.onRoiRegionEdited)

        self._line_edits = (self._width_le, self._height_le,
                            self._px_le, self._py_le)

        roi.sigRegionChangeFinished.connect(self.onRoiRegionChangeFinished)
        roi.sigRegionChangeFinished.emit(roi)  # fill the QLineEdit(s)

    @QtCore.pyqtSlot()
    def onRoiRegionEdited(self):
        w = int(self._width_le.text())
        h = int(self._height_le.text())
        px = int(self._px_le.text())
        py = int(self._py_le.text())

        # If 'update' == False, the state change will be remembered
        # but not processed and no signals will be emitted.
        self._roi.setSize((w, h), update=False)
        self._roi.setPos((px, py), update=False)
        self._roi.stateChanged(finish=False)
        self.roi_region_change_sgn.emit(self._roi.rank, True, w, h, px, py)

    @QtCore.pyqtSlot(object)
    def onRoiRegionChangeFinished(self, roi):
        """Connect to the signal from an ROI object."""
        w, h = [int(v) for v in self._roi.size()]
        px, py = [int(v) for v in self._roi.pos()]
        self.updateParameters(w, h, px, py)
        # inform widgets outside this window
        self.roi_region_change_sgn.emit(self._roi.rank, True, w, h, px, py)

    def updateParameters(self, w, h, px, py):
        self._width_le.setText(str(w))
        self._height_le.setText(str(h))
        self._px_le.setText(str(px))
        self._py_le.setText(str(py))

    def setEditable(self, editable):
        for w in self._line_edits:
            w.setDisabled(not editable)


class _SingleRoiCtrlWidget(_RoiCtrlWidgetBase):
    """Widget for controlling of a single ROI."""

    def __init__(self, roi, *, parent=None):
        """Initialization.

        :param RectROI roi: RectROI object.
        """
        super().__init__(roi, parent=parent)

        self.activate_cb = QtGui.QCheckBox("On")
        self.lock_cb = QtGui.QCheckBox("Lock")

        self.initUI()

        self.disableAllEdit()

        self.activate_cb.stateChanged.connect(self.onToggleRoiActivation)
        self.lock_cb.stateChanged.connect(self.onLock)

    def initUI(self):
        layout = QtGui.QHBoxLayout()
        layout.addWidget(QtGui.QLabel(
            f"ROI{self._roi.rank} ({self._roi.color[0]}): "))
        layout.addWidget(self.activate_cb)
        layout.addWidget(self.lock_cb)
        layout.addWidget(QtGui.QLabel("w: "))
        layout.addWidget(self._width_le)
        layout.addWidget(QtGui.QLabel("h: "))
        layout.addWidget(self._height_le)
        layout.addWidget(QtGui.QLabel("x: "))
        layout.addWidget(self._px_le)
        layout.addWidget(QtGui.QLabel("y: "))
        layout.addWidget(self._py_le)

        self.setLayout(layout)
        # left, top, right, bottom
        self.layout().setContentsMargins(2, 1, 2, 1)

    @QtCore.pyqtSlot(int)
    def onToggleRoiActivation(self, state):
        if state == QtCore.Qt.Checked:
            self._roi.show()
            self.enableAllEdit()
            self.roi_region_change_sgn.emit(
                self._roi.rank, True, *self._roi.size(), *self._roi.pos())
        else:
            self._roi.hide()
            self.disableAllEdit()
            self.roi_region_change_sgn.emit(
                self._roi.rank, False, *self._roi.size(), *self._roi.pos())

    @QtCore.pyqtSlot(int)
    def onLock(self, state):
        self._roi.setLocked(state == QtCore.Qt.Checked)
        self.setEditable(not state == QtCore.Qt.Checked)

    def disableAllEdit(self):
        self.setEditable(False)
        self.lock_cb.setDisabled(True)

    def enableAllEdit(self):
        self.lock_cb.setDisabled(False)
        self.setEditable(True)


class _RoisCtrlWidget(QtGui.QGroupBox):
    """Widget for controlling of a group of ROIs."""

    _available_roi_foms = OrderedDict({
        "sum": RoiFom.SUM,
        "mean": RoiFom.MEAN,
    })

    roi_fom_sgn = QtCore.pyqtSignal(object)

    def __init__(self, rois, *, parent=None):
        super().__init__(parent)

        mediator = Mediator()

        self._clear_roi_hist_btn = QtGui.QPushButton("Clear history")
        self._clear_roi_hist_btn.clicked.connect(mediator.roi_hist_clear_sgn)

        self._roi_displayed_range_le = QtGui.QLineEdit(str(600))
        validator = QtGui.QIntValidator()
        validator.setBottom(1)
        self._roi_displayed_range_le.setValidator(validator)
        self._roi_displayed_range_le.editingFinished.connect(
            mediator.onRoiDisplayedRangeChange)

        self._roi_fom_cb = QtGui.QComboBox()
        for v in self._available_roi_foms:
            self._roi_fom_cb.addItem(v)
        self._roi_fom_cb.currentTextChanged.connect(
            lambda x: self.roi_fom_sgn.emit(
                self._available_roi_foms[x]))
        self.roi_fom_sgn.connect(mediator.roi_fom_change_sgn)
        self._roi_fom_cb.currentTextChanged.emit(
            self._roi_fom_cb.currentText())

        self._roi_ctrls = []
        for roi in rois:
            widget = _SingleRoiCtrlWidget(roi)
            self._roi_ctrls.append(widget)
            widget.roi_region_change_sgn.connect(mediator.roi_region_change_sgn)

        self.initUI()

    def initUI(self):
        layout = QtGui.QGridLayout()
        layout.addWidget(QtGui.QLabel("ROI value: "), 0, 0)
        layout.addWidget(self._roi_fom_cb, 0, 1)
        layout.addWidget(QtGui.QLabel("Displayed range: "), 0, 2)
        layout.addWidget(self._roi_displayed_range_le, 0, 3)
        layout.addWidget(self._clear_roi_hist_btn, 0, 4)
        for i, roi_ctrl in enumerate(self._roi_ctrls, 1):
            layout.addWidget(roi_ctrl, i, 0, 1, 5)
        self.setLayout(layout)


class _ImageCtrlWidget(QtGui.QWidget):
    """Widget inside the action bar for masking image."""

    moving_avg_window_sgn = QtCore.pyqtSignal(int)

    def __init__(self, parent=None):
        """Initialization"""
        super().__init__(parent=parent)

        self._moving_avg_le = QtGui.QLineEdit(str(1))
        self._moving_avg_le.setValidator(QtGui.QIntValidator(1, 1000000))
        self._moving_avg_le.setMinimumWidth(60)
        self._moving_avg_le.returnPressed.connect(lambda:
            self.moving_avg_window_sgn.emit(int(self._moving_avg_le.text())))

        self.initUI()

        self.setFixedSize(self.minimumSizeHint())

    def initUI(self):
        layout = QtGui.QHBoxLayout()
        layout.addWidget(QtGui.QLabel("Moving average: "))
        layout.addWidget(self._moving_avg_le)

        self.setLayout(layout)
        self.layout().setContentsMargins(2, 1, 2, 1)


class _MaskCtrlWidget(QtGui.QWidget):
    """Widget inside the action bar for masking image."""

    threshold_mask_sgn = QtCore.pyqtSignal(int, int)

    def __init__(self, parent=None):
        """Initialization"""
        super().__init__(parent=parent)

        self._min_pixel_le = QtGui.QLineEdit(str(int(config["MASK_RANGE"][0])))
        self._min_pixel_le.setValidator(QtGui.QIntValidator())
        # avoid collapse on online and maxwell clusters
        self._min_pixel_le.setMinimumWidth(80)
        self._min_pixel_le.returnPressed.connect(
            self.onThresholdMaskChanged)
        self._max_pixel_le = QtGui.QLineEdit(str(int(config["MASK_RANGE"][1])))
        self._max_pixel_le.setValidator(QtGui.QIntValidator())
        self._max_pixel_le.setMinimumWidth(80)
        self._max_pixel_le.returnPressed.connect(
            self.onThresholdMaskChanged)

        self.initUI()

        self.setFixedSize(self.minimumSizeHint())

    def initUI(self):
        layout = QtGui.QHBoxLayout()
        layout.addWidget(QtGui.QLabel("Min. mask: "))
        layout.addWidget(self._min_pixel_le)
        layout.addWidget(QtGui.QLabel("Max. mask: "))
        layout.addWidget(self._max_pixel_le)

        self.setLayout(layout)
        self.layout().setContentsMargins(2, 1, 2, 1)

    def onThresholdMaskChanged(self):
        self.threshold_mask_sgn.emit(int(self._min_pixel_le.text()),
                                     int(self._max_pixel_le.text()))


class _ImageProcWidget(QtGui.QGroupBox):

    class RoiWidget(_RoiCtrlWidgetBase):
        def __init__(self, roi, *, parent=None):
            super().__init__(roi, parent=parent)

            self.initUI()

        def initUI(self):
            """Override."""
            layout = QtGui.QGridLayout()
            layout.addWidget(QtGui.QLabel("w: "), 2, 0)
            layout.addWidget(self._width_le, 2, 1)
            layout.addWidget(QtGui.QLabel("h: "), 2, 2)
            layout.addWidget(self._height_le, 2, 3)
            layout.addWidget(QtGui.QLabel("x: "), 3, 0)
            layout.addWidget(self._px_le, 3, 1)
            layout.addWidget(QtGui.QLabel("y: "), 3, 2)
            layout.addWidget(self._py_le, 3, 3)
            self.setLayout(layout)

    def __init__(self, *, parent=None):
        super().__init__(parent)

        mediator = Mediator()

        self.bkg_le = QtGui.QLineEdit(str(0.0))
        self.bkg_le.setValidator(QtGui.QDoubleValidator())

        self.update_image_btn = QtGui.QPushButton("Update image")

        self._auto_level_btn = QtGui.QPushButton("Auto level")
        self._auto_level_btn.clicked.connect(mediator.onAutoLevel)

        self.set_ref_btn = QtGui.QPushButton("Set reference")
        self.remove_ref_btn = QtGui.QPushButton("Remove reference")

        self.initUI()

    def initUI(self):
        """Override."""
        AR = QtCore.Qt.AlignRight

        layout = QtGui.QGridLayout()
        layout.addWidget(QtGui.QLabel("Subtract bkg: "), 0, 0, 1, 2, AR)
        layout.addWidget(self.bkg_le, 0, 2, 1, 2)
        layout.addWidget(self.update_image_btn, 2, 0, 1, 2)
        layout.addWidget(self._auto_level_btn, 2, 2, 1, 2)
        layout.addWidget(self.set_ref_btn, 3, 0, 1, 2)
        layout.addWidget(self.remove_ref_btn, 3, 2, 1, 2)
        self.setLayout(layout)


@SingletonWindow
class ImageToolWindow(AbstractWindow):
    """ImageToolWindow class.

    This is the second Main GUI which focuses on manipulating the image
    , e.g. selecting ROI, masking, cropping, normalization, for different
    data analysis scenarios.
    """

    title = "Image tool"

    _root_dir = osp.dirname(osp.abspath(__file__))

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._image_view = ImageAnalysis(hide_axis=False, parent=self)

        self._roi_ctrl_widget = _RoisCtrlWidget(
            self._image_view.rois, parent=self)
        self._image_proc_widget = _ImageProcWidget(parent=self)

        self._image_proc_widget.bkg_le.editingFinished.connect(
            self._image_view.onBkgChange)
        self._image_proc_widget.update_image_btn.clicked.connect(
            self.updateImage)
        self._image_proc_widget.set_ref_btn.clicked.connect(
            self._image_view.setImageRef)
        self._image_proc_widget.remove_ref_btn.clicked.connect(
            self._image_view.removeImageRef)

        #
        # image tool bar
        #

        self._tool_bar_image = self.addToolBar("image")

        self._image_ctrl = _ImageCtrlWidget()
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

        self._mask_ctrl = _MaskCtrlWidget()
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
        layout = QtGui.QGridLayout()
        layout.addWidget(self._image_view, 0, 0, 1, 3)
        layout.addWidget(self._roi_ctrl_widget, 1, 0, 1, 2)
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
