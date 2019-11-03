"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import os.path as osp
import functools

from PyQt5 import QtCore, QtGui, QtWidgets

from ..misc_widgets import Colors
from ..windows.base_window import AbstractWindow
from ..mediator import Mediator
from ..plot_widgets import ImageAnalysis, ImageViewF
from ..ctrl_widgets.smart_widgets import (
    SmartLineEdit, SmartBoundaryLineEdit
)
from ...pipeline.data_model import ImageData
from ...utils import cached_property
from ...config import config, MaskState
from ...algorithms import mask_image


class _SimpleImageData:
    """SimpleImageData which is used by ImageToolWindow.

    In ImageToolWindow, some properties of the image can be changed, for
    instance, background, threshold mask, etc.

    Attributes:
        pixel_size (float): pixel size of the detector.
        threshold_mask (tuple): (lower, upper) boundaries of the
            threshold mask.
        background (float): a uniform background value.
        masked (numpy.ndarray): image with threshold mask.
    """

    def __init__(self, image_data):
        """Initialization.

        Construct a _SimpleImageData instance from an ImageData instance.

        :param ImageData image_data: an ImageData instance.
        """
        if not isinstance(image_data, ImageData):
            raise TypeError("Input must be an ImageData instance.")

        self._pixel_size = image_data.pixel_size

        # This is only used for reset the image in the ImageTool, which
        # does not occur very often. Therefore, the copy is used to avoid
        # data sharing.
        # Note: image_data.mean does not contain any NaN
        self._image = image_data.mean.copy()

        # Note:: we do not copy 'masked_mean' since it also includes image_mask

        # image mask is plotted on top of the image in ImageTool

        self._bkg = image_data.background
        self._threshold_mask = image_data.threshold_mask

    @property
    def pixel_size(self):
        return self._pixel_size

    @property
    def background(self):
        return self._bkg

    @background.setter
    def background(self, v):
        if v == self._bkg:
            return
        self._image -= v - self._bkg  # in-place operation
        self._bkg = v

        # invalidate cache
        try:
            del self.__dict__['masked']
        except KeyError:
            pass

    @property
    def threshold_mask(self):
        return self._threshold_mask

    @threshold_mask.setter
    def threshold_mask(self, mask):
        if mask == self._threshold_mask:
            return

        self._threshold_mask = mask

        # invalid cache
        del self.__dict__['masked']

    @cached_property
    def masked(self):
        return mask_image(self._image,
                          threshold_mask=self._threshold_mask,
                          inplace=False)

    @classmethod
    def from_array(cls, arr):
        """Instantiate from an array."""
        return cls(ImageData.from_array(arr))


class _InformationWidget(QtWidgets.QFrame):
    """InformationWidget.

    Widget used to display the basic information of the current image data.
    """
    _LCD_DIGITS = 12

    def __init__(self, parent=None):
        super().__init__(parent)

        self._current_tid = QtWidgets.QLCDNumber(self._LCD_DIGITS)
        self._n_total_pulses = QtWidgets.QLCDNumber(self._LCD_DIGITS)
        self._n_kept_pulses = QtWidgets.QLCDNumber(self._LCD_DIGITS)
        self._dark_train_counter = QtWidgets.QLCDNumber(self._LCD_DIGITS)

        self.updatePulsesInfo(0, 0)

        self.initUI()

    def initUI(self):
        self._setLcdStyle(self._current_tid)
        self._setLcdStyle(self._n_total_pulses)
        self._setLcdStyle(self._n_kept_pulses)
        self._setLcdStyle(self._dark_train_counter)

        layout = QtGui.QGridLayout()
        AR = QtCore.Qt.AlignRight

        layout.addWidget(QtWidgets.QLabel("Current train ID: "), 0, 0, AR)
        layout.addWidget(self._current_tid, 0, 1)
        layout.addWidget(QtWidgets.QLabel("Total # of pulses/train: "), 1, 0, AR)
        layout.addWidget(self._n_total_pulses, 1, 1)
        layout.addWidget(QtWidgets.QLabel("# of kept pulses/train: "), 2, 0, AR)
        layout.addWidget(self._n_kept_pulses, 2, 1)
        layout.addWidget(QtWidgets.QLabel("# of dark trains: "), 3, 0, AR)
        layout.addWidget(self._dark_train_counter, 3, 1)
        self.setLayout(layout)

    def _setLcdStyle(self, lcd):
        lcd.setLineWidth(0)
        lcd.setSegmentStyle(QtWidgets.QLCDNumber.Flat)
        palette = lcd.palette()
        palette.setColor(palette.WindowText, QtGui.QColor(85, 85, 255))
        lcd.setPalette(palette)

    def updateTrainId(self, tid):
        self._current_tid.display(tid)

    def updatePulsesInfo(self, n_total, n_kept):
        self._n_total_pulses.display(n_total)
        self._n_kept_pulses.display(n_kept)

    def updateDarkTrainCount(self, count):
        self._dark_train_counter.display(count)


class _RoiCtrlWidgetBase(QtGui.QWidget):
    """Base class for RoiCtrlWidget.

    Implemented four QLineEdits (x, y, w, h) and their connection to the
    corresponding ROI.
    """
    # (rank, x, y, w, h) where rank starts from 1
    roi_geometry_change_sgn = QtCore.Signal(object)
    # (rank, visible)
    roi_visibility_change_sgn = QtCore.Signal(object)

    _pos_validator = QtGui.QIntValidator(-10000, 10000)
    _size_validator = QtGui.QIntValidator(1, 10000)

    def __init__(self, roi, *, parent=None):
        super().__init__(parent=parent)
        self._roi = roi

        self._width_le = SmartLineEdit()
        self._width_le.setValidator(self._size_validator)
        self._height_le = SmartLineEdit()
        self._height_le.setValidator(self._size_validator)
        self._px_le = SmartLineEdit()
        self._px_le.setValidator(self._pos_validator)
        self._py_le = SmartLineEdit()
        self._py_le.setValidator(self._pos_validator)
        self._width_le.returnPressed.connect(self.onRoiSizeEdited)
        self._height_le.returnPressed.connect(self.onRoiSizeEdited)
        self._px_le.returnPressed.connect(self.onRoiPositionEdited)
        self._py_le.returnPressed.connect(self.onRoiPositionEdited)

        self._line_edits = (self._width_le, self._height_le,
                            self._px_le, self._py_le)

        roi.sigRegionChangeFinished.connect(self.onRoiGeometryChangeFinished)

    @QtCore.pyqtSlot()
    def onRoiPositionEdited(self):
        x, y = [int(v) for v in self._roi.pos()]
        w, h = [int(v) for v in self._roi.size()]

        if self.sender() == self._px_le:
            x = int(self._px_le.text())
        elif self.sender() == self._py_le:
            y = int(self._py_le.text())

        # If 'update' == False, the state change will be remembered
        # but not processed and no signals will be emitted.
        self._roi.setPos((x, y), update=False)
        # trigger sigRegionChanged which moves the handler(s)
        # finish=False -> sigRegionChangeFinished will not emit, which
        # otherwise triggers infinite recursion
        self._roi.stateChanged(finish=False)

        self.roi_geometry_change_sgn.emit((self._roi.rank, x, y, w, h))

    @QtCore.pyqtSlot()
    def onRoiSizeEdited(self):
        x, y = [int(v) for v in self._roi.pos()]
        w, h = [int(v) for v in self._roi.size()]
        if self.sender() == self._width_le:
            w = int(self._width_le.text())
        elif self.sender() == self._height_le:
            h = int(self._height_le.text())

        # If 'update' == False, the state change will be remembered
        # but not processed and no signals will be emitted.
        self._roi.setSize((w, h), update=False)
        # trigger sigRegionChanged which moves the handler(s)
        # finish=False -> sigRegionChangeFinished will not emit, which
        # otherwise triggers infinite recursion
        self._roi.stateChanged(finish=False)

        self.roi_geometry_change_sgn.emit((self._roi.rank, x, y, w, h))

    @QtCore.pyqtSlot(object)
    def onRoiGeometryChangeFinished(self, roi):
        """Connect to the signal from an ROI object."""
        x, y = [int(v) for v in roi.pos()]
        w, h = [int(v) for v in roi.size()]
        self.updateParameters(x, y, w, h)
        # inform widgets outside this window
        self.roi_geometry_change_sgn.emit((self._roi.rank, x, y, w, h))

    def notifyRoiParams(self):
        # fill the QLineEdit(s)
        self._roi.sigRegionChangeFinished.emit(self._roi)

    def updateParameters(self, x, y, w, h):
        self.roi_geometry_change_sgn.disconnect()
        self._px_le.setText(str(x))
        self._py_le.setText(str(y))
        self._width_le.setText(str(w))
        self._height_le.setText(str(h))
        self.roi_geometry_change_sgn.connect(Mediator().onRoiGeometryChange)

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
        self.activate_cb.stateChanged.emit(self.activate_cb.checkState())
        self.lock_cb.stateChanged.connect(self.onLock)

    def initUI(self):
        layout = QtGui.QHBoxLayout()
        rank = self._roi.rank
        layout.addWidget(QtGui.QLabel(
            f"ROI{rank} ({config['ROI_COLORS'][rank-1]}): "))
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
        activated = state == QtCore.Qt.Checked
        if activated:
            self._roi.show()
            self.enableAllEdit()
        else:
            self._roi.hide()
            self.disableAllEdit()

        self.roi_visibility_change_sgn.emit((self._roi.rank, activated))

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


class _RoiCtrlWidgetGroup(QtGui.QGroupBox):
    """Widget for controlling of a group of ROIs."""

    def __init__(self, rois, *, parent=None):
        super().__init__(parent)

        self._roi_ctrls = []
        for roi in rois:
            widget = _SingleRoiCtrlWidget(roi)
            self._roi_ctrls.append(widget)

        self.initUI()
        self.initConnections()
        self.updateMetaData()

    def initUI(self):
        layout = QtGui.QGridLayout()
        for i, roi_ctrl in enumerate(self._roi_ctrls):
            layout.addWidget(roi_ctrl, i, 0, 1, 5)
        self.setLayout(layout)

    def initConnections(self):
        mediator = Mediator()

        for widget in self._roi_ctrls:
            widget.roi_geometry_change_sgn.connect(mediator.onRoiGeometryChange)
            widget.roi_visibility_change_sgn.connect(
                mediator.onRoiVisibilityChange)

    def updateMetaData(self):
        for i, widget in enumerate(self._roi_ctrls, 1):
            widget.notifyRoiParams()
            widget.roi_visibility_change_sgn.emit(
                (i, widget.activate_cb.checkState() == QtCore.Qt.Checked))


class _ImageCtrlWidget(QtGui.QGroupBox):
    def __init__(self, *, parent=None):
        super().__init__(parent)

        self.auto_update_cb = QtWidgets.QCheckBox("Update automatically")
        self.auto_update_cb.setChecked(True)
        self.update_image_btn = QtGui.QPushButton("Update image")
        self.update_image_btn.setEnabled(False)

        # It is just a placeholder
        self.moving_avg_le = SmartLineEdit(str(1))
        self.moving_avg_le.setValidator(QtGui.QIntValidator(1, 9999999))
        self.moving_avg_le.setMinimumWidth(60)
        self.moving_avg_le.setEnabled(False)

        self.threshold_mask_le = SmartBoundaryLineEdit(
            ', '.join([str(v) for v in config["MASK_RANGE"]]))
        # avoid collapse on online and maxwell clusters
        self.threshold_mask_le.setMinimumWidth(160)

        self.darksubtraction_cb = QtWidgets.QCheckBox("Subtract dark")
        self.darksubtraction_cb.setChecked(True)

        self.bkg_le = SmartLineEdit(str(0.0))
        self.bkg_le.setValidator(QtGui.QDoubleValidator())

        self.auto_level_btn = QtGui.QPushButton("Auto level")
        self.save_image_btn = QtGui.QPushButton("Save image")
        self.load_ref_btn = QtGui.QPushButton("Load reference")
        self.set_ref_btn = QtGui.QPushButton("Set reference")
        self.remove_ref_btn = QtGui.QPushButton("Remove reference")

        self.initUI()
        self.initConnections()

    def initUI(self):
        """Override."""
        layout = QtGui.QGridLayout()
        AR = QtCore.Qt.AlignRight

        row = 0
        layout.addWidget(self.update_image_btn, row, 0)
        layout.addWidget(self.auto_update_cb, row, 1, AR)

        row += 1
        layout.addWidget(self.auto_level_btn, row, 0)

        row += 1
        layout.addWidget(QtGui.QLabel("Moving average: "), row, 0, AR)
        layout.addWidget(self.moving_avg_le, row, 1)

        row += 1
        layout.addWidget(QtGui.QLabel("Threshold mask: "), row, 0, AR)
        layout.addWidget(self.threshold_mask_le, row, 1)

        row += 1
        layout.addWidget(self.darksubtraction_cb, row, 0, AR)

        row += 1
        layout.addWidget(QtGui.QLabel("Subtract background: "), row, 0, AR)
        layout.addWidget(self.bkg_le, row, 1)

        row += 1
        layout.addWidget(self.save_image_btn, row, 0)
        layout.addWidget(self.load_ref_btn, row, 1)

        row += 1
        layout.addWidget(self.set_ref_btn, row, 0)
        layout.addWidget(self.remove_ref_btn, row, 1)

        layout.setVerticalSpacing(20)
        self.setLayout(layout)

    def initConnections(self):
        self.auto_update_cb.toggled.connect(
            lambda: self.update_image_btn.setEnabled(
                not self.sender().isChecked()))

    def updateMetaData(self):
        self.threshold_mask_le.returnPressed.emit()
        self.darksubtraction_cb.toggled.emit(
            self.darksubtraction_cb.isChecked())
        self.bkg_le.returnPressed.emit()


class ImageToolWindow(AbstractWindow):
    """ImageToolWindow class.

    This is the second Main GUI which focuses on manipulating the image
    , e.g. selecting ROI, masking, normalization, for different
    data analysis scenarios.
    """

    title = "Image tool"

    _root_dir = osp.dirname(osp.abspath(__file__))

    _WIDTH, _HEIGHT = config['GUI']['IMAGE_TOOL_SIZE']

    __instance = None

    @classmethod
    def reset(cls):
        cls.__instance = None

    def __new__(cls, *args, **kwargs):
        """Create a singleton."""
        if cls.__instance is None:
            instance = super().__new__(cls, *args, **kwargs)
            instance._is_initialized = False
            cls.__instance = instance
            return instance

        instance = cls.__instance
        parent = instance.parent()
        if parent is not None:
            parent.registerWindow(instance)

        instance.show()
        instance.activateWindow()
        return instance

    def __init__(self, *args, **kwargs):
        if self._is_initialized:
            return
        super().__init__(*args, **kwargs)

        self._image_views = QtWidgets.QTabWidget(self)
        self._data_view = ImageAnalysis(hide_axis=False)
        self._dark_view = ImageViewF()
        self._image_views.addTab(self._data_view, "Corrected")
        self._image_views.addTab(self._dark_view, "Dark")

        # --------
        # tool bar
        # --------

        # ***************
        # masking actions
        # ***************

        self._tool_bar = self.addToolBar("tools")

        self._mask_at = self._addAction(self._tool_bar, "Mask", "mask.png")
        self._mask_at.setCheckable(True)
        self._unmask_at = self._addAction(
            self._tool_bar, "Unmask", "un_mask.png")
        self._unmask_at.setCheckable(True)
        self._clear_mask_at = self._addAction(
            self._tool_bar, "Clear mask", "clear_mask.png")
        self._save_img_mask_at = self._addAction(
            self._tool_bar, "Save image mask", "save_mask.png")
        self._load_img_mask_at = self._addAction(
            self._tool_bar, "Load image mask", "load_mask.png")

        self._exclusive_actions = {self._mask_at, self._unmask_at}

        # ****************
        # dark run actions
        # ****************

        self._tool_bar.addSeparator()

        self._record_at = self._addAction(
            self._tool_bar, "Record dark", "record.png")
        self._record_at.setCheckable(True)
        self._record_at.setEnabled(False)
        self._remove_at = self._addAction(
            self._tool_bar, "Remove dark", "remove_dark.png")

        self._tool_bar.addSeparator()

        # -----------------------------
        # Other ctrl widgets
        # -----------------------------

        self._roi_ctrl_widget = _RoiCtrlWidgetGroup(self._data_view.rois)

        self._info_widget = _InformationWidget()

        self._image_ctrl_widget = _ImageCtrlWidget()

        self._auto_update = self._image_ctrl_widget.auto_update_cb.isChecked()

        self.initUI()
        self.initConnections()
        self.updateMetaData()

        self.resize(self._WIDTH, self._HEIGHT)

        self.update()

        self._is_initialized = True

    def initUI(self):
        """Override."""
        AT = QtCore.Qt.AlignTop

        right_panel = QtWidgets.QWidget()
        right_panel_layout = QtGui.QVBoxLayout()
        right_panel_layout.addWidget(self._info_widget)
        right_panel_layout.addWidget(self._image_ctrl_widget)
        right_panel_layout.addStretch(1)
        right_panel.setLayout(right_panel_layout)

        layout = QtGui.QGridLayout()
        right_panel.setFixedSize(right_panel.minimumSizeHint())

        layout.addWidget(self._image_views, 0, 0, 1, 4)
        layout.addWidget(self._roi_ctrl_widget, 1, 0, 1, 4)
        layout.addWidget(right_panel, 0, 4, 2, 1, AT)

        self._cw.setLayout(layout)
        self.layout().setContentsMargins(0, 0, 0, 0)

    def initConnections(self):
        mediator = self._mediator

        # Note: the sequence of the following two 'connect'
        self._mask_at.toggled.connect(self._exclude_actions)
        self._mask_at.toggled.connect(functools.partial(
            self._data_view.onDrawToggled, MaskState.MASK))

        # Note: the sequence of the following two 'connect'
        self._unmask_at.toggled.connect(self._exclude_actions)
        self._unmask_at.toggled.connect(functools.partial(
            self._data_view.onDrawToggled, MaskState.UNMASK))

        self._clear_mask_at.triggered.connect(
            self._data_view.onClearImageMask)
        self._save_img_mask_at.triggered.connect(
            self._data_view.saveImageMask)
        self._load_img_mask_at.triggered.connect(
            self._data_view.loadImageMask)

        self._record_at.toggled.connect(self._mediator.onRdStateChange)
        self._record_at.toggled.emit(self._record_at.isChecked())
        self._remove_at.triggered.connect(self._mediator.onRdRemoveDark)

        self._image_ctrl_widget.auto_update_cb.toggled.connect(
            self._autoUpdateToggled)
        self._image_ctrl_widget.update_image_btn.clicked.connect(
            self.updateImage)
        self._image_ctrl_widget.auto_level_btn.clicked.connect(
            mediator.reset_image_level_sgn)
        self._image_ctrl_widget.save_image_btn.clicked.connect(
            self._data_view.writeImage)
        self._image_ctrl_widget.load_ref_btn.clicked.connect(
            self._data_view.loadReferenceImage)
        self._image_ctrl_widget.set_ref_btn.clicked.connect(
            self._data_view.setReferenceImage)
        self._image_ctrl_widget.remove_ref_btn.clicked.connect(
            self._data_view.removeReferenceImage)

        self._image_ctrl_widget.threshold_mask_le.value_changed_sgn.connect(
            lambda x: self._data_view.onThresholdMaskChange(x))
        self._image_ctrl_widget.threshold_mask_le.value_changed_sgn.connect(
            lambda x: mediator.onImageThresholdMaskChange(x))

        self._image_ctrl_widget.darksubtraction_cb.toggled.connect(
            self._mediator.onDarkSubtractionStateChange)

        self._image_ctrl_widget.bkg_le.value_changed_sgn.connect(
            lambda x: self._data_view.onBkgChange(float(x)))
        self._image_ctrl_widget.bkg_le.value_changed_sgn.connect(
            lambda x: mediator.onImageBackgroundChange(float(x)))

        self._image_views.currentChanged.connect(
            self.onImageViewTabChanged)

    def updateMetaData(self):
        """Override."""
        self._image_ctrl_widget.updateMetaData()

        return True

    def updateWidgetsF(self):
        """Update widgets.

        This method is called by the main GUI.
        """
        self._update_data_view()
        self._update_dark_view()

    def _update_data_view(self):
        # Always automatically get an image
        if self._data_view.image is None or self._auto_update:
            self.updateImage(auto_range=True, auto_levels=True)

    def _update_dark_view(self):
        data = self._data.get()
        if data is None:
            return

        if data.image.dark_mean is None:
            self._dark_view.clear()
        else:
            self._dark_view.setImage(data.image.dark_mean)

        self._info_widget.updateDarkTrainCount(data.image.dark_count)

    def updateImage(self, **kwargs):
        """Update the current image.

        :param kwargs: forward to ImageView.setImage().

        It is only used for updating the image manually.
        """
        data = self._data.get()
        if data is None:
            return

        try:
            self._data_view.setImageData(_SimpleImageData(data.image), **kwargs)
        except (AttributeError, TypeError):
            return

        self._info_widget.updateTrainId(data.tid)

        n_total = data.n_pulses
        self._info_widget.updatePulsesInfo(n_total, data.pidx.n_kept(n_total))

    @QtCore.pyqtSlot(bool)
    def _exclude_actions(self, checked):
        if checked:
            for at in self._exclusive_actions:
                if at != self.sender():
                    at.setChecked(False)

    def _addAction(self, tool_bar, description, filename):
        icon = QtGui.QIcon(osp.join(self._root_dir, "../icons/" + filename))
        action = QtGui.QAction(icon, description, tool_bar)
        tool_bar.addAction(action)
        return action

    def onImageViewTabChanged(self, idx):
        if self._image_views.tabText(idx) == 'Dark':
            self._record_at.setEnabled(True)
        else:
            if self._record_at.isChecked():
                self._record_at.trigger()
            self._record_at.setEnabled(False)

    def _autoUpdateToggled(self):
        self._auto_update = self.sender().isChecked()
