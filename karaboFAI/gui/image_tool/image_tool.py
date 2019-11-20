"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from enum import IntEnum
import os.path as osp
import functools

from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QDoubleValidator, QIcon, QIntValidator
from PyQt5.QtWidgets import (
    QAction, QCheckBox, QGridLayout, QHBoxLayout, QLabel, QMainWindow,
    QPushButton, QTabWidget, QVBoxLayout, QWidget
)

from .azimuthal_integ_1d_view import AzimuthalInteg1dView
from .corrected_view import CorrectedView
from .bulletin_view import BulletinView
from .dark_view import DarkView
from .geometry_view import GeometryView
from ..mediator import Mediator
from ..windows import _AbstractWindowMixin
from ..ctrl_widgets import _AbstractCtrlWidget
from ..ctrl_widgets.smart_widgets import (
    SmartLineEdit, SmartBoundaryLineEdit
)
from ...config import AnalysisType, config, MaskState


class _ImageCtrlWidget(_AbstractCtrlWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.auto_update_cb = QCheckBox("Update automatically")
        self.auto_update_cb.setChecked(True)
        self.update_image_btn = QPushButton("Update image")
        self.update_image_btn.setEnabled(False)

        # It is just a placeholder
        self.moving_avg_le = SmartLineEdit(str(1))
        self.moving_avg_le.setValidator(QIntValidator(1, 9999999))
        self.moving_avg_le.setMinimumWidth(60)
        self.moving_avg_le.setEnabled(False)

        self.threshold_mask_le = SmartBoundaryLineEdit(
            ', '.join([str(v) for v in config["MASK_RANGE"]]))
        # avoid collapse on online and maxwell clusters
        self.threshold_mask_le.setMinimumWidth(160)

        self.darksubtraction_cb = QCheckBox("Subtract dark")
        self.darksubtraction_cb.setChecked(True)

        self.bkg_le = SmartLineEdit(str(0.0))
        self.bkg_le.setValidator(QDoubleValidator())

        self.auto_level_btn = QPushButton("Auto level")
        self.save_image_btn = QPushButton("Save image")
        self.load_ref_btn = QPushButton("Load reference")
        self.set_ref_btn = QPushButton("Set reference")
        self.remove_ref_btn = QPushButton("Remove reference")

        self.initUI()
        self.initConnections()

    def initUI(self):
        """Override."""
        layout = QGridLayout()
        AR = Qt.AlignRight

        row = 0
        layout.addWidget(self.update_image_btn, row, 0)
        layout.addWidget(self.auto_update_cb, row, 1, AR)

        row += 1
        layout.addWidget(self.auto_level_btn, row, 0)

        row += 1
        layout.addWidget(QLabel("Moving average: "), row, 0, AR)
        layout.addWidget(self.moving_avg_le, row, 1)

        row += 1
        layout.addWidget(QLabel("Threshold mask: "), row, 0, AR)
        layout.addWidget(self.threshold_mask_le, row, 1)

        row += 1
        layout.addWidget(self.darksubtraction_cb, row, 0, AR)

        row += 1
        layout.addWidget(QLabel("Subtract background: "), row, 0, AR)
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
        """Override."""
        self.auto_update_cb.toggled.connect(
            lambda: self.update_image_btn.setEnabled(
                not self.sender().isChecked()))

    def updateMetaData(self):
        """Override."""
        self.threshold_mask_le.returnPressed.emit()
        self.darksubtraction_cb.toggled.emit(
            self.darksubtraction_cb.isChecked())
        self.bkg_le.returnPressed.emit()
        return True


class ImageToolWindow(QMainWindow, _AbstractWindowMixin):
    """ImageToolWindow class.

    This is the second Main GUI which focuses on manipulating the image
    , e.g. selecting ROI, masking, normalization, for different
    data analysis scenarios.
    """
    _title = "Image tool"

    _root_dir = osp.dirname(osp.abspath(__file__))

    _WIDTH, _HEIGHT = config['GUI']['IMAGE_TOOL_SIZE']

    class TabIndex(IntEnum):
        CORRECTED = 0
        DARK = 1
        AZIMUTHAL_INTEG_1D = 2
        GEOMETRY = 3

    def __init__(self, queue, *, pulse_resolved=True, parent=None):
        """Initialization.

        :param deque queue: data queue.
        :param bool pulse_resolved: whether the related data is
            pulse-resolved or not.
        """
        super().__init__(parent=parent)

        self._queue = queue
        self._pulse_resolved = pulse_resolved

        self._mediator = Mediator()

        try:
            title = parent.title + " - " + self._title
        except AttributeError:
            title = self._title  # for unit test where parent is None
        self.setWindowTitle(title)

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
        # ctrl panel
        # -----------------------------
        self._ctrl_widgets = []

        self._bulletin_view = self.createView(BulletinView)
        self._image_ctrl_widget = self.createCtrlWidget(_ImageCtrlWidget)

        # -----------------------------
        # view panel
        # -----------------------------

        self._views_tab = QTabWidget()
        self._prev_tab_idx = 0
        self._corrected_view = self.createView(CorrectedView)
        self._dark_view = self.createView(DarkView)
        self._azimuthal_integ_1d_view = self.createView(AzimuthalInteg1dView)
        self._geometry_view = self.createView(GeometryView)

        # Whether the view is updated automatically
        self._auto_update = True

        self._cw = QWidget()
        self.setCentralWidget(self._cw)

        self.initUI()
        self.initConnections()
        self.updateMetaData()

        self.resize(self._WIDTH, self._HEIGHT)

    def initUI(self):
        """Override."""
        corrected_tab_idx = self._views_tab.addTab(
            self._corrected_view, "Corrected")
        dark_tab_idx = self._views_tab.addTab(self._dark_view, "Dark")
        azimuthal_integ_tab_idx = self._views_tab.addTab(
            self._azimuthal_integ_1d_view, "Azimuthal integration 1D")
        geom_tab_idx = self._views_tab.addTab(self._geometry_view, "Geometry")
        if not config['REQUIRE_GEOMETRY']:
            self._views_tab.setTabEnabled(geom_tab_idx, False)

        assert(corrected_tab_idx == self.TabIndex.CORRECTED)
        assert(dark_tab_idx == self.TabIndex.DARK)
        assert(azimuthal_integ_tab_idx == self.TabIndex.AZIMUTHAL_INTEG_1D)
        assert(geom_tab_idx == self.TabIndex.GEOMETRY)
        self._prev_tab_idx = self._views_tab.currentIndex()

        ctrl_panel = QWidget()
        ctrl_panel_layout = QVBoxLayout()
        ctrl_panel_layout.addWidget(self._bulletin_view)
        ctrl_panel_layout.addWidget(self._image_ctrl_widget)
        ctrl_panel_layout.addStretch(1)
        ctrl_panel.setLayout(ctrl_panel_layout)
        ctrl_panel.setFixedWidth(ctrl_panel.minimumSizeHint().width() + 10)

        layout = QHBoxLayout()
        layout.addWidget(self._views_tab)
        layout.addWidget(ctrl_panel)
        self._cw.setLayout(layout)
        self.layout().setContentsMargins(0, 0, 0, 0)

    def initConnections(self):
        self._image_ctrl_widget.auto_update_cb.toggled.connect(
            self.onAutoUpdateToggled)

        mediator = self._mediator

        # Note: the sequence of the following two 'connect'
        self._mask_at.toggled.connect(self._exclude_actions)
        self._mask_at.toggled.connect(functools.partial(
            self._corrected_view.imageView.onDrawToggled, MaskState.MASK))

        # Note: the sequence of the following two 'connect'
        self._unmask_at.toggled.connect(self._exclude_actions)
        self._unmask_at.toggled.connect(functools.partial(
            self._corrected_view.imageView.onDrawToggled, MaskState.UNMASK))

        self._clear_mask_at.triggered.connect(
            self._corrected_view.imageView.onClearImageMask)
        self._save_img_mask_at.triggered.connect(
            self._corrected_view.imageView.saveImageMask)
        self._load_img_mask_at.triggered.connect(
            self._corrected_view.imageView.loadImageMask)

        self._record_at.toggled.connect(self._mediator.onRdStateChange)
        self._record_at.toggled.emit(self._record_at.isChecked())
        self._remove_at.triggered.connect(self._mediator.onRdRemoveDark)

        self._image_ctrl_widget.update_image_btn.clicked.connect(
            self.updateImage)
        self._image_ctrl_widget.auto_level_btn.clicked.connect(
            mediator.reset_image_level_sgn)
        self._image_ctrl_widget.save_image_btn.clicked.connect(
            self._corrected_view.imageView.writeImage)
        self._image_ctrl_widget.load_ref_btn.clicked.connect(
            self._corrected_view.imageView.loadReferenceImage)
        self._image_ctrl_widget.set_ref_btn.clicked.connect(
            self._corrected_view.imageView.setReferenceImage)
        self._image_ctrl_widget.remove_ref_btn.clicked.connect(
            self._corrected_view.imageView.removeReferenceImage)

        # use lambda here to facilitate unittest of slot call
        self._image_ctrl_widget.threshold_mask_le.value_changed_sgn.connect(
            lambda x: self._corrected_view.imageView.onThresholdMaskChange(x))
        self._image_ctrl_widget.threshold_mask_le.value_changed_sgn.connect(
            lambda x: mediator.onImageThresholdMaskChange(x))

        self._image_ctrl_widget.darksubtraction_cb.toggled.connect(
            self._mediator.onDarkSubtractionStateChange)

        self._image_ctrl_widget.bkg_le.value_changed_sgn.connect(
            lambda x: self._corrected_view.imageView.onBkgChange(float(x)))
        self._image_ctrl_widget.bkg_le.value_changed_sgn.connect(
            lambda x: mediator.onImageBackgroundChange(float(x)))

        self._views_tab.currentChanged.connect(self.onViewsTabChanged)

    def createView(self, view_class):
        view = view_class(pulse_resolved=self._pulse_resolved, parent=self)
        return view

    def createCtrlWidget(self, widget_class):
        """Register a ctrl widget.

        Ctrl widgets reside in (views of) ImageToolWindow should explicitly
        call this method to be registered.
        """
        widget = widget_class(pulse_resolved=self._pulse_resolved)
        self._ctrl_widgets.append(widget)
        return widget

    def updateMetaData(self):
        """Update metadata from all the ctrl widgets.

        :returns bool: True if all metadata successfully parsed
            and emitted, otherwise False.
        """
        for widget in self._ctrl_widgets:
            succeeded = widget.updateMetaData()
            if not succeeded:
                return False
        return True

    def reset(self):
        """Override."""
        pass

    @pyqtSlot()
    def updateImage(self):
        """Used for updating manually."""
        self._updateOnce(True)

    def updateWidgetsF(self):
        """Override."""
        self._updateOnce(self._auto_update)

    def _updateOnce(self, auto_update):
        if len(self._queue) == 0:
            return
        data = self._queue[0]

        # update bulletin
        self._bulletin_view.updateF(data, auto_update)
        # update other ImageView/PlotWidget in the activated tab
        self._views_tab.currentWidget().updateF(data, auto_update)

    @pyqtSlot(bool)
    def _exclude_actions(self, checked):
        if checked:
            for at in self._exclusive_actions:
                if at != self.sender():
                    at.setChecked(False)

    def _addAction(self, tool_bar, description, filename):
        icon = QIcon(osp.join(self._root_dir, "../icons/" + filename))
        action = QAction(icon, description, tool_bar)
        tool_bar.addAction(action)
        return action

    @pyqtSlot(int)
    def onViewsTabChanged(self, idx):
        # clean-up for the old tab
        if self._prev_tab_idx == self.TabIndex.AZIMUTHAL_INTEG_1D:
            self._mediator.unregisterAnalysis(AnalysisType.AZIMUTHAL_INTEG)

        if self._prev_tab_idx == self.TabIndex.DARK:
            if self._record_at.isChecked():
                self._record_at.trigger()
            self._record_at.setEnabled(False)

        # update Tab index
        self._prev_tab_idx = idx

        # set-up for the new tab
        if idx == self.TabIndex.DARK:
            self._record_at.setEnabled(True)

        if idx == self.TabIndex.AZIMUTHAL_INTEG_1D:
            self._mediator.registerAnalysis(AnalysisType.AZIMUTHAL_INTEG)

    @pyqtSlot(bool)
    def onAutoUpdateToggled(self, state):
        self._auto_update = state
