"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from enum import IntEnum
import os.path as osp

from PyQt5.QtCore import pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import (
    QHBoxLayout, QMainWindow, QTabWidget, QVBoxLayout, QWidget
)

from .azimuthal_integ_1d_view import AzimuthalInteg1dView
from .corrected_view import CorrectedView
from .calibration_view import CalibrationView
from .bulletin_view import BulletinView
from .reference_view import ReferenceView
from .geometry_view import GeometryView
from .transform_view import TransformView
from ..mediator import Mediator
from ..windows import _AbstractWindowMixin
from ..ctrl_widgets import ImageCtrlWidget, MaskCtrlWidget
from ...config import config


class ImageToolWindow(QMainWindow, _AbstractWindowMixin):
    """ImageToolWindow class.

    This is the second Main GUI which focuses on manipulating the image
    , e.g. selecting ROI, masking, normalization, for different
    data analysis scenarios.
    """
    _title = "Image tool"

    _root_dir = osp.dirname(osp.abspath(__file__))

    _WIDTH, _HEIGHT = config['GUI_IMAGE_TOOL_SIZE']

    mask_file_path_sgn = pyqtSignal(str)

    class TabIndex(IntEnum):
        OVERVIEW = 0
        GAIN_OFFSET = 1
        REFERENCE = 2
        AZIMUTHAL_INTEG_1D = 3
        GEOMETRY = 4
        IMAGE_TRANSFORM = 5

    def __init__(self, queue, *,
                 pulse_resolved=True, require_geometry=True, parent=None):
        """Initialization.

        :param deque queue: data queue.
        :param bool pulse_resolved: whether the related data is
            pulse-resolved or not.
        :param bool require_geometry: whether the detector requires a
            geometry to assemble its modules.
        """
        super().__init__(parent=parent)

        self._queue = queue
        self._pulse_resolved = pulse_resolved
        self._require_geometry = require_geometry

        self._mediator = Mediator()

        try:
            title = parent.title + " - " + self._title
        except AttributeError:
            title = self._title  # for unit test where parent is None
        self.setWindowTitle(title)

        self._ctrl_widgets = []

        # -----------------------------
        # ctrl panel
        # -----------------------------

        self._bulletin_view = self.createView(BulletinView)
        self._image_ctrl_widget = self.createCtrlWidget(ImageCtrlWidget)
        self._mask_ctrl_widget = self.createCtrlWidget(MaskCtrlWidget)

        # -----------------------------
        # view panel
        # -----------------------------

        self._views_tab = QTabWidget()
        self._corrected_view = self.createView(CorrectedView)
        self._calibration_view = self.createView(CalibrationView)
        self._reference_view = self.createView(ReferenceView)
        self._azimuthal_integ_1d_view = self.createView(AzimuthalInteg1dView)
        self._geometry_view = self.createView(GeometryView)
        self._transform_view = self.createView(TransformView)

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
            self._corrected_view, "Overview")
        cali_idx = self._views_tab.addTab(
            self._calibration_view, "Gain / offset")
        ref_idx = self._views_tab.addTab(self._reference_view, "Reference")
        azimuthal_integ_tab_idx = self._views_tab.addTab(
            self._azimuthal_integ_1d_view, "Azimuthal integration 1D")
        geom_idx = self._views_tab.addTab(self._geometry_view, "Geometry")
        if not self._require_geometry:
            self._views_tab.setTabEnabled(geom_idx, False)
        transform_idx = self._views_tab.addTab(self._transform_view, "Image transform")

        assert(corrected_tab_idx == self.TabIndex.OVERVIEW)
        assert(cali_idx == self.TabIndex.GAIN_OFFSET)
        assert(ref_idx == self.TabIndex.REFERENCE)
        assert(azimuthal_integ_tab_idx == self.TabIndex.AZIMUTHAL_INTEG_1D)
        assert(geom_idx == self.TabIndex.GEOMETRY)
        assert(transform_idx == self.TabIndex.IMAGE_TRANSFORM)

        ctrl_panel = QWidget()
        ctrl_panel_layout = QVBoxLayout()
        ctrl_panel_layout.addWidget(self._bulletin_view)
        ctrl_panel_layout.addWidget(self._image_ctrl_widget)
        ctrl_panel_layout.addWidget(self._mask_ctrl_widget)
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

        self._image_ctrl_widget.update_image_btn.clicked.connect(
            self.onUpdateWidgets)
        self._image_ctrl_widget.auto_level_btn.clicked.connect(
            mediator.reset_image_level_sgn)
        self._image_ctrl_widget.save_image_btn.clicked.connect(
            self._corrected_view.onSaveImage)

        self._mask_ctrl_widget.draw_mask_btn.toggled.connect(
            self._corrected_view.onDrawMask)
        self._mask_ctrl_widget.erase_mask_btn.toggled.connect(
            self._corrected_view.onEraseMask)
        self._mask_ctrl_widget.remove_btn.clicked.connect(
            self._corrected_view.onRemoveMask)
        self._mask_ctrl_widget.load_btn.clicked.connect(
            self._corrected_view.onLoadMask)
        self._mask_ctrl_widget.save_btn.clicked.connect(
            self._corrected_view.onSaveMask)
        self._mask_ctrl_widget.mask_save_in_modules_cb.toggled.connect(
            self._corrected_view.onMaskSaveInModulesChange)

        self._views_tab.tabBarClicked.connect(self.onViewsTabClicked)
        self._views_tab.currentChanged.connect(self.onViewsTabChanged)

    def onStart(self):
        for widget in self._ctrl_widgets:
            widget.onStart()

    def onStop(self):
        for widget in self._ctrl_widgets:
            widget.onStop()

    def createView(self, view_class):
        return view_class(pulse_resolved=self._pulse_resolved, parent=self)

    def createCtrlWidget(self, widget_class, *args, **kwargs):
        """Register a ctrl widget.

        Ctrl widgets reside in (views of) ImageToolWindow should explicitly
        call this method to be registered.
        """
        widget = widget_class(*args,
                              pulse_resolved=self._pulse_resolved,
                              require_geometry=self._require_geometry,
                              **kwargs)
        self._ctrl_widgets.append(widget)
        return widget

    def updateMetaData(self):
        """Update metadata from all the ctrl widgets.

        :returns bool: True if all metadata successfully parsed
            and emitted, otherwise False.
        """
        for widget in self._ctrl_widgets:
            if not widget.updateMetaData():
                return False
        return True

    def loadMetaData(self):
        """Load metadata from Redis and set child control widgets."""
        for widget in self._ctrl_widgets:
            widget.loadMetaData()

    def reset(self):
        """Override."""
        pass

    @pyqtSlot()
    def onUpdateWidgets(self):
        """Used for updating manually."""
        self.updateWidgets(True)

    def updateWidgetsF(self):
        """Override."""
        self.updateWidgets(self._auto_update)

    def updateWidgets(self, auto_update):
        if len(self._queue) == 0:
            return
        data = self._queue[0]

        # update bulletin
        self._bulletin_view.updateF(data, auto_update)
        # update other ImageView/PlotWidget in the activated tab
        self._views_tab.currentWidget().updateF(data, auto_update)

    @pyqtSlot(int)
    def onViewsTabClicked(self, idx):
        if self._views_tab.currentIndex() == idx:
            return
        self._views_tab.currentWidget().onDeactivated()

    @pyqtSlot(int)
    def onViewsTabChanged(self, idx):
        self._views_tab.currentWidget().onActivated()
        self.updateWidgets(True)  # force update

        self._mask_ctrl_widget.setInteractiveButtonsEnabled(
            self._views_tab.currentIndex() == self.TabIndex.OVERVIEW)

    @pyqtSlot(bool)
    def onAutoUpdateToggled(self, state):
        self._auto_update = state
