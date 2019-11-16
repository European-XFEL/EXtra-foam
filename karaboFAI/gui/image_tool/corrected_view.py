"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QColor, QIntValidator
from PyQt5.QtWidgets import (
    QCheckBox, QHBoxLayout, QLabel, QVBoxLayout, QWidget
)

from .simple_image_data import _SimpleImageData
from .base_view import _AbstractImageToolView
from ..mediator import Mediator
from ..plot_widgets import ImageAnalysis
from ..ctrl_widgets import Projection1DCtrlWidget, SmartLineEdit
from ..misc_widgets import Colors
from ...config import config


class _RoiCtrlWidgetBase(QWidget):
    """_RoiCtrlWidgetBase class.

    Base class for RoiCtrlWidget. Implemented four QLineEdits (x, y, w, h)
    and their connection to the corresponding ROI.
    """
    # (rank, x, y, w, h) where rank starts from 1
    roi_geometry_change_sgn = pyqtSignal(object)
    # (rank, visible)
    roi_visibility_change_sgn = pyqtSignal(object)

    _pos_validator = QIntValidator(-10000, 10000)
    _size_validator = QIntValidator(1, 10000)

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

    @pyqtSlot()
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

    @pyqtSlot()
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

    @pyqtSlot(object)
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
    """_SingleRoiCtrlWidget class.

    Widget for controlling of a single ROI.
    """
    def __init__(self, roi, *, parent=None):
        """Initialization.

        :param RectROI roi: RectROI object.
        """
        super().__init__(roi, parent=parent)

        self.activate_cb = QCheckBox("On")
        self.lock_cb = QCheckBox("Lock")

        self.initUI()

        self.disableAllEdit()

        self.activate_cb.stateChanged.connect(self.onToggleRoiActivation)
        self.activate_cb.stateChanged.emit(self.activate_cb.checkState())
        self.lock_cb.stateChanged.connect(self.onLock)

    def initUI(self):
        layout = QHBoxLayout()
        rank = self._roi.rank

        label = QLabel(f"ROI{rank}: ")
        palette = label.palette()
        # TODO: improve
        palette.setColor(palette.WindowText,
                         QColor(*getattr(Colors(), config['ROI_COLORS'][rank-1])))
        label.setPalette(palette)
        layout.addWidget(label)

        layout.addWidget(self.activate_cb)
        layout.addWidget(self.lock_cb)
        layout.addWidget(QLabel("w: "))
        layout.addWidget(self._width_le)
        layout.addWidget(QLabel("h: "))
        layout.addWidget(self._height_le)
        layout.addWidget(QLabel("x: "))
        layout.addWidget(self._px_le)
        layout.addWidget(QLabel("y: "))
        layout.addWidget(self._py_le)

        self.setLayout(layout)
        # left, top, right, bottom
        self.layout().setContentsMargins(2, 1, 2, 1)

    @pyqtSlot(int)
    def onToggleRoiActivation(self, state):
        activated = state == Qt.Checked
        if activated:
            self._roi.show()
            self.enableAllEdit()
        else:
            self._roi.hide()
            self.disableAllEdit()

        self.roi_visibility_change_sgn.emit((self._roi.rank, activated))

    @pyqtSlot(int)
    def onLock(self, state):
        self._roi.setLocked(state == Qt.Checked)
        self.setEditable(not state == Qt.Checked)

    def disableAllEdit(self):
        self.setEditable(False)
        self.lock_cb.setDisabled(True)

    def enableAllEdit(self):
        self.lock_cb.setDisabled(False)
        self.setEditable(True)


class CorrectedView(_AbstractImageToolView):
    """CorrectedView class.

    Widget for visualizing the corrected (masked, dark subtracted, etc.)
    image. ROI control widgets and 1D projection analysis control widget
    are included.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._image_view = ImageAnalysis(hide_axis=False, parent=self)

        self._proj1d_ctrl_widget = self.parent().createCtrlWidget(
            Projection1DCtrlWidget)

        self._roi_ctrls = []
        for roi in self._image_view.rois:
            widget = _SingleRoiCtrlWidget(roi)
            self._roi_ctrls.append(widget)

        self.initUI()
        self.initConnections()
        self.updateMetaData()

    def initUI(self):
        """Override."""
        roi_ctrl_layout = QVBoxLayout()
        for i, roi_ctrl in enumerate(self._roi_ctrls):
            roi_ctrl_layout.addWidget(roi_ctrl)

        ctrl_layout = QHBoxLayout()
        ctrl_layout.addLayout(roi_ctrl_layout)
        ctrl_layout.addWidget(self._proj1d_ctrl_widget)

        layout = QVBoxLayout()
        layout.addWidget(self._image_view)
        layout.addLayout(ctrl_layout)
        self.setLayout(layout)

    def initConnections(self):
        """Override."""
        mediator = Mediator()

        for widget in self._roi_ctrls:
            widget.roi_geometry_change_sgn.connect(mediator.onRoiGeometryChange)
            widget.roi_visibility_change_sgn.connect(
                mediator.onRoiVisibilityChange)

    def updateF(self, data, auto_update):
        """Override."""
        if auto_update or self._image_view.image is None:
            self._image_view.setImageData(_SimpleImageData(data.image))

    def updateMetaData(self):
        for i, widget in enumerate(self._roi_ctrls, 1):
            widget.notifyRoiParams()
            widget.roi_visibility_change_sgn.emit(
                (i, widget.activate_cb.checkState() == Qt.Checked))
