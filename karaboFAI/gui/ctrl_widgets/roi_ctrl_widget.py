"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

RoiCtrlWidget.

Author: Jun Zhu <jun.zhu@xfel.eu>, Ebad Kamil <ebad.kamil@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from ..pyqtgraph import QtCore, QtGui


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
        layout = QtGui.QHBoxLayout()
        layout.addWidget(self.activate_cb)
        layout.addWidget(QtGui.QLabel("w: "))
        layout.addWidget(self._width_le)
        layout.addWidget(QtGui.QLabel("h: "))
        layout.addWidget(self._height_le)
        layout.addWidget(QtGui.QLabel("x0: "))
        layout.addWidget(self._px_le)
        layout.addWidget(QtGui.QLabel("y0: "))
        layout.addWidget(self._py_le)
        layout.addWidget(self.lock_aspect_cb)
        layout.addWidget(self.lock_cb)

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