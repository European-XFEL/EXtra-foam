"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

ImageToolWindow.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from ..widgets.pyqtgraph import QtGui

from ..widgets import ImageView
from .base_window import AbstractWindow, SingletonWindow
from ..logger import logger


class ROICtrlWidget(QtGui.QGroupBox):

    def __init__(self, title, *, parent=None):
        """"""
        super().__init__(title, parent=parent)

        self._roi_width_le = QtGui.QLineEdit()
        self._roi_height_le = QtGui.QLineEdit()

        self._lock_cb = QtGui.QCheckBox("Lock")
        self._activate_cb = QtGui.QCheckBox("Activate")

        self.initUI()

    def initUI(self):
        wh_layout = QtGui.QHBoxLayout()
        wh_layout.addWidget(QtGui.QLabel("Width: "))
        wh_layout.addWidget(self._roi_width_le)
        wh_layout.addWidget(QtGui.QLabel("Height: "))
        wh_layout.addWidget(self._roi_height_le)

        cb_layout = QtGui.QHBoxLayout()
        cb_layout.addWidget(self._activate_cb)
        cb_layout.addWidget(self._lock_cb)

        layout = QtGui.QVBoxLayout()

        layout.addLayout(cb_layout)
        layout.addLayout(wh_layout)

        self.setLayout(layout)


class MaskCtrlWidget(QtGui.QGroupBox):

    def __init__(self, title, *, parent=None):
        """"""
        super().__init__(title, parent=parent)

        self._lock_cb = QtGui.QCheckBox("Lock")

        self.initUI()

    def initUI(self):

        layout = QtGui.QVBoxLayout()

        layout.addWidget(self._lock_cb)

        self.setLayout(layout)


@SingletonWindow
class ImageToolWindow(AbstractWindow):
    """ImageToolWindow class."""
    title = "image tool"

    def __init__(self, data, *, parent=None):
        super().__init__(data, parent=parent)

        self._image_view = ImageView()
        self._roi_ctrl1 = ROICtrlWidget("ROI 1")
        self._roi_ctrl2 = ROICtrlWidget("ROI 2")
        self._mask_panel = MaskCtrlWidget("Mask tool")

        self._update_image_btn = QtGui.QPushButton("Update image")
        self._update_image_btn.clicked.connect(self.updateImage)

        self.initUI()
        self.resize(800, 600)
        self.updateImage()

        logger.info("Open DrawMaskWindow")

    def initUI(self):
        """Override."""
        tool_layout = QtGui.QGridLayout()
        tool_layout.addWidget(self._roi_ctrl1, 0, 0, 1, 1)
        tool_layout.addWidget(self._roi_ctrl2, 1, 0, 1, 1)
        tool_layout.addWidget(self._mask_panel, 0, 1, 2, 1)

        layout = QtGui.QVBoxLayout()
        layout.addWidget(self._image_view)
        layout.addWidget(self._update_image_btn)
        layout.addLayout(tool_layout)

        self._cw.setLayout(layout)
        self._cw.layout().setContentsMargins(0, 0, 0, 0)

    def updateImage(self):
        """For updating image manually."""
        data = self._data.get()
        if data.empty():
            return

        self._image_view.setImage(data.image_mean)
