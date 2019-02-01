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


@SingletonWindow
class ImageToolWindow(AbstractWindow):
    """ImageToolWindow class."""
    title = "image tool"

    def __init__(self, data, *, parent=None):
        super().__init__(data, parent=parent)

        self._image_view = ImageView()

        self._update_image_btn = QtGui.QPushButton("Update image")
        self._update_image_btn.clicked.connect(self.updateImage)

        self.initUI()

        self.updateImage()

        logger.info("Open DrawMaskWindow")

    def initUI(self):
        """Override."""
        self.initPlotUI()

        layout = QtGui.QHBoxLayout()
        layout.addWidget(self._image_view)
        layout.addWidget(self._update_image_btn)
        self._cw.setLayout(layout)

    def initPlotUI(self):
        """Override."""
        pass

    def initCtrlUI(self):
        """Override."""
        pass

    def updateImage(self):
        """For updating image manually."""
        data = self._data.get()
        if data.empty():
            return

        self._image_view.setImage(data.image_mean)
