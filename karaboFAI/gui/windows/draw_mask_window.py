"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

DrawMaskWindow.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import silx
from silx.gui.plot.MaskToolsWidget import MaskToolsWidget
from silx.gui.colors import Colormap as SilxColormap

from ..pyqtgraph import QtGui

from .base_window import AbstractWindow, SingletonWindow
from ...logger import logger


@SingletonWindow
class DrawMaskWindow(AbstractWindow):
    """DrawMaskWindow class.

    A window which allows users to have a better visualization of the
    detector image and draw a mask for further azimuthal integration.
    The mask must be saved and then loaded in the main GUI manually.
    """
    title = "draw mask"

    def __init__(self, data, *, parent=None):
        super().__init__(data, parent=parent)

        self._image = silx.gui.plot.Plot2D()
        self._mask_panel = MaskToolsWidget(plot=self._image)

        self.initUI()
        self._updateImage()

        logger.info("Open DrawMaskWindow")

    def initUI(self):
        """Override."""
        self.initPlotUI()

        layout = QtGui.QHBoxLayout()
        layout.addWidget(self._image)
        layout.setStretch(0, 1)
        layout.addLayout(self.initCtrlUI())
        self._cw.setLayout(layout)

    def initPlotUI(self):
        """Override."""
        self._image.setKeepDataAspectRatio(True)
        self._image.setYAxisInverted(True)
        # normalization options: LINEAR or LOGARITHM
        self._image.setDefaultColormap(
            SilxColormap('viridis', normalization=SilxColormap.LINEAR))

    def initCtrlUI(self):
        """Override."""
        self._image.getMaskAction().setVisible(False)

        self._mask_panel.setDirection(QtGui.QBoxLayout.TopToBottom)
        self._mask_panel.setMultipleMasks("single")

        update_image_btn = QtGui.QPushButton("Update image")
        update_image_btn.clicked.connect(self._updateImage)
        update_image_btn.setMinimumHeight(60)

        ctrl_widget = QtGui.QVBoxLayout()
        ctrl_widget.addWidget(self._mask_panel)
        ctrl_widget.addWidget(update_image_btn)
        ctrl_widget.addStretch(1)

        return ctrl_widget

    def _updateImage(self):
        """For updating image manually."""
        data = self._data.get()
        if data.image is None:
            return

        # TODO: apply the mask to data processing on the fly!
        # self._mask_panel.getSelectionMask()

        self._image.addImage(data.image.masked_mean)
