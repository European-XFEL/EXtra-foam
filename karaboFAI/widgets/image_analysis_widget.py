"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

ImageAnalysisWidget.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from ..widgets.pyqtgraph import ImageView

from .misc_widgets import  colorMapFactory
from ..config import config


class ImageAnalysisWidget(ImageView):
    """ImageAnalysisWidget class.

    Widget used for displaying and analyzing the assembled image.
    TODO: the ImageView widget needs to be customized.
    """
    def __init__(self, *, parent=None):
        """Initialization."""
        super().__init__(parent=parent)

        self._is_initialized = False
        # TODO: logarithmic level
        self.setColorMap(colorMapFactory[config["COLOR_MAP"]])

    def update(self, data):
        self.setImage(data.image_mean, autoRange=False,
                      autoLevels=(not self._is_initialized))

        if not self._is_initialized:
            self._is_initialized = True
