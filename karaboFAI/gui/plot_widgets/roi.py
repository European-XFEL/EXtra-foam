"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

Region of interest (ROI) widgets.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from ..pyqtgraph import QtCore, ROI

from ..misc_widgets import make_pen


class RectROI(ROI):
    """Rectangular ROI widget.

    Note: the widget is slightly different from pyqtgraph.RectROI
    """
    def __init__(self, rank, color, pos, size, *, style=QtCore.Qt.SolidLine):
        """Initialization.

        :param int rank: rank of the ROI.
        :param str color: color of the ROI.
        """
        super().__init__(pos, size,
                         translateSnap=True,
                         scaleSnap=True,
                         pen=make_pen(color, style=style))

        self.rank = rank
        self.color = color

    def setLocked(self, locked):
        if locked:
            self.translatable = False
            self.removeHandle(0)
            self._handle_info = None
        else:
            self.translatable = True
            self._addHandle()
            self._handle_info = self.handles[0]

    def _addHandle(self):
        """An alternative to addHandle in parent class."""
        # position, scaling center
        self.addScaleHandle([1, 1], [0, 0])
