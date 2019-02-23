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
    def __init__(self, pos, size, *, lock=True, **args):
        """Initialization.

        :param bool lock: whether the ROI is modifiable.
        """
        super().__init__(pos, size, translateSnap=True, scaleSnap=True, **args)

        if lock:
            self.translatable = False
        else:
            self._add_handle()
            self._handle_info = self.handles[0]  # there is only one handler

    def lockAspect(self):
        self._handle_info['lockAspect'] = True

    def unLockAspect(self):
        self._handle_info['lockAspect'] = False

    def lock(self):
        self.translatable = False
        self.removeHandle(0)
        self._handle_info = None

    def unLock(self):
        self.translatable = True
        self._add_handle()
        self._handle_info = self.handles[0]

    def _add_handle(self):
        """An alternative to addHandle in parent class."""
        # position, scaling center
        self.addScaleHandle([1, 1], [0, 0])


class CropROI(ROI):
    """Rectangular cropping widget."""
    def __init__(self, pos, size):
        """Initialization."""
        super().__init__(pos, size,
                         translateSnap=True,
                         scaleSnap=True,
                         pen=make_pen('y', width=1, style=QtCore.Qt.DashDotLine))

        self._add_handles()

    def _add_handles(self):
        # position, scaling center
        self.addScaleHandle([1, 1], [0, 0])
        self.addScaleHandle([1, 0.5], [0, 0.5])
        self.addScaleHandle([0.5, 1], [0.5, 0])
        self.addScaleHandle([0, 0.5], [1, 0.5])
        self.addScaleHandle([0.5, 0], [0.5, 1])