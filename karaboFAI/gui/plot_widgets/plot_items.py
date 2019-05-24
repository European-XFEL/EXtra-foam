"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

Various PlotItems.

ImageItem, MaskItem.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import numpy as np

from .. import pyqtgraph as pg
from ..pyqtgraph import GraphicsObject, QtCore, QtGui

from ..misc_widgets import make_pen, make_brush
from ...config import ImageMaskChange


class ImageItem(pg.ImageItem):
    """ImageItem with mouseHover event."""
    mouse_moved_sgn = QtCore.pyqtSignal(int, int, float)  # (x, y, value)
    draw_started_sgn = QtCore.pyqtSignal(int, int)  # (x, y)
    draw_region_changed_sgn = QtCore.pyqtSignal(int, int)  # (x, y)
    draw_finished_sgn = QtCore.pyqtSignal()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.drawing = False

    def hoverEvent(self, ev):
        """Override."""
        if ev.isExit():
            x = -1  # out of image
            y = -1  # out of image
            value = 0.0
        else:
            pos = ev.pos()
            x = int(pos.x())
            y = int(pos.y())
            value = self.image[y, x]

        self.mouse_moved_sgn.emit(x, y, value)

    def mousePressEvent(self, ev):
        """Override."""
        if self.drawing and ev.button() == QtCore.Qt.LeftButton:
            ev.accept()
            pos = ev.pos()
            self.draw_started_sgn.emit(int(pos.x()), int(pos.y()))
        else:
            ev.ignore()

    def mouseMoveEvent(self, ev):
        """Override."""
        if self.drawing:
            ev.accept()
            pos = ev.pos()
            self.draw_region_changed_sgn.emit(int(pos.x()), int(pos.y()))
        else:
            ev.ignore()

    def mouseReleaseEvent(self, ev):
        """Override."""
        if self.drawing and ev.button() == QtCore.Qt.LeftButton:
            ev.accept()
            self.draw_finished_sgn.emit()
        else:
            ev.ignore()


class MaskItem(GraphicsObject):
    """Mask item used for drawing mask on an ImageItem."""
    # ImageMaskChange, x, y, w, h
    mask_region_change_sgn = QtCore.Signal(object, int, int, int, int)

    _mask = None  # QImage
    _mask_rect = QtCore.QRectF(0, 0, 0, 0)

    _TRANSPARENT = QtGui.QColor(0, 0, 0, 0)
    _OPAQUE = QtGui.QColor(0, 0, 0, 255)

    def __init__(self, item):
        """Initialization.

        :param ImageItem item: a reference to the masked image.
        """
        super().__init__()
        if not isinstance(item, pg.ImageItem):
            raise TypeError("Input item must be an ImageItem instance.")
        self._image_item = item
        if isinstance(item, ImageItem):
            item.draw_started_sgn.connect(self.onDrawStarted)
            item.draw_region_changed_sgn.connect(self.onDrawRegionChanged)
            item.draw_finished_sgn.connect(self.onDrawFinished)

        self._brush = make_brush('b')  # brush for drawing the bounding box

        self.draw_type = None

        self._p1 = None
        self._p2 = None

    def boundingRect(self):
        """Override."""
        return self._mask_rect

    @QtCore.pyqtSlot(int, int)
    def onDrawStarted(self, x, y):
        self._p1 = (x, y)

    @QtCore.pyqtSlot(int, int)
    def onDrawRegionChanged(self, x, y):
        self.prepareGeometryChange()
        self._p2 = (x, y)

    def _selectedRect(self):
        if self._p1 is None or self._p2 is None:
            return QtCore.QRectF(0, 0, 0, 0)

        rect = QtCore.QRectF(QtCore.QPoint(*self._p1), QtCore.QPoint(*self._p2))
        return rect.intersected(self._mask_rect)

    @QtCore.pyqtSlot()
    def onDrawFinished(self):
        rect = self._selectedRect()
        x = int(rect.x())
        y = int(rect.y())
        w = int(rect.width())
        h = int(rect.height())
        self.mask_region_change_sgn.emit(self.draw_type, x, y, w, h)

        self._p1 = None
        self._p2 = None

        for i in range(x, x+w):
            for j in range(y, y+h):
                if self.draw_type == ImageMaskChange.MASK:
                    self._mask.setPixelColor(i, j, self._OPAQUE)
                else:
                    self._mask.setPixelColor(i, j, self._TRANSPARENT)

        self._image_item.update()

    def clear(self):
        if self._mask is None:
            return

        self.mask_region_change_sgn.emit(ImageMaskChange.CLEAR, 0, 0, 0, 0)

        self._mask.fill(self._TRANSPARENT)
        self._image_item.update()

    def updateImage(self):
        h, w = self._image_item.image.shape
        if self._mask is None:
            self.__class__._mask = QtGui.QImage(w, h,
                                                QtGui.QImage.Format_Alpha8)
            self._mask.fill(self._TRANSPARENT)
            self.__class__._mask_rect = QtCore.QRectF(0, 0, w, h)

    def paint(self, p, *args):
        if self._mask is None:
            return

        p.setRenderHint(QtGui.QPainter.Antialiasing)
        p.setPen(make_pen('b', width=4))

        p.drawImage(self.boundingRect(), self._mask)
        p.drawRect(self._selectedRect())

    def toNDArray(self):
        w = self._mask.width()
        h = self._mask.height()

        mask_array = np.zeros((h, w), dtype=bool)
        for i in range(w):
            for j in range(h):
                mask_array[j, i] = self._mask.pixelColor(i, j) == self._OPAQUE

        return mask_array

    def updateMask(self, mask):
        """Update the image mask.

        :param np.ndarray mask: mask in ndarray. shape = (h, w)
        """
        h, w = mask.shape
        self.__class__._mask = QtGui.QImage(w, h, QtGui.QImage.Format_Alpha8)

        for i in range(w):
            for j in range(h):
                if mask[j, i]:
                    self._mask.setPixelColor(i, j, self._OPAQUE)
                else:
                    self._mask.setPixelColor(i, j, self._TRANSPARENT)
        self.__class__._mask_rect = QtCore.QRectF(0, 0, w, h)
        self._image_item.update()
