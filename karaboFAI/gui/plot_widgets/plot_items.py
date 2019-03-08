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

        self.image_mask = None

    def hoverEvent(self, ev):
        """Override."""
        if self.drawing or \
                (not ev.isExit() and ev.acceptDrags(QtCore.Qt.LeftButton)):
            # block events from other items
            ev.acceptClicks(QtCore.Qt.LeftButton)
            ev.acceptClicks(QtCore.Qt.RightButton)

        elif not ev.isExit() and self.removable:
            # accept context menu clicks
            ev.acceptClicks(QtCore.Qt.RightButton)

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
    image_mask_change_sgn = QtCore.Signal(object, int, int, int, int)

    _mask = None

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

        self._first_corner = None
        self._second_corner = None

    def boundingRect(self):
        """Override."""
        if self._first_corner is None or self._second_corner is None:
            return QtCore.QRectF(0, 0, 0, 0)

        rect = QtCore.QRectF(QtCore.QPoint(*self._first_corner),
                             QtCore.QPoint(*self._second_corner)).normalized()
        if self._mask is None:
            return rect

        s = self._mask.size()
        return rect.intersected(QtCore.QRectF(0, 0, s.width(), s.height()))

    @QtCore.pyqtSlot(int, int)
    def onDrawStarted(self, x, y):
        self._first_corner = (x, y)

    @QtCore.pyqtSlot(int, int)
    def onDrawRegionChanged(self, x, y):
        self.prepareGeometryChange()
        self._second_corner = (x, y)

    @QtCore.pyqtSlot()
    def onDrawFinished(self):
        rect = self.boundingRect()
        x1 = int(rect.x())
        y1 = int(rect.y())
        x2 = int(rect.x() + rect.width())
        y2 = int(rect.y() + rect.height())
        self.image_mask_change_sgn.emit(self.draw_type, x1, y1, x2, y2)

        self._first_corner = None
        self._second_corner = None

        # TODO: use C code
        for i in range(x1, x2):
            for j in range(y1, y2):
                if self.draw_type == ImageMaskChange.MASK:
                    self._mask.setPixelColor(i, j, self._OPAQUE)
                else:
                    self._mask.setPixelColor(i, j, self._TRANSPARENT)

        self._image_item.update()

    def clear(self):
        if self._mask is None:
            return

        self.image_mask_change_sgn.emit(ImageMaskChange.CLEAR, 0, 0, 0, 0)

        self._mask.fill(self._TRANSPARENT)
        self._image_item.update()

    def set(self):
        image = self._image_item.image
        if self._mask is None:
            self.__class__._mask = QtGui.QImage(*image.shape[::-1],
                                                QtGui.QImage.Format_Alpha8)
            self._mask.fill(self._TRANSPARENT)

    def paint(self, p, *args):
        if self._mask is None:
            return

        p.setRenderHint(QtGui.QPainter.Antialiasing)
        p.setPen(make_pen('c', width=4))

        s = self._mask.size()
        p.drawImage(QtCore.QRectF(0, 0, s.width(), s.height()), self._mask)

        p.drawRect(self.boundingRect())
