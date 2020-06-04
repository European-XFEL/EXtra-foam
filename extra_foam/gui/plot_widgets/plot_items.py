"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import numpy as np

from PyQt5.QtGui import QColor, QImage, QPainter, QPainterPath, QPicture
from PyQt5.QtCore import pyqtSignal, pyqtSlot, QPoint, QRectF, Qt

from .. import pyqtgraph as pg

from ..misc_widgets import FColor
from ...config import config, MaskState
from ...ipc import ImageMaskPub


class ImageItem(pg.ImageItem):
    """ImageItem with mouseHover event."""
    mouse_moved_sgn = pyqtSignal(int, int, float)  # (x, y, value)
    draw_started_sgn = pyqtSignal(int, int)  # (x, y)
    draw_region_changed_sgn = pyqtSignal(int, int)  # (x, y)
    draw_finished_sgn = pyqtSignal()

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
        if self.drawing and ev.button() == Qt.LeftButton:
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
        if self.drawing and ev.button() == Qt.LeftButton:
            ev.accept()
            self.draw_finished_sgn.emit()
        else:
            ev.ignore()


class MaskItem(pg.GraphicsObject):
    """Mask item used for drawing mask on an ImageItem."""

    _TRANSPARENT = QColor(0, 0, 0, 0)
    _COLOR_FORMAT = QImage.Format_ARGB32

    def __init__(self, item):
        """Initialization.

        :param ImageItem item: a reference to the masked image item.
        """
        super().__init__()
        if not isinstance(item, ImageItem):
            raise TypeError("Input item must be an ImageItem instance.")

        self._image_item = item
        item.draw_started_sgn.connect(self.onDrawStarted)
        item.draw_region_changed_sgn.connect(self.onDrawRegionChanged)
        item.draw_finished_sgn.connect(self.onDrawFinished)

        # pen for drawing the bounding box
        self._boundary_color = FColor.mkPen(
            config['GUI_MASK_BOUNDING_BOX_COLOR'])
        self._fill_color = FColor.mkColor(
            config['GUI_MASK_FILL_COLOR'], alpha=180)

        self.state = MaskState.UNMASK
        self._mask = None  # QImage
        self._mask_rect = QRectF(0, 0, 0, 0)
        self._mask_pub = ImageMaskPub()

        self._p1 = None
        self._p2 = None

    def maybeInitializeMask(self, shape):
        h, w = shape
        if self._mask is None:
            self._mask = QImage(w, h, self._COLOR_FORMAT)
            self._mask.fill(self._TRANSPARENT)
            self._mask_rect = QRectF(0, 0, w, h)

    def boundingRect(self):
        """Override."""
        return self._mask_rect

    @pyqtSlot(int, int)
    def onDrawStarted(self, x, y):
        self._p1 = (x, y)

    @pyqtSlot(int, int)
    def onDrawRegionChanged(self, x, y):
        self.prepareGeometryChange()
        self._p2 = (x, y)

    def _selectedRect(self):
        if self._p1 is None or self._p2 is None:
            return QRectF(0, 0, 0, 0)

        rect = QRectF(QPoint(*self._p1), QPoint(*self._p2))
        return rect.intersected(self._mask_rect)

    @pyqtSlot()
    def onDrawFinished(self):
        rect = self._selectedRect()
        x = int(rect.x())
        y = int(rect.y())
        w = int(rect.width())
        h = int(rect.height())

        if self.state == MaskState.MASK:
            self._mask_pub.draw((x, y, w, h))
        elif self.state == MaskState.UNMASK:
            self._mask_pub.erase((x, y, w, h))

        self._p1 = None
        self._p2 = None

        for i in range(x, x+w):
            for j in range(y, y+h):
                if self.state == MaskState.MASK:
                    self._mask.setPixelColor(i, j, self._fill_color)
                elif self.state == MaskState.UNMASK:
                    self._mask.setPixelColor(i, j, self._TRANSPARENT)

        self._image_item.update()

    def removeMask(self):
        """Completely remove the current mask."""
        # The following sequence ensures that it can be used to rescue if
        # there is discrepancy between ImageTool and the ImageProcessor.

        self._mask_pub.remove()

        if self._mask is None:
            return

        self._mask.fill(self._TRANSPARENT)
        self._image_item.update()

    def paint(self, p, *args):
        if self._mask is None:
            return

        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(self._boundary_color)

        p.drawImage(self.boundingRect(), self._mask)
        p.drawRect(self._selectedRect())

    def mask(self):
        w = self._mask.width()
        h = self._mask.height()

        mask_array = np.zeros((h, w), dtype=bool)
        for i in range(w):
            for j in range(h):
                mask_array[j, i] = \
                    self._mask.pixelColor(i, j) == self._fill_color

        return mask_array

    def setMask(self, mask):
        """Set a given image mask.

        :param np.ndarray mask: mask in ndarray. shape = (h, w)
        """
        self._mask_pub.set(mask)

        h, w = mask.shape
        self._mask = QImage(w, h, self._COLOR_FORMAT)

        for i in range(w):
            for j in range(h):
                if mask[j, i]:
                    self._mask.setPixelColor(i, j, self._fill_color)
                else:
                    self._mask.setPixelColor(i, j, self._TRANSPARENT)
        self._mask_rect = QRectF(0, 0, w, h)
        self._image_item.update()


class RectROI(pg.ROI):
    """Rectangular ROI widget.

    Note: the widget is slightly different from pyqtgraph.RectROI
    """
    def __init__(self, idx, *, pos=(0, 0), size=(1, 1), **kwargs):
        """Initialization.

        :param int idx: index of the ROI.
        :param tuple pos: (x, y) of the left-upper corner.
        :param tuple size: (w, h) of the ROI.
        """
        super().__init__(pos, size,
                         translateSnap=True,
                         scaleSnap=True, **kwargs)

        self._index = idx

    @property
    def index(self):
        return self._index

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


class CurvePlotItem(pg.GraphicsObject):
    """CurvePlotItem."""

    def __init__(self, x=None, y=None, *, pen=None):
        """Initialization."""
        super().__init__()

        self._path = None

        self._x = None
        self._y = None

        self._pen = FColor.mkPen('g') if pen is None else pen

        self.setData(x, y)

    def setData(self, x, y):
        """PlotItem interface."""
        self._x = [] if x is None else x
        self._y = [] if y is None else y

        if len(self._x) != len(self._y):
            raise ValueError("'x' and 'y' data have different lengths!")

        self._path = None
        # Schedules a redraw of the area covered by rect in this item.
        self.update()
        self.informViewBoundsChanged()

    def preparePath(self):
        p = QPainterPath()

        x, y = self._x, self._y
        if len(x) >= 2:
            p.moveTo(x[0], y[0])
            for px, py in zip(x[1:], y[1:]):
                p.lineTo(px, py)

        self._path = p
        # Prepares the item for a geometry change.
        self.prepareGeometryChange()

    def paint(self, painter, *args):
        """Override."""
        if self._path is None:
            self.preparePath()

        painter.setPen(self._pen)
        painter.drawPath(self._path)

    def boundingRect(self):
        """Override."""
        if self._path is None:
            self.preparePath()
        return self._path.boundingRect()


class BarGraphItem(pg.GraphicsObject):
    """BarGraphItem"""
    def __init__(self, x=None, y=None, *, width=1.0, pen=None, brush=None):
        """Initialization."""
        super().__init__()

        self._picture = None

        self._x = None
        self._y = None

        if width > 1.0 or width <= 0:
            width = 1.0
        self._width = width

        if pen is None and brush is None:
            self._pen = FColor.mkPen(None)
            self._brush = FColor.mkBrush('b')
        else:
            self._pen = FColor.mkPen(None) if pen is None else pen
            self._brush = FColor.mkBrush(None) if brush is None else brush

        self.setData(x, y)

    def setData(self, x, y):
        """PlotItem interface."""
        self._x = [] if x is None else x
        self._y = [] if y is None else y

        if len(self._x) != len(self._y):
            raise ValueError("'x' and 'y' data have different lengths!")

        self._picture = None
        self.update()
        self.informViewBoundsChanged()

    def drawPicture(self):
        self._picture = QPicture()
        p = QPainter(self._picture)
        p.setPen(self._pen)
        p.setBrush(self._brush)

        # Now it works for bar plot with equalized gaps
        # TODO: extend it
        if len(self._x) > 1:
            width = self._width * (self._x[1] - self._x[0])
        else:
            width = self._width

        for x, y in zip(self._x, self._y):
            p.drawRect(QRectF(x - width/2, 0, width, y))

        p.end()
        self.prepareGeometryChange()

    def paint(self, painter, *args):
        """Override."""
        if self._picture is None:
            self.drawPicture()
        self._picture.play(painter)

    def boundingRect(self):
        """Override."""
        if self._picture is None:
            self.drawPicture()
        return QRectF(self._picture.boundingRect())


class StatisticsBarItem(pg.GraphicsObject):
    """StatisticsBarItem."""

    def __init__(self, x=None, y=None, *, y_min=None, y_max=None, beam=None,
                 line=False, pen=None):
        """Initialization.

        Note: y is not used for now.
        """
        super().__init__()

        self._path = None

        self._x = None
        self._y = None
        self._y_min = None
        self._y_max = None

        self._beam = 0.0 if beam is None else beam
        self._line = line
        self._pen = FColor.mkPen('p') if pen is None else pen

        self.setData(x, y, y_min=y_min, y_max=y_max)

    def setData(self, x, y, y_min=None, y_max=None, beam=None):
        """PlotItem interface."""
        self._x = [] if x is None else x
        self._y = [] if y is None else y

        self._y_min = self._y if y_min is None else y_min
        self._y_max = self._y if y_max is None else y_max
        if beam is not None:
            # keep the default beam if not specified
            self._beam = beam

        if len(self._x) != len(self._y):
            raise ValueError("'x' and 'y' data have different lengths!")
        if not len(self._y) == len(self._y_min) == len(self._y_max):
            raise ValueError(
                "'y_min' and 'y_max' data have different lengths!")
        self._path = None
        self.update()
        self.informViewBoundsChanged()

    def setBeam(self, w):
        self._beam = w

    def isEmptyGraph(self):
        return not bool(len(self._x))

    def preparePath(self):
        p = QPainterPath()

        beam = self._beam
        for x, u, l in zip(self._x, self._y_min, self._y_max):
            # plot the lower horizontal lines
            p.moveTo(x - beam / 2., l)
            p.lineTo(x + beam / 2., l)

            # plot the vertical line
            p.moveTo(x, l)
            p.lineTo(x, u)

            # plot the upper horizontal line
            p.moveTo(x - beam / 2., u)
            p.lineTo(x + beam / 2., u)

        if self._line and len(self._x) > 2:
            p.moveTo(self._x[-1], self._y[-1])
            for x, y in zip(reversed(self._x[:-1]), reversed(self._y[:-1])):
                p.lineTo(x, y)

        self._path = p
        self.prepareGeometryChange()

    def paint(self, painter, *args):
        """Override."""
        if self._path is None:
            self.preparePath()

        painter.setPen(self._pen)
        painter.drawPath(self._path)

    def boundingRect(self):
        """Override."""
        if self._path is None:
            self.preparePath()
        return self._path.boundingRect()
