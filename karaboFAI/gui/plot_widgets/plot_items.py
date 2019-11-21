"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import numpy as np

from PyQt5.QtGui import QColor, QImage, QPainter, QPainterPath, QPicture
from PyQt5.QtCore import pyqtSignal, pyqtSlot, QPoint, QRectF, Qt

from .. import pyqtgraph as pg

from ..misc_widgets import make_brush, make_pen
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

    _mask = None  # QImage
    _mask_rect = QRectF(0, 0, 0, 0)

    _TRANSPARENT = QColor(0, 0, 0, 0)
    _OPAQUE = QColor(0, 0, 0, 255)

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
        self._pen = make_pen(config['MASK_BOUNDING_BOX_COLOR'])

        self.state = MaskState.UNMASK
        self._mask_pub = ImageMaskPub()

        self._p1 = None
        self._p2 = None

    @classmethod
    def resetMask(cls):
        cls._mask = None
        cls._mask_rect = QRectF(0, 0, 0, 0)

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
            self._mask_pub.add((x, y, w, h))
        elif self.state == MaskState.UNMASK:
            self._mask_pub.erase((x, y, w, h))

        self._p1 = None
        self._p2 = None

        for i in range(x, x+w):
            for j in range(y, y+h):
                if self.state == MaskState.MASK:
                    self._mask.setPixelColor(i, j, self._OPAQUE)
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

    def onSetImage(self):
        h, w = self._image_item.image.shape
        if self._mask is None:
            self.__class__._mask = QImage(w, h, QImage.Format_Alpha8)
            self._mask.fill(self._TRANSPARENT)
            self.__class__._mask_rect = QRectF(0, 0, w, h)

    def paint(self, p, *args):
        if self._mask is None:
            return

        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(self._pen)

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

    def loadMask(self, mask):
        """Load a given image mask.

        :param np.ndarray mask: mask in ndarray. shape = (h, w)
        """
        self._mask_pub.set(mask)

        h, w = mask.shape
        self.__class__._mask = QImage(w, h, QImage.Format_Alpha8)

        for i in range(w):
            for j in range(h):
                if mask[j, i]:
                    self._mask.setPixelColor(i, j, self._OPAQUE)
                else:
                    self._mask.setPixelColor(i, j, self._TRANSPARENT)
        self.__class__._mask_rect = QRectF(0, 0, w, h)
        self._image_item.update()


class RectROI(pg.ROI):
    """Rectangular ROI widget.

    Note: the widget is slightly different from pyqtgraph.RectROI
    """
    def __init__(self, rank, *, pos=(0, 0), size=(1, 1), **kwargs):
        """Initialization.

        :param int rank: rank of the ROI.
        :param tuple pos: (x, y) of the left-upper corner.
        :param tuple size: (w, h) of the ROI.
        """
        super().__init__(pos, size,
                         translateSnap=True,
                         scaleSnap=True, **kwargs)

        self.rank = rank

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


class BarPlotItem(pg.GraphicsObject):
    """_BarPlotItem"""
    def __init__(self, x=None, y=None, width=1.0, pen=None, brush=None):
        """Initialization."""
        super().__init__()

        self._picture = None

        self._x = None
        self._y = None

        self._width = None
        self.width = width

        self._pen = make_pen('g') if pen is None else pen
        self._brush = make_brush('b') if brush is None else brush

        self.setData(x, y)

    def setData(self, x, y, width=None, pen=None):
        """PlotItem interface."""
        self._x = [] if x is None else x
        self._y = [] if y is None else y

        if len(self._x) != len(self._y):
            raise ValueError("'x' and 'y' data have different lengths!")

        self.width = width

        if pen is not None:
            self._pen = pen

        self._picture = None
        self.update()
        self.prepareGeometryChange()
        self.informViewBoundsChanged()

    @property
    def width(self):
        return self._width

    @width.setter
    def width(self, value):
        if value is None:
            return

        if value > 1.0 or value <= 0:
            value = 1.0  # set to default
        self._width = value

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
            rect = QRectF(x - width/2, 0, width, y)
            p.drawRect(rect)

        p.end()
        self.prepareGeometryChange()

    def paint(self, p, *args):
        if self._picture is None:
            self.drawPicture()
        self._picture.play(p)

    def boundingRect(self):
        if self._picture is None:
            self.drawPicture()
        return QRectF(self._picture.boundingRect())


class ErrorBarItem(pg.GraphicsObject):
    """ErrorBarItem."""

    def __init__(self,
                 x=None,
                 y=None,
                 y_min=None,
                 y_max=None,
                 beam=None,
                 pen=None):
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
        self._pen = make_pen('b') if pen is None else pen

        self.setData(x, y, y_min=y_min, y_max=y_max)

    def setData(self, x, y, y_min=None, y_max=None, beam=None, pen=None):
        """PlotItem interface."""
        self._x = [] if x is None else x
        self._y = [] if y is None else y

        self._y_min = self._y if y_min is None else y_min
        self._y_max = self._y if y_max is None else y_max

        if len(self._x) != len(self._y):
            raise ValueError("'x' and 'y' data have different lengths!")
        if not len(self._y) == len(self._y_min) == len(self._y_max):
            raise ValueError(
                "'y_min' and 'y_max' data have different lengths!")

        if beam is not None and beam >= 0.0:
            self._beam = beam
        if pen is not None:
            self._pen = pen

        self._path = None
        self.update()
        self.prepareGeometryChange()
        self.informViewBoundsChanged()

    def drawPath(self):
        p = QPainterPath()

        x = self._x

        for i in range(len(x)):
            # plot the lower horizontal lines
            p.moveTo(x[i] - self._beam / 2., self._y_min[i])
            p.lineTo(x[i] + self._beam / 2., self._y_min[i])

            # plot the vertical line
            p.moveTo(x[i], self._y_min[i])
            p.lineTo(x[i], self._y_max[i])

            # plot the upper horizontal line
            p.moveTo(x[i] - self._beam / 2., self._y_max[i])
            p.lineTo(x[i] + self._beam / 2., self._y_max[i])

        self._path = p
        self.prepareGeometryChange()

    def paint(self, p, *args):
        if self._path is None:
            self.drawPath()

        p.setPen(self._pen)
        p.drawPath(self._path)

    def boundingRect(self):
        if self._path is None:
            self.drawPath()
        return self._path.boundingRect()
