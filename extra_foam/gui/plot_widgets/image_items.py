"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import abc
from collections.abc import Callable
import weakref

import numpy as np

from PyQt5.QtGui import QColor, QImage, QPainter, QPicture, QTransform, QFont, QPen
from PyQt5.QtCore import (
    pyqtSignal, pyqtSlot, QPoint, QPointF, QRectF, Qt
)

from .. import pyqtgraph as pg
from ..pyqtgraph import Point
from ..pyqtgraph import functions as fn

from ..misc_widgets import FColor
from ...algorithms import quick_min_max
from ...config import config, MaskState
from ...ipc import ImageMaskPub


class ImageItem(pg.GraphicsObject):
    """GraphicsObject displaying a 2D image.

    Implemented based on pyqtgraph.ImageItem.
    """

    image_changed_sgn = pyqtSignal()

    mouse_moved_sgn = pyqtSignal(int, int, float)  # (x, y, value)
    draw_started_sgn = pyqtSignal(int, int)  # (x, y)
    draw_region_changed_sgn = pyqtSignal(int, int)  # (x, y)
    draw_finished_sgn = pyqtSignal()

    def __init__(self, image=None, parent=None):
        super().__init__(parent=parent)

        self._image = None   # original image data
        self._qimage = None  # rendered image for display

        self._levels = None  # [min, max]
        self._auto_level_quantile = 0.99
        self._lut = None
        self._ds_rate = (1., 1.)  # down-sample rates

        # In some cases, a modified lookup table is used to handle both
        # rescaling and LUT more efficiently
        self._fast_lut = None

        self.setImage(image, auto_levels=True)

        self.drawing = False

    def width(self):
        if self._image is None:
            return None
        return self._image.shape[1]

    def height(self):
        if self._image is None:
            return None
        return self._image.shape[0]

    def boundingRect(self):
        """Override."""
        if self._image is None:
            return QRectF(0., 0., 0., 0.)
        return QRectF(0., 0., float(self.width()), float(self.height()))

    def setLevels(self, levels):
        """Set image colormap scaling levels.

        :param tuple levels: (min, max).
        """
        if self._levels != levels:
            self._levels = levels
            self._fast_lut = None
            self.setImage(auto_levels=False)

    def getLevels(self):
        return self._levels

    def setLookupTable(self, lut, update=True):
        if lut is not self._lut:
            self._lut = lut
            self._fast_lut = None
            if update:
                self.setImage(auto_levels=False)

    def clear(self):
        self._image = None
        self.prepareGeometryChange()
        self.informViewBoundsChanged()
        self.update()

    def setImage(self, image=None, auto_levels=False):
        image_changed = False
        if image is None:
            if self._image is None:
                return
        else:
            image_changed = True
            shape_changed = \
                self._image is None or image.shape != self._image.shape

            image = image.view(np.ndarray)

            if self._image is None or image.dtype != self._image.dtype:
                self._fast_lut = None

            self._image = image

            if shape_changed:
                self.prepareGeometryChange()
                self.informViewBoundsChanged()

        if auto_levels:
            self._levels = quick_min_max(
                self._image, q=self._auto_level_quantile)

        self._qimage = None
        self.update()

        if image_changed:
            self.image_changed_sgn.emit()

    def dataTransform(self):
        """Return data transform.

        The transform maps from this image's input array to its local
        coordinate system.

        This transform corrects for the transposition that occurs when
        image data is interpreted in row-major order.
        """
        # Might eventually need to account for downsampling / clipping here
        tr = QTransform()
        # transpose
        tr.scale(1, -1)
        tr.rotate(-90)
        return tr

    def inverseDataTransform(self):
        """Return inverse data transform.

        The transform maps from this image's local coordinate system to
        its input array.
        """
        tr = QTransform()
        # transpose
        tr.scale(1, -1)
        tr.rotate(-90)
        return tr

    def mapToData(self, obj):
        tr = self.inverseDataTransform()
        return tr.map(obj)

    def mapFromData(self, obj):
        tr = self.dataTransform()
        return tr.map(obj)

    def render(self):
        """Convert data to QImage for displaying."""
        if self._image is None or self._image.size == 0:
            return

        # Request a lookup table
        if isinstance(self._lut, Callable):
            lut = self._lut(self._image)
        else:
            lut = self._lut

        # Downsample

        # reduce dimensions of image based on screen resolution
        o = self.mapToDevice(QPointF(0, 0))
        x = self.mapToDevice(QPointF(1, 0))
        y = self.mapToDevice(QPointF(0, 1))

        # Check if graphics view is too small to render anything
        if o is None or x is None or y is None:
            return

        w = Point(x-o).length()
        h = Point(y-o).length()
        if w == 0 or h == 0:
            self._qimage = None
            return

        xds = max(1, int(1.0 / w))
        yds = max(1, int(1.0 / h))
        # TODO: replace fn.downsample
        image = fn.downsample(self._image, xds, axis=1)
        image = fn.downsample(image, yds, axis=0)
        self._ds_rate = (xds, yds)

        # Check if downsampling reduced the image size to zero due to inf values.
        if image.size == 0:
            return

        # if the image data is a small int, then we can combine levels + lut
        # into a single lut for better performance
        levels = self._levels
        if levels is not None and image.dtype in (np.ubyte, np.uint16):
            if self._fast_lut is None:
                eflsize = 2**(image.itemsize*8)
                ind = np.arange(eflsize)
                minlev, maxlev = levels
                levdiff = maxlev - minlev
                # avoid division by 0
                levdiff = 1 if levdiff == 0 else levdiff
                if lut is None:
                    efflut = fn.rescaleData(
                        ind, scale=255./levdiff, offset=minlev, dtype=np.ubyte)
                else:
                    lutdtype = np.min_scalar_type(lut.shape[0]-1)
                    efflut = fn.rescaleData(
                        ind, scale=(lut.shape[0]-1)/levdiff,
                        offset=minlev, dtype=lutdtype, clip=(0, lut.shape[0]-1))
                    efflut = lut[efflut]

                self._fast_lut = efflut
            lut = self._fast_lut
            levels = None

        # TODO: replace fn.makeARGB and fn.makeQImage
        argb, alpha = fn.makeARGB(image, lut=lut, levels=levels)
        self._qimage = fn.makeQImage(argb, alpha, transpose=False)

    def paint(self, p, *args):
        """Override."""
        if self._image is None:
            return

        if self._qimage is None:
            self.render()
            if self._qimage is None:
                return

        p.drawImage(QRectF(0, 0, *self._image.shape[::-1]), self._qimage)

    def histogram(self):
        """Return estimated histogram of image pixels.

        :returns: (hist, bin_centers)
        """
        if self._image is None or self._image.size == 0:
            return None, None

        step = (max(1, int(np.ceil(self._image.shape[0] / 200))),
                max(1, int(np.ceil(self._image.shape[1] / 200))))

        sliced_data = self._image[::step[0], ::step[1]]

        lb, ub = np.nanmin(sliced_data), np.nanmax(sliced_data)

        if np.isnan(lb) or np.isnan(ub):
            # the data are all-nan
            return None, None

        if lb == ub:
            # degenerate image, arange will fail
            lb -= 0.5
            ub += 0.5

        n_bins = 500
        if sliced_data.dtype.kind in "ui":
            # step >= 1
            step = np.ceil((ub - lb) / n_bins)
            # len(bins) >= 2
            bins = np.arange(lb, ub + 0.01 * step, step, dtype=int)
        else:
            # for float data, let numpy select the bins.
            bins = np.linspace(lb, ub, n_bins)

        hist, bin_edges = np.histogram(sliced_data, bins=bins)

        return hist, (bin_edges[:-1] + bin_edges[1:]) / 2.

    def setPxMode(self, state):
        """Set ItemIgnoresTransformations flag.

        Set whether the item ignores transformations and draws directly
        to screen pixels.

        :param bool state: If True, the item will not inherit any scale or
            rotation transformations from its parent items, but its position
            will be transformed as usual.
        """
        self.setFlag(self.ItemIgnoresTransformations, state)

    @pyqtSlot()
    def setScaledMode(self):
        """Slot connected in GraphicsView."""
        self.setPxMode(False)

    def pixelSize(self):
        """Override.

        Return scene-size of a single pixel in the image.
        """
        br = self.sceneBoundingRect()
        if self._image is None:
            return 1, 1
        return br.width() / self.width(), br.height() / self.height()

    def viewTransformChanged(self):
        """Override."""
        o = self.mapToDevice(QPointF(0, 0))
        x = self.mapToDevice(QPointF(1, 0))
        y = self.mapToDevice(QPointF(0, 1))
        w = Point(x-o).length()
        h = Point(y-o).length()
        if w == 0 or h == 0:
            self._qimage = None
            return
        xds = max(1, int(1.0 / w))
        yds = max(1, int(1.0 / h))
        if (xds, yds) != self._ds_rate:
            self._qimage = None
            self.update()

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
            value = self._image[y, x]

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

    def __init__(self, item, parent=None):
        """Initialization.

        :param ImageItem item: a reference to the masked image item.
        """
        super().__init__(parent=parent)
        if not isinstance(item, ImageItem):
            raise TypeError("Input item must be an ImageItem instance.")

        self._image_item = weakref.ref(item)
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

        self._updateImageItem()

    def removeMask(self):
        """Completely remove the current mask."""
        # The following sequence ensures that it can be used to rescue if
        # there is discrepancy between ImageTool and the ImageProcessor.

        self._mask_pub.remove()

        if self._mask is None:
            return

        self._mask.fill(self._TRANSPARENT)
        self._updateImageItem()

    def paint(self, p, *args):
        """Override."""
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
        self._updateImageItem()

    def _updateImageItem(self):
        image_item = self._image_item()
        if image_item is not None:
            image_item.update()


class RectROI(pg.ROI):
    """Rectangular ROI widget.

    Note: the widget is slightly different from pyqtgraph.RectROI
    """
    def __init__(self, idx, *, pos=(0, 0), size=(1, 1), label="", pen=None, parent=None):
        """Initialization.

        :param int idx: index of the ROI.
        :param tuple pos: (x, y) of the left-upper corner.
        :param tuple size: (w, h) of the ROI.
        :param None/QPen pen: QPen to draw the ROI.
        """
        super().__init__(pos, size,
                         translateSnap=True,
                         scaleSnap=True,
                         pen=pen,
                         parent=parent)
        if pen is None:
            pen = QPen(Qt.SolidLine)

        self._index = idx
        self._label = None

        self._label_item = pg.TextItem(color=pen.color())
        font = QFont()
        font.setPointSizeF(20)
        self._label_item.setFont(font)
        self._label_item.setParentItem(self)
        self.setLabel(label)

        if len(label) == 0:
            self._label_item.hide()

    def label(self):
        return self._label

    def setLabel(self, label):
        self._label_item.setText(label)

        if len(label) == 0:
            self._label_item.hide()
        else:
            self._label_item.show()

        self._label = label

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


class GeometryItem(pg.GraphicsObject):
    def __init__(self, pen=None, brush=None, parent=None):
        super().__init__(parent=parent)

        if pen is None and brush is None:
            self._pen = FColor.mkPen('b')
            self._brush = FColor.mkBrush(None)
        else:
            self._pen = FColor.mkPen(None) if pen is None else pen
            self._brush = FColor.mkBrush(None) if brush is None else brush

        self._picture = None

    @abc.abstractmethod
    def _drawPicture(self):
        raise NotImplementedError

    def _updatePicture(self):
        self._picture = None
        self.prepareGeometryChange()
        self.informViewBoundsChanged()

    @abc.abstractmethod
    def setGeometry(self, *args, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def clearGeometry(self):
        raise NotImplementedError

    def boundingRect(self):
        """Override."""
        if self._picture is None:
            self._drawPicture()
        return QRectF(self._picture.boundingRect())

    def paint(self, p, *args):
        """Override."""
        if self._picture is None:
            self._drawPicture()
        self._picture.play(p)


class RingItem(GeometryItem):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._cx = None
        self._cy = None
        self._radials = None

        self.clearGeometry()

    def _drawPicture(self):
        """Override."""
        self._picture = QPicture()

        p = QPainter(self._picture)
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(self._pen)
        p.setBrush(self._brush)

        c = QPointF(self._cx, self._cy)
        for r in self._radials:
            p.drawEllipse(c, r, r)

        p.end()

    def setGeometry(self, cx, cy, radials):
        """Override."""
        self._cx = cx
        self._cy = cy
        self._radials = radials
        self._updatePicture()

    def clearGeometry(self):
        """Override."""
        self._cx = 0
        self._cy = 0
        self._radials = []
        self._updatePicture()
