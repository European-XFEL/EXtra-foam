"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

ImageView and SinglePulseImageView.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import numpy as np

from ..widgets.pyqtgraph import (
    QtGui, QtCore, HistogramLUTWidget, ImageItem, ROI
)
from .misc_widgets import colorMapFactory, PenFactory
from .plot_widget import PlotWidget
from ..data_processing import quick_min_max
from ..logger import logger
from ..config import config


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


class ImageView(QtGui.QWidget):
    """ImageView class.

    Widget used for displaying and analyzing a single image.

    Note: it is different from the ImageView in pyqtgraph!
    """
    ROI_POS0 = (50, 50)
    ROI_SIZE0 = (50, 50)

    def __init__(self, *, parent=None, level_mode='mono', lock_roi=True):
        """Initialization.

        :param str level_mode: 'mono' or 'rgba'. If 'mono', then only
            a single set of black/white level lines is drawn, and the
            levels apply to all channels in the image. If 'rgba', then
            one set of levels is drawn for each channel.
        """
        super().__init__(parent=parent)
        try:
            parent.registerPlotWidget(self)
        except AttributeError:
            pass

        self.roi1 = RectROI(self.ROI_POS0, self.ROI_SIZE0,
                            lock=lock_roi,
                            pen=PenFactory.__dict__[config["ROI_COLORS"][0]])
        self.roi2 = RectROI(self.ROI_POS0, self.ROI_SIZE0,
                            lock=lock_roi,
                            pen=PenFactory.__dict__[config["ROI_COLORS"][1]])
        self.roi1.hide()
        self.roi2.hide()

        self._plot_widget = PlotWidget()
        self._image_item = ImageItem(border='w')
        self._plot_widget.addItem(self._image_item)
        self._plot_widget.addItem(self.roi1)
        self._plot_widget.addItem(self.roi2)
        self.invertY(True)
        self.setAspectLocked(True)

        self._hist_widget = HistogramLUTWidget()
        self._hist_widget.setLevelMode(level_mode)
        self._hist_widget.setImageItem(self._image_item)

        self.setColorMap(colorMapFactory[config["COLOR_MAP"]])

        self._is_initialized = False
        self._image = None
        self._image_levels = None

        self.initUI()

    def initUI(self):
        layout = QtGui.QHBoxLayout()
        layout.addWidget(self._plot_widget)
        layout.addWidget(self._hist_widget)
        self.setLayout(layout)
        self.layout().setContentsMargins(0, 0, 0, 0)
        self.layout().setSpacing(0)

        self._plot_widget.hideAxis()

    def update(self, data):
        """karaboFAI interface."""
        self.setImage(data.image_mean,
                      auto_range=False,
                      auto_levels=(not self._is_initialized))

        self.updateROI(data)

        if not self._is_initialized:
            self._is_initialized = True

    def updateROI(self, data):
        if data.roi1 is not None:
            self.roi1.show()
            w, h, cx, cy = data.roi1
            self.roi1.setSize((w, h), update=False)
            self.roi1.setPos((cx, cy), update=False)
        else:
            self.roi1.hide()

        if data.roi2 is not None:
            self.roi2.show()
            w, h, cx, cy = data.roi2
            self.roi2.setSize((w, h), update=False)
            self.roi2.setPos((cx, cy), update=False)
        else:
            self.roi2.hide()

    def setImage(self, img, *, auto_range=True, auto_levels=True):
        """Set the current displayed image.

        :param np.ndarray img: the image to be displayed.
        :param bool auto_range: whether to scale/pan the view to fit
            the image.
        :param bool auto_levels: whether to update the white/black levels
            to fit the image.
        """
        self._image = img

        self._image_item.setImage(self._image, autoLevels=False)

        if auto_levels:
            self._image_levels = quick_min_max(self._image)
            self.setLevels(rgba=[self._image_levels])

        if auto_range:
            self._plot_widget.plotItem.vb.autoRange()

    def setLevels(self, *args, **kwargs):
        """Set the min/max (bright and dark) levels.

        See HistogramLUTItem.setLevels.
        """
        self._hist_widget.setLevels(*args, **kwargs)

    def clear(self):
        self._image_item.clear()

    def setColorMap(self, cm):
        """Set colormap for the displayed image.

        :param cm: a ColorMap object.
        """
        self._hist_widget.gradient.setColorMap(cm)

    def setAspectLocked(self, *args, **kwargs):
        self._plot_widget.setAspectLocked(*args, **kwargs)

    def invertY(self, *args, **kwargs):
        self._plot_widget.plotItem.invertY(*args, **kwargs)

    def addItem(self, *args, **kwargs):
        self._plot_widget.addItem(*args, **kwargs)

    def close(self):
        self.parent().unregisterPlotWidget(self)
        super().close()


class SinglePulseImageView(ImageView):
    """SinglePulseImageView class.

    Widget used for displaying the assembled image of a single pulse.
    """
    def __init__(self, *, pulse_id=0, parent=None):
        """Initialization."""
        super().__init__(parent=parent)

        self.pulse_id = pulse_id

        self.setColorMap(colorMapFactory[config["COLOR_MAP"]])

    def update(self, data):
        """Override."""
        images = data.images
        threshold_mask = data.threshold_mask

        max_id = len(data.intensity) - 1
        if self.pulse_id <= max_id:
            np.clip(images[self.pulse_id], *threshold_mask,
                    images[self.pulse_id])
        else:
            logger.error("<VIP pulse ID>: VIP pulse ID ({}) > Maximum "
                         "pulse ID ({})".format(self.pulse_id, max_id))
            return

        self.setImage(images[self.pulse_id],
                      auto_range=False,
                      auto_levels=(not self._is_initialized))

        self.updateROI(data)

        if not self._is_initialized:
            self._is_initialized = True


class RoiImageView(ImageView):
    """RoiImageView class.

    Widget used for displaying the ROI for the assembled image.
    """
    def __init__(self, *, roi1=True, parent=None):
        """Initialization.

        :param bool roi1: True for displaying ROI1 and False for ROI2.
        """
        super().__init__(parent=parent)

        self._is_roi1 = roi1

        self.roi1.hide()
        self.roi2.hide()
        self.setColorMap(colorMapFactory[config["COLOR_MAP"]])

    def update(self, data):
        """Override."""
        image = data.image_mean

        if self._is_roi1:
            if data.roi1 is None:
                return
            w, h, cx, cy = data.roi1
        else:
            if data.roi2 is None:
                return
            w, h, cx, cy = data.roi2

        self.setImage(image[cy:cy+h, cx:cx+w],
                      auto_range=False,
                      auto_levels=(not self._is_initialized))

        if not self._is_initialized:
            self._is_initialized = True
