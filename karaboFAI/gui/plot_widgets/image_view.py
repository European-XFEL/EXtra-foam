"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

ImageView and other derivative ImageView widgets.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import numpy as np

from .. import pyqtgraph as pg
from ..pyqtgraph import HistogramLUTWidget, QtCore, QtGui

from .plot_widget import PlotWidget
from .plot_items import ImageItem, MaskItem
from .roi import CropROI, RectROI
from ..misc_widgets import colorMapFactory, make_brush, make_pen
from ..mediator import Mediator
from ...algorithms import intersection, quick_min_max
from ...config import config
from ...logger import logger


class ImageView(QtGui.QWidget):
    """ImageView class.

    A widget used for displaying a single image. Two ROI widgets are
    embedded in this widget.

    Note: it is different from the ImageView in pyqtgraph!
    """
    ROI1_POS0 = (50, 50)
    ROI2_POS0 = (60, 60)
    ROI_SIZE0 = (100, 100)

    def __init__(self, *, level_mode='mono',
                 lock_roi=True, hide_axis=True, parent=None):
        """Initialization.

        :param str level_mode: 'mono' or 'rgba'. If 'mono', then only
            a single set of black/white level lines is drawn, and the
            levels apply to all channels in the image. If 'rgba', then
            one set of levels is drawn for each channel.
        :param bool lock_roi: True for non-movable/non-translatable ROI.
            This is used in ImageView with passive ROI.
        :param bool hide_axis: True for hiding left and bottom axes.
        """
        super().__init__(parent=parent)
        try:
            parent.registerPlotWidget(self)
        except AttributeError:
            pass

        self.roi1 = RectROI(self.ROI1_POS0, self.ROI_SIZE0,
                            lock=lock_roi,
                            pen=make_pen(config["ROI_COLORS"][0]))
        self.roi2 = RectROI(self.ROI2_POS0, self.ROI_SIZE0,
                            lock=lock_roi,
                            pen=make_pen(config["ROI_COLORS"][1]))
        self.roi1.hide()
        self.roi2.hide()

        self._plot_widget = PlotWidget()
        if hide_axis:
            self._plot_widget.hideAxis()

        self._image_item = pg.ImageItem(border='w')
        self._mask_item = MaskItem(self._image_item)

        self._plot_widget.addItem(self._image_item)
        self._plot_widget.addItem(self._mask_item)
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

    def update(self, data):
        """karaboFAI interface."""
        pass

    def updateROI(self, data):
        if data.roi.roi1 is not None:
            self.roi1.show()
            w, h, px, py = data.roi.roi1
            self.roi1.setSize((w, h), update=False)
            self.roi1.setPos((px, py), update=False)
        else:
            self.roi1.hide()

        if data.roi.roi2 is not None:
            self.roi2.show()
            w, h, px, py = data.roi.roi2
            self.roi2.setSize((w, h), update=False)
            self.roi2.setPos((px, py), update=False)
        else:
            self.roi2.hide()

    @property
    def image(self):
        return self._image

    def setImage(self, img, *, auto_range=True, auto_levels=True):
        """Set the current displayed image.

        :param np.ndarray img: the image to be displayed.
        :param bool auto_range: whether to scale/pan the view to fit
            the image.
        :param bool auto_levels: whether to update the white/black levels
            to fit the image.
        """
        self._image_item.setImage(img, autoLevels=False)
        self._image = img
        self._mask_item.updateImage()

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

    def setBorder(self, *args, **kwargs):
        self._image_item.setBorder(*args, **kwargs)

    def setAspectLocked(self, *args, **kwargs):
        self._plot_widget.setAspectLocked(*args, **kwargs)

    def invertY(self, *args, **kwargs):
        self._plot_widget.plotItem.invertY(*args, **kwargs)

    def addItem(self, *args, **kwargs):
        self._plot_widget.addItem(*args, **kwargs)

    def removeItem(self, *args, **kwargs):
        self._plot_widget.removeItem(*args, **kwargs)

    def close(self):
        self.parent().unregisterPlotWidget(self)
        super().close()


class ImageAnalysis(ImageView):
    """ImageAnalysis widget.

    Advance image analysis widget built on top of ImageView widget.
    It provides tools like masking, cropping, etc.
    """
    # restore_flag, x, y, w, h
    crop_area_change_sgn = QtCore.pyqtSignal(bool, int, int, int, int)
    # ImageMaskChange, x, y, w, h
    mask_region_change_sgn = QtCore.Signal(object, int, int, int, int)

    def __init__(self, *args, **kwargs):
        """Initialization."""
        super().__init__(*args, **kwargs)

        self._plot_widget.setTitle('')  # reserve space for displaying

        mediator = Mediator()

        # set the customized ImageItem
        self._image_item = ImageItem(border='w')
        self._image_item.mouse_moved_sgn.connect(self.onMouseMoved)
        self._mask_item = MaskItem(self._image_item)
        self._mask_item.mask_region_change_sgn.connect(self.onMaskRegionChange)
        self.mask_region_change_sgn.connect(mediator.onMaskRegionChange)

        # re-add items to keep the order
        self._plot_widget.clear()
        self._plot_widget.addItem(self._image_item)
        self._plot_widget.addItem(self._mask_item)
        self._plot_widget.addItem(self.roi1)
        self._plot_widget.addItem(self.roi2)

        self.invertY(True)
        self.setAspectLocked(True)
        self._hist_widget.setImageItem(self._image_item)

        # add cropping widget
        self.crop = CropROI((0, 0), (100, 100))
        self.crop.hide()
        self._plot_widget.addItem(self.crop)

        self._image_data = None
        self._moving_average_window = 1

    def setImageData(self, image_data):
        """Set the ImageData.

        :param ImageData image_data: ImageData instance
        """
        self._image_data = image_data
        if image_data is not None:
            self.setImage(image_data.masked_mean)

    @QtCore.pyqtSlot(int, int, float)
    def onMouseMoved(self, x, y, v):
        x, y = self._image_data.pos(x, y)
        if x < 0 or y < 0:
            self._plot_widget.setTitle('')
        else:
            self._plot_widget.setTitle(f'x={x}, y={y}, value={round(v, 1)}')

    def onCropToggle(self):
        if self.crop.isVisible():
            self.crop.hide()
        else:
            if self._image is not None:
                self.crop.setPos(0, 0)
                self.crop.setSize(self._image.shape[::-1])
            self.crop.show()

    def onCropConfirmed(self):
        if not self.crop.isVisible():
            return

        if self._image is None:
            return

        x, y = self.crop.pos()
        w, h = self.crop.size()
        h0, w0 = self._image.shape

        w, h, x, y = [int(v) for v in intersection(w, h, x, y, w0, h0, 0, 0)]
        if w > 0 and h > 0:
            # there is intersection
            self.setImage(self._image[y:y+h, x:x+w], auto_levels=False)
            # convert x, y to position at the original image
            x, y = self._image_data.pos(x, y)
            self.crop_area_change_sgn.emit(False, x, y, w, h)
            self._image_data.crop_area = (x, y, w, h)

        self.crop.hide()

    def onRestoreImage(self):
        if self._image_data is None:
            return

        self.crop_area_change_sgn.emit(True, 0, 0, 0, 0)
        self._image_data.crop_area = None
        self.setImage(self._image_data.masked_mean, auto_levels=False)

    @QtCore.pyqtSlot()
    def onBkgChange(self):
        self._image_data.background = float(self.sender().text())
        self.setImage(self._image_data.masked_mean,
                      auto_levels=False, auto_range=False)

    @QtCore.pyqtSlot(float, float)
    def onThresholdMaskChange(self, v0, v1):
        # recalculate the unmasked mean image
        self._image_data.threshold_mask = (v0, v1)
        self.setImage(self._image_data.masked_mean)

    @QtCore.pyqtSlot(object, int, int, int, int)
    def onMaskRegionChange(self, tp, x, y, w, h):
        x, y = self._image_data.pos(x, y)
        self.mask_region_change_sgn.emit(tp, x, y, w, h)

    @QtCore.pyqtSlot(bool)
    def onDrawToggled(self, draw_type, checked):
        self._mask_item.draw_type = draw_type
        self._image_item.drawing = checked

    def clearMask(self):
        self._mask_item.clear()


class AssembledImageView(ImageView):
    """AssembledImageView class.

    Widget for displaying the assembled image of the average of all pulses
    in a train.
    """
    def __init__(self, *, parent=None):
        """Initialization."""
        super().__init__(parent=parent)

        self._threshold_mask = None  # current threshold mask

        self.setColorMap(colorMapFactory[config["COLOR_MAP"]])

    def update(self, data):
        """Override."""
        threshold_mask = data.image.threshold_mask
        if threshold_mask != self._threshold_mask:
            self._is_initialized = False
            self._threshold_mask = threshold_mask

        self.setImage(data.image.masked_mean,
                      auto_range=False,
                      auto_levels=(not self._is_initialized))

        self.updateROI(data)

        if not self._is_initialized:
            self._is_initialized = True


class SinglePulseImageView(ImageView):
    """SinglePulseImageView class.

    Widget for displaying the assembled image of a single pulse.
    """
    def __init__(self, *, pulse_id=0, parent=None):
        """Initialization."""
        super().__init__(parent=parent)

        self.pulse_id = pulse_id

    def update(self, data):
        """Override."""
        images = data.image.images
        threshold_mask = data.image.threshold_mask

        max_id = data.n_pulses - 1
        if self.pulse_id <= max_id:
            np.clip(images[self.pulse_id], *threshold_mask,
                    images[self.pulse_id])
        else:
            logger.error("<VIP pulse ID>: VIP pulse ID ({}) > Maximum "
                         "pulse ID ({})".format(self.pulse_id, max_id))
            return

        self.setImage(images[self.pulse_id])

        self.updateROI(data)


class RoiImageView(ImageView):
    """RoiImageView class.

    Widget for displaying the ROI for the assembled image.
    """
    def __init__(self, roi1=True, **kwargs):
        """Initialization.

        :param bool roi1: True for displaying ROI1 and False for ROI2.
        """
        super().__init__(**kwargs)

        self._is_roi1 = roi1

        self.roi1.hide()
        self.roi2.hide()

    def update(self, data):
        """Override."""
        image = data.image.masked_mean

        if self._is_roi1:
            if data.roi.roi1 is None:
                return
            w, h, px, py = data.roi.roi1
        else:
            if data.roi.roi2 is None:
                return
            w, h, px, py = data.roi.roi2

        self.setImage(image[py:py+h, px:px+w])
