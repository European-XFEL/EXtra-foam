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

from .base_plot_widget import PlotWidget
from .plot_items import ImageItem, MaskItem
from .roi import RectROI
from ..misc_widgets import colorMapFactory
from ..mediator import Mediator
from ...algorithms import quick_min_max
from ...config import config, redis_connection
from ...logger import logger


class ImageView(QtGui.QWidget):
    """ImageView class.

    A widget used for displaying a single image. Two ROI widgets are
    embedded in this widget.

    Note: it is different from the ImageView in pyqtgraph!
    """
    ROI_X0 = 50
    ROI_Y0 = 50
    ROI_SIZE0 = (100, 100)

    def __init__(self, *, level_mode='mono',
                 hide_axis=True, color_map=None, parent=None):
        """Initialization.

        :param str level_mode: 'mono' or 'rgba'. If 'mono', then only
            a single set of black/white level lines is drawn, and the
            levels apply to all channels in the image. If 'rgba', then
            one set of levels is drawn for each channel.
        :param bool hide_axis: True for hiding left and bottom axes.
        """
        super().__init__(parent=parent)
        try:
            parent.registerPlotWidget(self)
        except AttributeError:
            pass

        self._mediator = Mediator()

        self._rois = []
        self._initializeROIs()

        self._plot_widget = PlotWidget()
        if hide_axis:
            self._plot_widget.hideAxis()

        self._image_item = pg.ImageItem(border='w')
        self._mask_item = MaskItem(self._image_item)

        self._plot_widget.addItem(self._image_item)
        self._plot_widget.addItem(self._mask_item)
        for roi in self._rois:
            self._plot_widget.addItem(roi)
        self.invertY(True)
        self.setAspectLocked(True)

        self._hist_widget = HistogramLUTWidget()
        self._hist_widget.setLevelMode(level_mode)
        self._hist_widget.setImageItem(self._image_item)

        if color_map is None:
            self.setColorMap(colorMapFactory[config["GUI"]["COLOR_MAP"]])
        else:
            self.setColorMap(colorMapFactory["thermal"])

        self._is_initialized = False
        self._image = None
        self._image_levels = None

        self.initUI()

        self._mediator.reset_image_level_sgn.connect(self._updateImage)

    def initUI(self):
        layout = QtGui.QHBoxLayout()
        layout.addWidget(self._plot_widget)
        layout.addWidget(self._hist_widget)
        self.setLayout(layout)
        self.layout().setContentsMargins(0, 0, 0, 0)
        self.layout().setSpacing(0)

    def reset(self):
        # TODO: check
        self._image_item.clear()

    def update(self, data):
        """karaboFAI interface."""
        pass

    def _initializeROIs(self):
        for i, color in enumerate(config["ROI_COLORS"], 1):
            roi = RectROI(i, color,
                          (self.ROI_X0 + 10*i, self.ROI_Y0 + 10*i),
                          self.ROI_SIZE0)
            roi.hide()
            self._rois.append(roi)

    def updateROI(self, data):
        """Update ROIs.

        Update ROI through data instead of passing signals to ensure that
        visualization of ROIs and calculation of ROI data are synchronized.
        """
        for i, roi in enumerate(self._rois, 1):
            roi_area = getattr(data.roi, f"roi{i}")
            if roi_area is not None:
                roi.show()
                x, y, w, h = roi_area
                roi.setSize((w, h), update=False)
                roi.setPos((x, y), update=False)
            else:
                roi.hide()

    @property
    def image(self):
        return self._image

    @property
    def rois(self):
        return self._rois

    def setImage(self, img, *, auto_range=False, auto_levels=False):
        """Set the current displayed image.

        :param np.ndarray img: the image to be displayed.
        :param bool auto_range: whether to scale/pan the view to fit
            the image. defaut = False
        :param bool auto_levels: whether to update the white/black levels
            to fit the image. default = False
        """
        self._image_item.setImage(img, autoLevels=False)
        self._image = img
        self._mask_item.onSetImage()

        if auto_levels:
            self._image_levels = quick_min_max(self._image)
            self.setLevels(rgba=[self._image_levels])

        if auto_range:
            self._plot_widget.plotItem.vb.autoRange()

    def _updateImage(self):
        """Re-display the current image with auto_levels."""
        if self._image is None:
            return
        self.setImage(self._image, auto_levels=True)

    def setLevels(self, *args, **kwargs):
        """Set the min/max (bright and dark) levels.

        See HistogramLUTItem.setLevels.
        """
        self._hist_widget.setLevels(*args, **kwargs)

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
    It provides tools like masking, etc.
    """

    def __init__(self, *args, **kwargs):
        """Initialization."""
        super().__init__(*args, **kwargs)

        self._plot_widget.setTitle('')  # reserve space for displaying

        # set the customized ImageItem
        self._image_item = ImageItem(border='w')
        self._image_item.mouse_moved_sgn.connect(self.onMouseMoved)
        self._mask_item = MaskItem(self._image_item)
        self._mask_item.mask_region_change_sgn.connect(
            self._mediator.onImageMaskRegionChange)

        # re-add items to keep the order
        self._plot_widget.clear()
        self._plot_widget.addItem(self._image_item)
        self._plot_widget.addItem(self._mask_item)
        for roi in self._rois:
            roi.setLocked(False)
            self._plot_widget.addItem(roi)

        self.invertY(True)
        self.setAspectLocked(True)
        self._hist_widget.setImageItem(self._image_item)

        self._image_data = None

    def setImageData(self, image_data, **kwargs):
        """Set the ImageData.

        :param _SimpleImageData image_data: _SimpleImageData instance.
        """
        self._image_data = image_data
        if image_data is None:
            return

        self.setImage(image_data.masked)

    def setImageRef(self):
        """Set the displayed image as reference image."""
        if self._image_data is not None:
            self._image_data.set_reference()

    def removeImageRef(self):
        """Remove reference image."""
        if self._image_data is not None:
            self._image_data.remove_reference()

    @QtCore.pyqtSlot(int, int, float)
    def onMouseMoved(self, x, y, v):
        if x < 0 or y < 0:
            self._plot_widget.setTitle('')
        else:
            self._plot_widget.setTitle(f'x={x}, y={y}, value={round(v, 1)}')

    @QtCore.pyqtSlot(int)
    def onMovingAverageWindowChange(self, v):
        if self._image_data is None:
            return

        self._image_data.set_ma_window(v)

    @QtCore.pyqtSlot(float)
    def onBkgChange(self, bkg):
        if self._image_data is None:
            return

        self._image_data.background = bkg
        self.setImage(self._image_data.masked)

    @QtCore.pyqtSlot(object)
    def onThresholdMaskChange(self, mask_range):
        if self._image_data is None:
            return

        self._image_data.threshold_mask = mask_range
        self.setImage(self._image_data.masked)

    @QtCore.pyqtSlot(bool)
    def onDrawToggled(self, masking_state, checked):
        self._mask_item.masking_state = masking_state
        self._image_item.drawing = checked

    @QtCore.pyqtSlot()
    def onClearImageMask(self):
        self._mask_item.clearMask()

    def saveImageMask(self):
        file_path = QtGui.QFileDialog.getSaveFileName()[0]
        if not file_path:
            logger.error("Please specify the image mask file!")
            return

        self._saveImageMaskImp(file_path)

    def _saveImageMaskImp(self, file_path):
        if self._image_data is None:
            logger.error("Image is not found!")
            return

        np.save(file_path, self._mask_item.toNDArray())
        logger.info(f"Image mask saved in {file_path}.npy!")

    def loadImageMask(self):
        file_path = QtGui.QFileDialog.getOpenFileName()[0]
        if not file_path:
            logger.error("Please specify the image mask file!")
            return

        self._loadImageMaskImp(file_path)

    def _loadImageMaskImp(self, file_path):
        if self._image is None:
            logger.error("Cannot load image mask without image!")
            return

        try:
            image_mask = np.load(file_path)
            if image_mask.shape != self._image.shape:
                msg = "The shape of image mask is different from the image!"
                logger.error(msg)
                return

            logger.info(f"Image mask loaded from {file_path}!")

            self._mask_item.loadMask(image_mask)
            self._publish_image_mask(image_mask)

        except (IOError, OSError) as e:
            logger.error(f"Cannot load mask from {file_path}")

    def _publish_image_mask(self, image_mask):
        r = redis_connection()
        buf = np.packbits(image_mask).tobytes()
        h, w = image_mask.shape
        r.publish("command:image_mask", str((2, 0, 0, w, h, w, h)))
        r.publish("command:image_mask", buf)


class AssembledImageView(ImageView):
    """AssembledImageView class.

    Widget for displaying the assembled image of the average of all pulses
    in a train.
    """
    def __init__(self, *, parent=None):
        """Initialization."""
        super().__init__(parent=parent)

        self.setColorMap(colorMapFactory[config["GUI"]["COLOR_MAP"]])

    def update(self, data):
        """Override."""
        self.setImage(data.image.masked_mean,
                      auto_levels=(not self._is_initialized))
        self.updateROI(data)

        if not self._is_initialized:
            self._is_initialized = True


class PumpProbeImageView(ImageView):
    """PumpProbeImageView class.

    Widget for displaying the on or off image in the pump-probe analysis.
    """
    def __init__(self, on=True, *, roi=False, diff=False, parent=None):
        """Initialization.

        :param bool on: True for display the on image while False for
            displaying the off image.
        :param bool roi: True for displaying the ROI while False for
            displaying the whole image.
        :param bool diff: True for displaying on - off ROI instead of
            off ROI. Ignored if roi == False. This option is not enabled
            for the whole image because of the concern of performance.
        """
        super().__init__(parent=parent)

        self._on = on
        self._roi = roi
        self._diff = diff
        if self._roi:
            self._plot_widget.removeItem(self._mask_item)

        self.setColorMap(colorMapFactory[config["GUI"]["COLOR_MAP"]])

    def update(self, data):
        """Override."""
        if self._on:
            if self._roi:
                img = data.pp.on_roi
            else:
                img = data.pp.on_image_mean
        else:
            if self._roi:
                if self._diff and data.pp.off_roi is not None:
                    img = data.pp.on_roi - data.pp.off_roi
                else:
                    img = data.pp.off_roi
            else:
                img = data.pp.off_image_mean

        if img is None:
            return

        self.setImage(img, auto_levels=(not self._is_initialized))
        if not self._roi:
            self.updateROI(data)

        if not self._is_initialized:
            self._is_initialized = True


class SinglePulseImageView(ImageView):
    """SinglePulseImageView class.

    Widget for displaying the assembled image of a single pulse.
    """
    def __init__(self, *, pulse_index=0, parent=None):
        """Initialization."""
        super().__init__(parent=parent)

        self.pulse_index = pulse_index

    def update(self, data):
        """Override."""
        images = data.image.images
        threshold_mask = data.image.threshold_mask

        max_id = data.n_pulses - 1
        if self.pulse_index <= max_id:
            np.clip(images[self.pulse_index], *threshold_mask,
                    images[self.pulse_index])
        else:
            logger.error("<VIP pulse index>: VIP pulse index ({}) > Maximum "
                         "pulse index ({})".format(self.pulse_index, max_id))
            return

        self.setImage(images[self.pulse_index],
                      auto_levels=(not self._is_initialized))

        self.updateROI(data)

        if not self._is_initialized:
            self._is_initialized = True


class RoiImageView(ImageView):
    """RoiImageView class.

    Widget for displaying the ROI for the assembled image.
    """
    def __init__(self, rank, **kwargs):
        """Initialization."""
        super().__init__(**kwargs)

        self._plot_widget.removeItem(self._mask_item)

        self._rank = rank

    def update(self, data):
        """Override."""
        image = data.image.masked_mean

        roi = getattr(data.roi, f"roi{self._rank}")

        if roi is None:
            return
        x, y, w, h = roi

        self.setImage(image[y:y+h, x:x+w], auto_range=True, auto_levels=True)


class BinImageView(ImageView):
    """BinImageView class.

    Widget for displaying the image in selected bins.
    """
    def __init__(self, index=0, *, parent=None):
        """Initialization.

        :param int index: index of bins
        """
        super().__init__(parent=parent)

        self._index = index

        self.setColorMap(colorMapFactory[config["GUI"]["COLOR_MAP"]])

    def update(self, data):
        """Override."""
        pass
