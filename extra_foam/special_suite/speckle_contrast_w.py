import time
import numpy as np
import multiprocessing

import pyFAI

from PyQt5.QtCore import Qt, pyqtSlot
from PyQt5.QtGui import QDoubleValidator, QIntValidator
from PyQt5.QtWidgets import QCheckBox, QSplitter, QPushButton, QLabel, QFrame, QHBoxLayout

import extra_data
from extra_geom import AGIPD_1MGeometry
from extra_foam.algorithms import hist_with_stats

from extra_foam.gui.ctrl_widgets import _SingleRoiCtrlWidget
from extra_foam.gui.ctrl_widgets.smart_widgets import (
    SmartBoundaryLineEdit, SmartLineEdit, SmartSliceLineEdit,
    SmartStringLineEdit
)

from extra_foam.gui.plot_widgets import ImageViewF, PlotWidgetF
from .special_analysis_base import (
    create_special, profiler, QThreadKbClient, QThreadWorker,
    _BaseAnalysisCtrlWidgetS, _SpecialAnalysisBase
)

from ..gui.misc_widgets import FColor
from ..gui.plot_widgets.image_items import CircleROI
import scipy


def integrate_cone(pos):
    i, x, y = pos
    fit_2d = ai.getFit2D()
    if not (np.isclose(fit_2d["centerX"], x) and np.isclose(fit_2d["centerY"], y)):
        ai.setFit2D(7800, x, y)

    q_min = 0.013
    q_max = 0.05

    _, I = ai.integrate1d(image,
                          300,
                          radial_range=(q_min, q_max),
                          azimuth_range=(i * cone_width, (i + 1) * cone_width),
                          mask=mask,
                          unit="q_A^-1",
                          dummy=np.nan)
    return I

def setup(_ai, image_shape, image_buffer, mask_buffer, _cone_width):
    global ai
    ai = _ai
    global image
    image = np.frombuffer(image_buffer)
    image.shape = image_shape
    global mask
    mask = np.frombuffer(mask_buffer)
    mask.shape = image_shape
    global cone_width
    cone_width = _cone_width

class SpeckleContrastProcessor(QThreadWorker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.photonBinningEnabled = False
        self.circle_roi_geom_st = None

        self._output_channel_fmt = "MID_DET_AGIPD1M-1/DET/{}CH0:xtdf"
        self._ppt = "image.data"

        self._geom = AGIPD_1MGeometry.from_quad_positions(quad_pos=[
            (-543.0, 657.0),
            (-607.0, -42.0),
            (532.0, -229.0),
            (588.0, 471.0),
        ])

        self.reset()

    def onRoiGeometryChange(self, value):
        super().onRoiGeometryChange(value)

        self._speckle_contrast_trend.clear()
        self._mask_updated = True

    def calc_contrast(self, adu_data, mask):
        """Derive pixel counts for k-unit photons from ADU, and from that a
           speckle contrast estimate.
           :param adu_data:       2darray (pixels) of ADU values
           :param adu_per_photon: constant factor to divide ADUs by
        """
        selected_data = adu_data[mask == 1]
        self.bin_photons(selected_data, out=selected_data)
        k_mean = np.nanmean(selected_data)

        try:
            k_hist = np.histogram(selected_data[~np.isnan(selected_data)],
                                  np.arange(-0.5, np.nanmax(selected_data) + 1, 1),
                                  density=True)
        except ValueError:
            return

        # contrast equation
        k = k_hist[0]
        try:
            contrast = k[0] / k[1] - 1 / k_mean
        except IndexError:
            contrast = 0.0

        return contrast

    def calc_beam_center(self):
        if self._averaged_image is None:
            self.log.error("No images available")
            return None

        desired_shape = (16 * 512, 128)
        image_buffer = multiprocessing.Array("d", desired_shape[0] * desired_shape[1], lock=False)
        image = np.frombuffer(image_buffer)
        image.shape = desired_shape
        image[...] = np.mean(self._binned_averaged_image, (0)).reshape(desired_shape)

        mask_buffer = multiprocessing.Array("d", desired_shape[0] * desired_shape[1], lock=False)
        mask = np.frombuffer(mask_buffer)
        mask.shape = desired_shape
        mask[...] = np.isnan(image).astype('bool')

        agipd = pyFAI.detectors.Detector(200e-6, 200e-6)
        agipd.aliases = ["AGIPD"]
        agipd.shape = image.shape
        agipd.mask = np.zeros(image.shape)
        agipd.IS_CONTINUOUS = False
        agipd.IS_FLAT = True
        agipd.set_pixel_corners(self._geom.to_distortion_array())

        final_image_shape = self._identity_mask.shape
        px, py = final_image_shape[1] // 2, final_image_shape[0] // 2
        ai = pyFAI.AzimuthalIntegrator(detector=agipd)
        ai.setFit2D(7800, px, py)
        ai.wavelength = 1.38e-10

        # Set constants
        number_of_cones = 8
        cone_width = 360 // number_of_cones

        def get_SAXS_var(pos, worker_pool):
            x, y = pos
            _I = worker_pool.map(integrate_cone, [(i, x, y) for i in range(number_of_cones)])

            return np.nanmean(np.nanvar(np.log(np.array(_I)), 0), 0)

        with multiprocessing.Pool(processes=2, initializer=setup, initargs=(ai, image.shape, image_buffer, mask_buffer, cone_width)) as worker_pool:
            opt_result = scipy.optimize.minimize(get_SAXS_var,
                                                 (px + 1, py),
                                                 args=(worker_pool),
                                                 method="nelder-mead",
                                                 options={ "maxiter": 20, "xatol": 1, "fatol": 1 })
            center = np.around(opt_result.x).astype(int)
            final_variance = get_SAXS_var(center, worker_pool)

        self.log.info(f"Found beam center at ({center[0]}, {center[1]}), with variance {final_variance:.5f}")
        return center

    def bin_photons(self, data, out=None):
        tmp = np.zeros(data.shape, dtype=data.dtype) if out is None else out

        np.add(data, 0.5 * self.ADU_threshold, out=tmp)
        np.divide(tmp, self.ADU_threshold, out=tmp)
        np.floor(tmp, out=tmp)

        return tmp

    def bin_and_clamp_photons(self, data, out=None):
        tmp = np.zeros(data.shape, dtype=data.dtype) if out is None else out

        self.bin_photons(data, out=tmp)
        tmp[tmp < 0] = 0

        return tmp

    def process(self, data):
        avg_and_position = lambda img: self._geom.position_modules_fast(np.nanmean(img, axis=0))[0]
        data, meta = data["raw"], data["meta"]

        tid = self.getTrainId(meta)
        rsd = { key.split()[0]: { self._ppt : data[key] } for key in data }

        frame = extra_data.stack_detector_data(rsd, self._ppt)

        # If this is the first image in the stream, initialize self._averaged_image
        if self._averaged_image is None:
            self._averaged_image = frame
            self._binned_averaged_image = self.bin_and_clamp_photons(frame)
        else:
            # Otherwise, update the running average of the images
            self._averaged_image        += (frame                             - self._averaged_image)        / self._image_count
            self._binned_averaged_image += (self.bin_and_clamp_photons(frame) - self._binned_averaged_image) / self._image_count
        self._image_count += 1

        image = avg_and_position(self.bin_and_clamp_photons(frame) if self.photonBinningEnabled else frame)
        avg_image = avg_and_position(self._binned_averaged_image if self.photonBinningEnabled else self._averaged_image)

        self.updateMasks(image)
        self._speckle_contrast_trend.append(self.calc_contrast(avg_and_position(frame), self._mask))

        return {
            "image": image,
            "averaged_image": avg_image,
            "trendline": self._speckle_contrast_trend
        }

    def updateMasks(self, image):
        # Initialize the identity mask
        if self._identity_mask is None:
            self._identity_mask = ~np.isnan(image)

        # If necessary, apply the user-defined mask
        if self._mask_updated:
            self._mask = np.zeros(image.shape)

            if self._roi_geom_st:
                x, y, w, h = self._roi_geom_st
                self._mask[y : y + h, x : x + w] = 1
            elif self.circle_roi_geom_st:
                x, y, w, h = self.circle_roi_geom_st
                radius = w // 2
                center_x = x + radius
                center_y = y + radius

                py, px = np.ogrid[0:image.shape[0], 0:image.shape[1]]
                radius_mask = (px - center_x)**2 + (py - center_y)**2 <= radius**2
                self._mask[radius_mask] = 1

            self._mask *= self._identity_mask
            self._mask_updated = False

        elif (self._roi_geom_st is None and self.circle_roi_geom_st is None) or self._mask is None:
            self._mask = self._identity_mask

    def reset(self):
        self._mask = None
        self._mask_updated = False
        self._averaged_image = None

        self._image_count = 0
        self._train_ids = []
        self._speckle_contrast_trend = []
        self._identity_mask = None

        self.ADU_threshold = 65

    def sources(self):
        return [(self._output_channel_fmt.format(i), self._ppt, 1) for i in range(16)]


class AveragedImageView(ImageViewF):
    def __init__(self, *, parent=None):
        """Initialization."""
        super().__init__(has_roi=True, parent=parent)

        self.circle_roi = CircleROI(2, (50, 50),
                                    radius=50,
                                    pen=FColor.mkPen("r", width=2, style=Qt.SolidLine))
        self.circle_roi.hide()
        self._rois.append(self.circle_roi)
        self._plot_widget.addItem(self.circle_roi)

    def updateF(self, data):
        self.setImage(data["averaged_image"])


class ImageView(ImageViewF):
    def __init__(self, display_key, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._display_key = display_key

    def updateF(self, data):
        self.setImage(data[self._display_key])

class SpeckleContrastCtrlWidget(_BaseAnalysisCtrlWidgetS):
    # Signal to notify that the circle ROI has been updated.
    # The parameters are: (activated, x, y, w, h)
    circleRoiUpdated = pyqtSignal(bool, int, int, int, int)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # This control is to be set later with addRoiCtrl()
        self.circle_roi_ctrl = None

        # Photon binning widgets
        self.photon_binning_chkbox = QCheckBox("Photon binning")
        self.adu_threshold_widget = SmartLineEdit("65")
        self.adu_threshold_widget.setValidator(QIntValidator(0, 1000))

        # Beam center widgets
        self.find_center_btn = QPushButton("Find beam center")
        beam_center_x_desc_label = QLabel("x: ")
        self.beam_center_x_label = QLabel()
        beam_center_y_desc_label = QLabel("y: ")
        self.beam_center_y_label = QLabel()
        beam_center_hbox = QHBoxLayout()
        beam_center_hbox.addWidget(self.find_center_btn)
        beam_center_hbox.setStretch(0, 6)
        beam_center_hbox.addWidget(beam_center_x_desc_label)
        beam_center_hbox.addWidget(self.beam_center_x_label)
        beam_center_hbox.addStretch(2)
        beam_center_hbox.addWidget(beam_center_y_desc_label)
        beam_center_hbox.addWidget(self.beam_center_y_label)
        beam_center_hbox.addStretch(2)

        self.layout().addRow(self.photon_binning_chkbox)
        self.layout().addRow("ADU threshold:", self.adu_threshold_widget)
        self.layout().addRow(beam_center_hbox)

    def addRoiCtrl(self, roi):
        self.circle_roi_ctrl = _SingleRoiCtrlWidget(roi, mediator=self, with_lock=False)
        self.circle_roi_ctrl.setLabel("Circle ROI")
        self.circle_roi_ctrl.roi_geometry_change_sgn.connect(self.onRoiGeometryChange)
        self.circle_roi_ctrl.notifyRoiParams()

        self.center_circle_roi_btn = QPushButton("Center around beam")

        hline = QFrame()
        hline.setFrameShape(QFrame.HLine)
        self.layout().addRow(hline)
        self.layout().addRow(self.circle_roi_ctrl)
        self.layout().addRow(self.center_circle_roi_btn)

    @pyqtSlot(object)
    def onRoiGeometryChange(self, value):
        if self.sender() is self.circle_roi_ctrl:
            self.circleRoiUpdated.emit(value[1], value[3], value[4], value[5], value[6])

class SpeckleContrastTrendlineView(PlotWidgetF):
    def __init__(self, *, parent=None):
        super().__init__(parent=parent)

        self.setLabel("left", "Value")
        self.setLabel("bottom", "Train index")
        self.setTitle("Speckle contrast")

        self._speckle_contrast = self.plotCurve(name="Speckle contrast")

    def updateF(self, data):
        self._speckle_contrast.setData(range(len(data["trendline"])), data["trendline"])


@create_special(SpeckleContrastCtrlWidget, SpeckleContrastProcessor, QThreadKbClient)
class SpeckleContrastWindow(_SpecialAnalysisBase):
    icon = "cam_view.png"
    _title = "Speckle contrast"
    _long_title = "Speckle contrast"

    def __init__(self, topic):
        super().__init__(topic)

        self._beam_center = None

        self._image_view = ImageView(display_key="image", parent=self)
        self._image_view.setTitle("Current image")
        self._avg_image_view = AveragedImageView(parent=self)
        self._avg_image_view.setTitle("Avg. image")
        self._trendline = SpeckleContrastTrendlineView(parent=self)

        self.initUI()
        self.initConnections()
        self.startWorker()

    def initUI(self):
        self._ctrl_widget_st.addRoiCtrl(self._avg_image_view.circle_roi)

        cw = self.centralWidget()

        hsplitter = QSplitter(Qt.Horizontal)
        hsplitter.addWidget(self._image_view)
        hsplitter.addWidget(self._avg_image_view)

        vsplitter = QSplitter(Qt.Vertical)
        vsplitter.addWidget(hsplitter)
        vsplitter.addWidget(self._trendline)
        vsplitter.setSizes([3 * self._TOTAL_H / 4, self._TOTAL_H / 4])

        cw.addWidget(vsplitter)
        cw.setSizes([self._TOTAL_W / 4, 3 * self._TOTAL_W / 4])

        self.resize(self._TOTAL_W, self._TOTAL_H)

    def initConnections(self):
        self._ctrl_widget_st.find_center_btn.clicked.connect(
            self._onFindCenterClicked
        )
        self._ctrl_widget_st.photon_binning_chkbox.stateChanged.connect(
            self._onPhotonBinningToggled
        )
        self._ctrl_widget_st.center_circle_roi_btn.clicked.connect(
            self._onCenterCircleRoiClicked
        )
        self._ctrl_widget_st.circleRoiUpdated.connect(
            self._onCircleRoiUpdated
        )
        self._ctrl_widget_st.adu_threshold_widget.value_changed_sgn.connect(
            self._onAduThresholdSet
        )

    @pyqtSlot()
    def _onFindCenterClicked(self):
        self._beam_center = self._worker_st.calc_beam_center()

        if self._beam_center is not None:
            self._ctrl_widget_st.beam_center_x_label.setText(str(self._beam_center[0]))
            self._ctrl_widget_st.beam_center_y_label.setText(str(self._beam_center[1]))

    @pyqtSlot(int)
    def _onPhotonBinningToggled(self, state):
        self._worker_st.photonBinningEnabled = (state == Qt.Checked)

    @pyqtSlot()
    def _onCenterCircleRoiClicked(self):
        if self._beam_center is None:
            self._worker_st.log.error("Beam center not found yet")
            return

        x, y = self._beam_center
        circle_roi = self._avg_image_view.circle_roi
        radius = circle_roi.size()[0] // 2
        circle_roi.setPos((x - radius, y - radius))

    @pyqtSlot(bool, int, int, int, int)
    def _onCircleRoiUpdated(self, activated, x, y, w, h):
        self._worker_st.circle_roi_geom_st = (x, y, w, h) if activated else None
        self._worker_st._mask_updated = True

    @pyqtSlot(object)
    def _onAduThresholdSet(self, new_threshold):
        self._worker_st.ADU_threshold = int(new_threshold)
