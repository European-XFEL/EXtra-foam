import time
import operator
import multiprocessing
from enum import Enum, auto

import pyFAI
import scipy
import numpy as np

from PyQt5.QtGui import QIntValidator
from PyQt5.QtCore import Qt, pyqtSlot, pyqtSignal
from PyQt5.QtWidgets import (QCheckBox, QSplitter, QPushButton, QLabel, QFrame,
                             QHBoxLayout, QTabWidget, QGraphicsLineItem,
                             QStackedWidget, QComboBox, QGridLayout, QWidget, QFileDialog)

import pasha
import extra_data
from extra_geom import AGIPD_1MGeometry

from extra_foam.algorithms import nanmean, welford_update, update_mean
from extra_foam.geometries import AGIPD_1MGeometryFast
from extra_foam.gui.ctrl_widgets import _SingleRoiCtrlWidget
from extra_foam.gui.ctrl_widgets.smart_widgets import (
    SmartBoundaryLineEdit, SmartLineEdit, SmartSliceLineEdit,
    SmartStringLineEdit
)

from extra_foam.gui.plot_widgets import ImageViewF, PlotWidgetF
from .special_analysis_base import ( create_special, QThreadKbClient,
                                     QThreadWorker, _BaseAnalysisCtrlWidgetS, _SpecialAnalysisBase, BlockTimer )

from ..gui.misc_widgets import FColor
from ..gui.plot_widgets.image_items import CircleROI


DEFAULT_GEOMETRY = [
    (-543.0, 657.0),
    (-607.0, -42.0),
    (532.0, -229.0),
    (588.0, 471.0),
]


image_count = pasha.array((1,))
current_train = None
min_values_train = None
max_values_train = None

averaged_train = None
binned_train = None
binned_averaged_train = None

m_values = None
s_values = None
var_values = None
std_dev_values = None

images = None

class NestedCircleROI(CircleROI):
    def __init__(self, radius_buffer, radius_checker, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._radius_checker = radius_checker
        self._radius_buffer = radius_buffer

    def setOtherRoi(self, roi):
        self._other_roi = roi

    def checkPointMove(self, handle, pos, modifiers):
        point = self.mapSceneToParent(pos)
        center_x = self.pos()[0] + self.size()[0] / 2
        center_y = self.pos()[1] + self.size()[0] / 2

        new_radius = np.sqrt((point.x() - center_x)**2 + (point.y() - center_y)**2)
        other_radius = self._other_roi.size()[0] / 2

        return self._radius_checker(new_radius, other_radius + self._radius_buffer)


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


class ImageOpcodes(Enum):
    MIN = auto()
    MAX = auto()
    AVG = auto()
    BINNED_AVG = auto()
    VARIANCE = auto()

class ImageCodes(Enum):
    CURRENT_TRAIN = 0
    AVG_TRAIN = 1
    MIN_VALUES_TRAIN = 2
    MAX_VALUES_TRAIN = 3
    VARIANCE = 4
    STD_DEV = 5

def process_train(opcode):
    global averaged_train, binned_averaged_train, m_values, s_values, var_values, std_dev_values

    if opcode is ImageOpcodes.MIN:
        np.minimum(min_values_train, current_train, out=min_values_train, dtype=np.float32)
    elif opcode is ImageOpcodes.MAX:
        np.maximum(max_values_train, current_train, out=max_values_train, dtype=np.float32)
    elif opcode is ImageOpcodes.AVG:
        update_mean(int(image_count[0]), current_train, averaged_train)
    elif opcode is ImageOpcodes.BINNED_AVG:
        update_mean(int(image_count[0]), binned_train, binned_averaged_train)
    elif opcode is ImageOpcodes.VARIANCE:
        welford_update(int(image_count[0]), current_train, m_values, s_values, var_values, std_dev_values)


def position_train(obj):
    imgcode, photon_binning = obj

    geom = AGIPD_1MGeometry.from_quad_positions(quad_pos=DEFAULT_GEOMETRY)

    if imgcode is ImageCodes.CURRENT_TRAIN:
        train = binned_train if photon_binning else current_train
    elif imgcode is ImageCodes.AVG_TRAIN:
        train = binned_averaged_train if photon_binning else averaged_train
    elif imgcode is ImageCodes.MIN_VALUES_TRAIN:
        train = min_values_train
    elif imgcode is ImageCodes.MAX_VALUES_TRAIN:
        train = max_values_train
    elif imgcode is ImageCodes.VARIANCE:
        train = var_values
    elif imgcode is ImageCodes.STD_DEV:
        train = std_dev_values

    mean_train = nanmean(train, axis=0)
    geom.position_modules_fast(mean_train, out=images[imgcode.value])

class SpeckleContrastProcessor(QThreadWorker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._photonBinningEnabled = False
        self.circle_roi_geom_st = None

        self._output_channel_fmt = "MID_DET_AGIPD1M-1/DET/{}CH0:xtdf"
        self._ppt = "image.data"

        self.setGeometryFromQuads(DEFAULT_GEOMETRY)

        self.reset()

    def onRoiGeometryChange(self, value):
        super().onRoiGeometryChange(value)

        self._mask_updated = True

    def setPhotonBinning(self, status):
        self._photonBinningEnabled = status
        self._averaged_image = self.bin_photons(current_train) if status else current_train

    def calc_contrast(self, clamped_data):
        """Derive pixel counts for k-unit photons from ADU, and from that a
           speckle contrast estimate.
           :param adu_data:       2darray (pixels) of ADU values
           :param adu_per_photon: constant factor to divide ADUs by
        """
        if self.pulse_of_interest >= clamped_data.shape[0]:
            raise ValueError("Pulse is invalid.")
        pulse = clamped_data[self.pulse_of_interest]
        reshaped_mask = np.zeros(pulse.shape, dtype=bool)

        try:
            self._geom.dismantle_all_modules(self._mask, reshaped_mask)
        except ValueError as e:
            self.log.info("Skipping speckle contrast calculation due to updated geometry")
            return

        selected_data = pulse[reshaped_mask == 1]
        k_mean = nanmean(selected_data)

        try:
            k_hist = np.histogram(selected_data[~np.isnan(selected_data)],
                                  np.arange(-0.5, np.nanmax(selected_data) + 1, 1),
                                  density=True)
        except ValueError:
            return 0.0

        # contrast equation
        k = k_hist[0]
        try:
            contrast = k[0] / k[1] - 1 / k_mean
        except IndexError:
            contrast = 0.0

        return contrast

    def calc_beam_center(self):
        if averaged_train is None:
            self.log.error("No images available")
            return None

        desired_shape = (16 * 512, 128)

        global image, mask, cone_width, ai
        image = pasha.array(desired_shape, dtype=np.float32)
        image[...] = nanmean(binned_averaged_train, axis=0).reshape(desired_shape)
        mask = pasha.array(desired_shape, dtype=np.float32)
        mask[...] = np.isnan(image).astype('bool')

        agipd = pyFAI.detectors.Detector(200e-6, 200e-6)
        agipd.aliases = ["AGIPD"]
        agipd.shape = image.shape
        agipd.mask = np.zeros(image.shape)
        agipd.IS_CONTINUOUS = False
        agipd.IS_FLAT = True
        agipd.set_pixel_corners(self._geom2.to_distortion_array())

        final_image_shape = self._mask.shape
        px, py = final_image_shape[1] // 2, final_image_shape[0] // 2
        ai = pyFAI.AzimuthalIntegrator(detector=agipd)
        ai.setFit2D(7800, px, py)
        ai.wavelength = 1.38e-10

        # Set constants
        number_of_cones = 8
        cone_width = 360 // number_of_cones
        import sys

        def get_SAXS_var(pos, worker_pool):
            x, y = pos
            with BlockTimer(f"{x:.2f}, {y:.2f}") as _:
                _I = worker_pool.map(integrate_cone, [(i, x, y) for i in range(number_of_cones)], chunksize=4)

            return nanmean(np.nanvar(np.log(np.array(_I)), 0), axis=0)

        with multiprocessing.Pool(processes=2) as worker_pool:
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
        tmp[tmp < 0] = 0

        return tmp

    def avg_and_position(self, img, out=None):
        tmp = self._geom.output_array_for_position_fast() if out is None else out
        self._geom.position_all_modules(nanmean(img, axis=0), out=tmp)
        return tmp

    def process(self, data):
        data, meta = data["raw"], data["meta"]

        tid = self.getTrainId(meta)
        rsd = { key.split()[0]: { self._ppt : data[key] } for key in data }

        global images, m_values, s_values, var_values, std_dev_values, image_count, current_train, averaged_train, binned_train, binned_averaged_train, min_values_train, max_values_train
        image_count += 1

        train = extra_data.stack_detector_data(rsd, self._ppt)

        # If this is the first image in the stream, initialize images
        if averaged_train is None:
            current_train = pasha.array(train.shape, dtype=np.float32)
            current_train[...] = train

            binned_train = pasha.array(train.shape, dtype=np.float32)
            self.bin_photons(current_train, out=binned_train)

            averaged_train = pasha.array(train.shape, dtype=np.float32)
            averaged_train[...] = current_train
            binned_averaged_train = pasha.array(train.shape, dtype=np.float32)
            binned_averaged_train[...] = binned_train

            min_values_train = pasha.array(current_train.shape, dtype=np.float32)
            min_values_train[...] = current_train
            max_values_train = pasha.array(current_train.shape, dtype=np.float32)
            max_values_train[...] = current_train

            m_values = pasha.array(current_train.shape, dtype=np.float32)
            m_values[...] = current_train
            s_values = pasha.array(current_train.shape, dtype=np.float32)
            s_values[...] = 0
            var_values = pasha.array(current_train.shape, dtype=np.float32)
            var_values[...] = current_train
            std_dev_values = pasha.array(current_train.shape, dtype=np.float32)
            std_dev_values[...] = current_train

            foo = self._geom.output_array_for_position_fast()
            images = pasha.array((6, *foo.shape), dtype=np.float32)
            images[...] = np.nan

            self._pool = multiprocessing.Pool(6)
        else:
            # Otherwise, update the images
            current_train[...] = train
            self.bin_photons(current_train, out=binned_train)

            with BlockTimer("processing", False) as _:
                self._pool.map(process_train, list(ImageOpcodes))

        with BlockTimer("avg_and_position") as _:
            self._pool.map(position_train, [(imgcode, self._photonBinningEnabled) for imgcode in list(ImageCodes)])

        self.updateMasks(images[ImageCodes.CURRENT_TRAIN.value])
        self._speckle_contrast_trend.append(self.calc_contrast(binned_train))

        return {
            "image": images[ImageCodes.CURRENT_TRAIN.value],
            "averaged_image": images[ImageCodes.AVG_TRAIN.value],
            "min_values": images[ImageCodes.MIN_VALUES_TRAIN.value],
            "max_values": images[ImageCodes.MAX_VALUES_TRAIN.value],
            "var_values": images[ImageCodes.VARIANCE.value],
            "std_values": images[ImageCodes.STD_DEV.value],
            "trendline": self._speckle_contrast_trend
        }

    def updateMasks(self, image):
        # If necessary, apply the user-defined mask
        if self._mask_updated:
            self._speckle_contrast_trend.clear()

            self._mask = np.zeros(image.shape, dtype=bool)

            if self._roi_geom_st:
                x, y, w, h = self._roi_geom_st
                self._mask[y : y + h, x : x + w] = 1
            elif self.outer_circle_roi_geom:
                x, y, w, h = self.outer_circle_roi_geom
                radius = w // 2
                center_x = x + radius
                center_y = y + radius

                py, px = np.ogrid[0:image.shape[0], 0:image.shape[1]]
                radius_mask = (px - center_x)**2 + (py - center_y)**2 <= radius**2
                self._mask[radius_mask] = True

                # Repeat for the inner circle
                x, y, w, h = self.inner_circle_roi_geom
                radius = w // 2
                center_x = x + radius
                center_y = y + radius
                radius_mask = (px - center_x)**2 + (py - center_y)**2 <= radius**2
                self._mask[radius_mask] = False

            self._mask_updated = False

        elif (self._roi_geom_st is None and self.outer_circle_roi_geom is None) or self._mask is None:
            self._mask = np.ones(image.shape, dtype=bool)

    def reset(self):
        self._mask = None
        self._mask_updated = False

        global averaged_train, image_count
        averaged_train = None

        self.outer_circle_roi_geom = None
        self.inner_circle_roi_geom = None

        image_count[0] = 0
        self._train_ids = []
        self._speckle_contrast_trend = []

        self.pulse_of_interest = 10
        self.ADU_threshold = 65

    def sources(self):
        return [(self._output_channel_fmt.format(i), self._ppt, 1) for i in range(16)]

    def setGeometryFromFile(self, file_path):
        self._geom2 = AGIPD_1MGeometry.from_crystfel_geom(file_path)
        self._geom = AGIPD_1MGeometryFast.from_crystfel_geom(file_path)
        self._mask_updated = True

    def setGeometryFromQuads(self, positions):
        self._geom2 = AGIPD_1MGeometry.from_quad_positions(quad_pos=positions)
        modules = [[fragment.corner_pos for fragment in module] for module in self._geom2.modules]
        self._geom = AGIPD_1MGeometryFast(modules)
        self._mask_updated = True


class AveragedImageView(ImageViewF):
    _CROSSHAIR_HALF_LEN = 40

    def __init__(self, *, parent=None):
        """Initialization."""
        super().__init__(has_roi=True, parent=parent)

        crosshair_pen = FColor.mkPen("i")
        self._horiz_beam_crosshair = QGraphicsLineItem(0., 0., 1., 1.)
        self._horiz_beam_crosshair.hide()
        self._horiz_beam_crosshair.setPen(crosshair_pen)
        self._vertical_beam_crosshair = QGraphicsLineItem(0., 0., 1., 1.)
        self._vertical_beam_crosshair.hide()
        self._vertical_beam_crosshair.setPen(crosshair_pen)

        self.circle_roi = NestedCircleROI(5, operator.gt,
                                          2,
                                          (50, 50),
                                          radius=50,
                                          pen=FColor.mkPen("r", width=2, style=Qt.SolidLine))

        outer_roi_pos = self.circle_roi.pos()
        outer_roi_size = self.circle_roi.size()
        inner_roi_radius = 20
        inner_circle_pos = (self.circle_roi.pos()[0] + (50 - 40) / 2, self.circle_roi.pos()[1] + (50 - 40) / 2)
        self.circle_roi2 = NestedCircleROI(-5, operator.lt,
                                           3,
                                           inner_circle_pos,
                                           radius=40,
                                           pen=FColor.mkPen("r", width=2, style=Qt.SolidLine))
        self.circle_roi2.setOtherRoi(self.circle_roi)
        self.circle_roi.setOtherRoi(self.circle_roi2)

        self.circle_roi.sigRegionChanged.connect(self._onCircleRoiChanged)
        self.circle_roi2.sigRegionChanged.connect(self._onCircleRoiChanged)

        self.circle_roi.hide()
        self.circle_roi2.hide()
        self._rois.append(self.circle_roi)
        self._rois.append(self.circle_roi2)

        self._plot_widget.addItem(self.circle_roi)
        self._plot_widget.addItem(self.circle_roi2)
        self._plot_widget.addItem(self._horiz_beam_crosshair)
        self._plot_widget.addItem(self._vertical_beam_crosshair)

    @pyqtSlot(object)
    def _onCircleRoiChanged(self, roi):
        parent = self.circle_roi if roi == self.circle_roi else self.circle_roi2
        child = self.circle_roi2 if roi == self.circle_roi else self.circle_roi

        radius_diff = (parent.size()[0] - child.size()[0]) / 2
        new_child_pos = (parent.pos()[0] + radius_diff, parent.pos()[1] + radius_diff)
        child.setPos(new_child_pos, update=False)

    def updateF(self, data):
        self.setImage(data["averaged_image"])

    def setCrossHairPos(self, x, y):
        self._horiz_beam_crosshair.setLine(x - self._CROSSHAIR_HALF_LEN, y,
                                           x + self._CROSSHAIR_HALF_LEN, y)
        self._vertical_beam_crosshair.setLine(x, y - self._CROSSHAIR_HALF_LEN,
                                              x, y + self._CROSSHAIR_HALF_LEN)
        self._horiz_beam_crosshair.show()
        self._vertical_beam_crosshair.show()


class ImageView(ImageViewF):
    def __init__(self, display_key, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._display_key = display_key

    def updateF(self, data):
        self.setImage(data[self._display_key])

class GeometrySource(Enum):
    FILE = 0
    QUAD_POSITIONS = 1

class SpeckleContrastCtrlWidget(_BaseAnalysisCtrlWidgetS):
    # Signal to notify that the circle ROI has been updated.
    # The parameters are: (activated, x, y, w, h)
    circleRoiUpdated = pyqtSignal(bool, int, int, int, int)

    # List of tuples of SmartLineEdit's, each tuple holding the x and y
    # coordinate of a quadrant.
    quad_position_les = []

    # Path to a CrystFEL geometry file
    geometry_file_path = None

    # Which geometry source the user has selected
    geometry_source = GeometrySource.QUAD_POSITIONS

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # This control is to be set later with addRoiCtrl()
        self.circle_roi_ctrl = None

        # Geometry widgets
        quad_positions_layout = QGridLayout()
        quad_positions_layout.addWidget(QLabel("x"), 0, 1)
        quad_positions_layout.addWidget(QLabel("y"), 0, 2)
        for quadrant in range(4):
            x_le = SmartLineEdit(str(DEFAULT_GEOMETRY[quadrant][0]))
            y_le = SmartLineEdit(str(DEFAULT_GEOMETRY[quadrant][1]))
            self.quad_position_les.append((x_le, y_le))

            quad_positions_layout.addWidget(QLabel(str(quadrant)), quadrant + 1, 0)
            quad_positions_layout.addWidget(x_le, quadrant + 1, 1)
            quad_positions_layout.addWidget(y_le, quadrant + 1, 2)

        quad_positions_widget = QWidget()
        quad_positions_widget.setLayout(quad_positions_layout)

        geometry_file_btn = QPushButton("Select geometry file")
        geometry_file_btn.clicked.connect(self._onGeometryButtonClicked)
        self._geometry_file_path_le = SmartLineEdit()
        geometry_file_layout = QHBoxLayout()
        geometry_file_layout.addWidget(geometry_file_btn)
        geometry_file_layout.addWidget(self._geometry_file_path_le)
        geometry_file_widget = QWidget()
        geometry_file_widget.setLayout(geometry_file_layout)

        geometry_stackwidget = QStackedWidget()
        geometry_stackwidget.addWidget(quad_positions_widget)
        geometry_stackwidget.addWidget(geometry_file_widget)

        geometry_combobox = QComboBox()
        geometry_combobox.addItem("Quad positions")
        geometry_combobox.addItem("CrystFEL file")
        geometry_combobox.activated.connect(geometry_stackwidget.setCurrentIndex)
        geometry_combobox.activated.connect(self._onGeometrySourceChanged)

        self.apply_geometry_btn = QPushButton("Apply")

        hline = QFrame()
        hline.setFrameShape(QFrame.HLine)

        # Pulse-of-interest widgets
        self.pulse_le = SmartLineEdit("10")
        self.pulse_le.setValidator(QIntValidator(0, 2700))

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

        self.layout().addRow("Geometry source: ", geometry_combobox)
        self.layout().addRow(geometry_stackwidget)
        self.layout().addRow(self.apply_geometry_btn)
        self.layout().addRow(hline)
        self.layout().addRow("Pulse of interest: ", self.pulse_le)
        self.layout().addRow(self.photon_binning_chkbox)
        self.layout().addRow("ADU threshold:", self.adu_threshold_widget)
        self.layout().addRow(beam_center_hbox)

    def addRoiCtrl(self, roi, roi2):
        self.roi2 = roi2

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

    @pyqtSlot(int)
    def _onGeometrySourceChanged(self, index):
        if index == 0:
            self.geometry_source = GeometrySource.QUAD_POSITIONS
        elif index == 1:
            self.geometry_source = GeometrySource.FILE

        print(self.geometry_source)

    @pyqtSlot()
    def _onGeometryButtonClicked(self):
        result_tuple = QFileDialog.getOpenFileName(self, "Select geometry file", filter="CrystFEL files (*.geom)")
        if result_tuple is not None:
            self.geometry_file_path = result_tuple[0]
            self._geometry_file_path_le.setText(result_tuple[0])


class SpeckleContrastTrendlineView(PlotWidgetF):
    def __init__(self, *, parent=None):
        super().__init__(parent=parent)

        self.setLabel("left", "Value")
        self.setLabel("bottom", "Train index")
        self.setTitle("Speckle contrast")

        self._speckle_contrast = self.plotScatter(name="Speckle contrast")

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
        self._ctrl_widget_st.addRoiCtrl(self._avg_image_view.circle_roi, self._avg_image_view.circle_roi2)

        cw = self.centralWidget()

        hsplitter = QSplitter(Qt.Horizontal)
        hsplitter.addWidget(self._image_view)
        hsplitter.addWidget(self._avg_image_view)

        vsplitter = QSplitter(Qt.Vertical)
        vsplitter.addWidget(hsplitter)
        vsplitter.addWidget(self._trendline)
        vsplitter.setSizes([3 * self._TOTAL_H / 4, self._TOTAL_H / 4])

        max_value_view = ImageView(display_key="max_values", parent=self)
        max_value_view.setTitle("Max value")
        min_value_view = ImageView(display_key="min_values", parent=self)
        min_value_view.setTitle("Min value")
        std_value_view = ImageView(display_key="std_values", parent=self)
        std_value_view.setTitle("Standard deviation")
        var_value_view = ImageView(display_key="var_values", parent=self)
        var_value_view.setTitle("Variance")
        stats_top_hsplitter = QSplitter(Qt.Horizontal)
        stats_top_hsplitter.addWidget(max_value_view)
        stats_top_hsplitter.addWidget(min_value_view)
        stats_bottom_hsplitter = QSplitter(Qt.Horizontal)
        stats_bottom_hsplitter.addWidget(std_value_view)
        stats_bottom_hsplitter.addWidget(var_value_view)
        stats_vsplitter = QSplitter(Qt.Vertical)
        stats_vsplitter.addWidget(stats_top_hsplitter)
        stats_vsplitter.addWidget(stats_bottom_hsplitter)

        tab_widget = QTabWidget()
        tab_widget.addTab(vsplitter, "Overview")
        tab_widget.addTab(stats_vsplitter, "Statistics")

        cw.addWidget(tab_widget)
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
        self._avg_image_view.circle_roi2.sigRegionChangeFinished.connect(
            lambda roi: self._onCircleRoiUpdated(True, *roi.pos(), *roi.size())
        )
        self._ctrl_widget_st.circleRoiUpdated.connect(
            self._onCircleRoiUpdated
        )
        self._ctrl_widget_st.adu_threshold_widget.value_changed_sgn.connect(
            self._onAduThresholdSet
        )
        self._ctrl_widget_st.pulse_le.value_changed_sgn.connect(
            self._onPoiChanged
        )
        self._ctrl_widget_st.apply_geometry_btn.clicked.connect(
            self._onGeometryChanged
        )

    @pyqtSlot()
    def _onFindCenterClicked(self):
        with BlockTimer("Beam center") as _:
            self._beam_center = self._worker_st.calc_beam_center()

        if self._beam_center is not None:
            self._ctrl_widget_st.beam_center_x_label.setText(str(self._beam_center[0]))
            self._ctrl_widget_st.beam_center_y_label.setText(str(self._beam_center[1]))
            self._avg_image_view.setCrossHairPos(self._beam_center[0], self._beam_center[1])

    @pyqtSlot(int)
    def _onPhotonBinningToggled(self, state):
        self._worker_st.setPhotonBinning(state == Qt.Checked)

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
        if activated:
            self._worker_st.outer_circle_roi_geom = (*self._avg_image_view.circle_roi.pos(), *self._avg_image_view.circle_roi.size())
            self._worker_st.inner_circle_roi_geom = (*self._avg_image_view.circle_roi2.pos(), *self._avg_image_view.circle_roi2.size())
        else:
            self._worker_st.outer_circle_roi_geom = None
        self._worker_st._mask_updated = True

        if activated:
            self._avg_image_view.circle_roi2.show()
        else:
            self._avg_image_view.circle_roi2.hide()

    @pyqtSlot(object)
    def _onAduThresholdSet(self, new_threshold):
        self._worker_st.ADU_threshold = int(new_threshold)

    @pyqtSlot(object)
    def _onPoiChanged(self, new_poi):
        self._worker_st.pulse_of_interest = int(new_poi)

    @pyqtSlot()
    def _onGeometryChanged(self):
        if self._ctrl_widget_st.geometry_source is GeometrySource.FILE:
            self._worker_st.setGeometryFromFile(self._ctrl_widget_st.geometry_file_path)
        elif self._ctrl_widget_st.geometry_source is GeometrySource.QUAD_POSITIONS:
            positions = [(float(pos_les[0].text()), float(pos_les[1].text())) for pos_les in self._ctrl_widget_st.quad_position_les]
            self._worker_st.setGeometryFromQuads(positions)
