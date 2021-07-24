"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Cammille Carinan <cammille.carinan@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from extra_data import stack_detector_data
import numpy as np
from pathlib import Path
from scipy.optimize import curve_fit

from extra_foam import geometries
from extra_foam.algorithms import correct_image_data, nanmean_image_data
from extra_foam.config import config

from extra_foam.special_suite import logger
from .models import (
    Device, Projection, Pulse, Train, Trains, ViewData)
from ..special_analysis_base import profiler, QThreadWorker


# geometry config
GEOMETRY_COORDINATES = [
        [-124.1, 3.112],
        [-133.068, -110.604],
        [0.988, -125.236],
        [4.528, -4.912]
    ]
GEOMETRY_FILE = str(Path(geometries.__file__).parent / 'dssc_geo_june19.h5')

# extra-foam config
IMAGE_DTYPE = config['SOURCE_PROC_IMAGE_DTYPE']
RAW_IMAGE_DTYPE = config['SOURCE_RAW_IMAGE_DTYPE']
AXES = [-2, -1]


class SpotSizeProcessor(QThreadWorker):
    """Gotthard pump-probe analysis processor.

    Attributes:
        _output_channel (str): output channel name.
        _on_slicer (slice): a slicer used to slice on-pulses in a train.
        _off_slicer (slice): a slicer used to slice off-pulses in a train.
        _poi_index (int): index of the pulse of interest for pump-probe.
        _dark_slicer (slice): a slicer used to slice dark pulses in a train.
        _dark_poi_index (int): index of the pulse of interest for dark.
        _vfom_ma (numpy.ndarray): moving average of the vector figure-of-merit
            data. Shape=(pulses, pixels)
    """

    trains = Trains()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._detectors = []
        self._ppt = "data.image.data"
        self._device = ''
        self._slice = slice(None, None)

        # Load geometry
        self._geom = geometries.load_geometry(
            detector="DSSC",
            stack_only=False,
            filepath=GEOMETRY_FILE,
            coordinates=GEOMETRY_COORDINATES,
            n_modules=16
        )
        self._dark = None
        self._subtract_dark = False

        # Arrays
        self._virtual_array = None
        self._assembled_array = None

        # Variables
        self._pulse_num = 0
        self._analysis_type = "Train"
        self._axis = 0
        self._result = None

    # ----------------------------------------------------------------------
    # Slots

    def onLoadDarkRun(self, dirpath):
        """Override."""
        self._dark = np.load(dirpath).astype(np.float32)

    def onSubtractDark(self, subtract):
        self._subtract_dark = subtract

    def onDetectorChanged(self, detector: str):
        detectors = [detector]
        if '*' in detector:
            detector = detector.replace('*', '{}')
            detectors = [detector.format(i) for i in range(2)]

        self._detectors = detectors

    def onDeviceChanged(self, device: str):
        self._device = device

    def onPulseSliceChanged(self, pulse_slice: slice):
        self._slice = pulse_slice

    def onPulseNumChanged(self, pulse_num: int):
        self._pulse_num = pulse_num

    def onAnalysisTypeChanged(self, analysis_type):
        self._analysis_type = analysis_type

    def onAxisChanged(self, axis: int):
        self._axis = AXES[axis]

    def onHistBinsChanged(self, n_bins: int):
        self.trains.hist_bins = n_bins

    # ----------------------------------------------------------------------
    # Overridden methods

    def sources(self):
        """Override."""
        sources = []

        # Add detectors
        sources += [(detector, self._ppt, 1) for detector in self._detectors]

        # Add motor
        if self._device:
            sources += [(self._device, "actualPosition", 1)]
        return sources

    @profiler("Spot size grating Processor")
    def process(self, data):
        """Override."""
        data, meta = data["raw"], data["meta"]
        view_data = ViewData()

        # 1. Assemble data
        if len(self._detectors) > 1:
            train_images = self._assemble(data)
        else:
            # array = data[f"{self._detectors[-1]} {self._ppt}"]
            # train_images = array.squeeze(axis=1).astype(np.float32)
            array = data[f"{self._detectors[-1]} {self._ppt}"]
            train_images = array.reshape(1, *array.shape).astype(np.float32)

        # 2. Subtract dark image. Has to be done before pulse slicing
        if self._subtract_dark:
            correct_image_data(train_images, offset=self._dark,
                               intradark=True, detector="DSSC")

        # # 3. Slice
        # train_images = train_images[self._slice]
        # # Check if number of pulses have been changed
        # num_pulses = len(train_images)
        # if num_pulses != self.trains.num_pulses:
        #     self.trains.num_pulses = num_pulses
        #     self.trains.reset()

        # 3. Calculate pulse-resolved analysis
        pulses = []
        for pulse_num, pulse_image in enumerate(train_images):
            pulse, view = self._process_image(pulse_image)
            pulses.append(pulse)
            # Save the view
            if self._analysis_type == "Pulse" and pulse_num == self._pulse_num:
                view_data = view

        # 4. Calculate train-resolved analysis
        if self._analysis_type == "Train":
            # Calculate train values
            image = nanmean_image_data(train_images)
            _, view_data = self._process_image(image)

        # 5. Get other data
        device = Device(name=self._device,
                        value=data[f"{self._device} actualPosition"])

        # 6. Record
        train_data = Train(trainId=self.getTrainId(meta),
                           device=device,
                           pulses=pulses)

        # 7. Update train
        trains = self.trains
        trains.add(train_data)

        # 8. Add more bender scan view data
        widths = trains.width_train.mean
        if self._analysis_type == "Pulse":
            widths = trains.width_pulse.get_values(self._pulse_num)
        view_data.widths = widths

        return dict(**self.trains.data, **{"view": view_data.data})

    def reset(self):
        """Override."""
        self.trains.reset()

    # ----------------------------------------------------------------------
    # Helper methods

    def _assemble(self, data):
        # 0. Get and rearrange the detector data to train dict
        train = {}
        for detector in self._detectors:
            image = data[f"{detector} {self._ppt}"]
            train[detector] = {self._ppt: image}
        n_pulses = image.shape[0]

        # 1. Get virtual data as template
        virtual = self._virtual_array
        if virtual is None or virtual.shape[0] != n_pulses:
            self._virtual_array = virtual = get_modules_file(train, self._ppt)

        # 2. (Re)Create assembled array
        assembled = self._assembled_array
        if assembled is None or assembled.shape[0] != n_pulses:
            assembled = self._geom.output_array_for_position_fast(
                extra_shape=(n_pulses,), dtype=IMAGE_DTYPE)
            self._assembled_array = assembled

        # 2. Something
        self._geom.position_all_modules(virtual, out=assembled)

        return assembled

    @profiler("Gaussian")
    def _gaussian(self, x, y):
        # 5. Fit projection with gaussian fitting
        pos, width = np.nan, np.nan
        fit = None
        if x.size != 0:
            try:
                p0, _ = curve_fit(gaussian, x, y, p0=initial_p0(x, y))
                fit = gaussian(x, *p0)
                if fit is None or np.isnan(fit.sum()):
                    return pos, width, None

                pos = p0[1]
                width = fwhm(p0[2])
            except (TypeError, RuntimeError):
                logger.debug("Fit did not converge.")

        return pos, width, fit

    @profiler("Process Image")
    def _process_image(self, image):
        proj = np.nanmean(self.getRoiData(image), axis=self._axis)
        x = np.where(np.isfinite(proj))[0]
        y = adjust_baseline(proj[x])
        pos, width, fit = self._gaussian(x, y)

        pulse = Pulse(pos=pos, width=width)
        view = ViewData(image=image,
                        proj=Projection(x, y, fit, pos, width),)
        return pulse, view


# Assembling logic

def get_modules_bridge(data, src):
    """
    In the file, the data is separated into arrays of different
    modules. The layout of data for each module is:
    - calibrated, (memory cells, x, y)
    - raw, (memory cells, 1, x, y)

    - calibrated, "image.data", (modules, x, y, memory cells)
    - raw, "image.data", (modules, x, y, memory cells)
    -> (memory cells, modules, y, x)
    """
    return np.moveaxis(np.moveaxis(data[src], 3, 0), 3, 2)


def get_modules_file(data, src):
    """
    - calibrated, "image.data", (memory cells, modules, y, x)
    - raw, "image.data", (memory cell, 1, modules, y, x)
    -> (memory cells, modules, y, x)
    """
    modules_data = stack_detector_data(data, src, real_array=False)

    dtype = modules_data.dtype
    if dtype == IMAGE_DTYPE:
        return modules_data
    if dtype == RAW_IMAGE_DTYPE:
        return modules_data.squeeze(axis=1)

    raise RuntimeError(f"Unknown detector data type: {dtype}!")


# Gaussian

def initial_p0(x, y):
    x0 = np.average(x, weights=y)
    sx = np.sqrt(np.average((x - x0) ** 2, weights=y))
    return y.max(), x0, sx, y.min()


def gaussian(x, a, b, c, d):
    return a * np.exp(-(x - b) ** 2 / (2 * c ** 2)) + d


def fwhm(sigma):
    return 2.355 * sigma


def adjust_baseline(y, baseline=None):
    """Adjusts baseline to zero to the minimum or to the specified value"""
    if baseline is None:
        baseline = y.min()

    return y - baseline
