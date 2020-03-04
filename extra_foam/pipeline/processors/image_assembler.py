"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from abc import ABC, abstractmethod

import json
import numpy as np

from extra_data import stack_detector_data

from .base_processor import _BaseProcessor, _RedisParserMixin
from ..exceptions import AssemblingError
from ...config import config, GeomAssembler, DataSource
from ...database import Metadata as mt
from ...ipc import process_logger as logger
from ...utils import profiler


_IMAGE_DTYPE = config['SOURCE_PROC_IMAGE_DTYPE']
_RAW_IMAGE_DTYPE = config['SOURCE_RAW_IMAGE_DTYPE']


import re


def stack_detector_data(train, data, axis=-3, modules=16, fillvalue=np.nan,
                        real_array=True):
    """Stack data from detector modules in a train.

    TODO: remove when extra-data has a new release

    Parameters
    ----------
    train: dict
        Train data.
    data: str
        The path to the device parameter of the data you want to stack, e.g. 'image.data'.
    axis: int
        Array axis on which you wish to stack (default is -3).
    modules: int
        Number of modules composing a detector (default is 16).
    fillvalue: number
        Value to use in place of data for missing modules. The default is nan
        (not a number) for floating-point data, and 0 for integers.
    real_array: bool
        If True (default), copy the data together into a real numpy array.
        If False, avoid copying the data and return a limited array-like wrapper
        around the existing arrays. This is sufficient for assembling images
        using detector geometry, and allows better performance.
    Returns
    -------
    combined: numpy.array
        Stacked data for requested data path.
    """

    if not train:
        raise ValueError("No data")

    dtypes, shapes, empty_mods = set(), set(), set()
    modno_arrays = {}
    for device in train:
        det_mod_match = re.search(r'/DET/(\d+)CH', device)
        if not det_mod_match:
            raise ValueError("Non-detector source: {}".format(device))
        modno = int(det_mod_match.group(1))

        try:
            array = train[device][data]
        except KeyError:
            continue
        dtypes.add(array.dtype)
        shapes.add(array.shape)
        modno_arrays[modno] = array

    if len(dtypes) > 1:
        raise ValueError("Arrays have mismatched dtypes: {}".format(dtypes))
    if len(shapes) > 1:
        s1, s2, *_ = sorted(shapes)
        if len(shapes) > 2 or (s1[0] != 0) or (s1[1:] != s2[1:]):
            raise ValueError("Arrays have mismatched shapes: {}".format(shapes))
        empty_mods = {n for n, a in modno_arrays.items() if a.shape == s1}
        for modno in empty_mods:
            del modno_arrays[modno]
        shapes.remove(s1)
    if max(modno_arrays) >= modules:
        raise IndexError("Module {} is out of range for a detector with {} modules"
                         .format(max(modno_arrays), modules))

    dtype = dtypes.pop()
    shape = shapes.pop()
    stack = StackView(
        modno_arrays, modules, shape, dtype, fillvalue, stack_axis=axis
    )
    if real_array:
        return stack.asarray()

    return stack


class StackView:
    """Limited array-like object holding detector data from several modules.
    Access is limited to either a single module at a time or all modules
    together, but this is enough to assemble detector images.

    TODO: remove when extra-data has a new release
    """
    def __init__(self, data, nmodules, mod_shape, dtype, fillvalue,
                 stack_axis=-3):
        self._nmodules = nmodules
        self._data = data  # {modno: array}
        self.dtype = dtype
        self._fillvalue = fillvalue
        self._mod_shape = mod_shape
        self.ndim = len(mod_shape) + 1
        self._stack_axis = stack_axis
        if self._stack_axis < 0:
            self._stack_axis += self.ndim
        sax = self._stack_axis
        self.shape = mod_shape[:sax] + (nmodules,) + mod_shape[sax:]

    def __repr__(self):
        return "<VirtualStack (shape={}, {}/{} modules, dtype={})>".format(
            self.shape, len(self._data), self._nmodules, self.dtype,
        )

    # Multidimensional slicing
    def __getitem__(self, slices):
        if not isinstance(slices, tuple):
            slices = (slices,)

        missing_dims = self.ndim - len(slices)
        if Ellipsis in slices:
            ix = slices.index(Ellipsis)
            missing_dims += 1
            slices = slices[:ix] + (slice(None, None),) * missing_dims + slices[ix + 1:]
        else:
            slices = slices + (slice(None, None),) * missing_dims

        modno = slices[self._stack_axis]
        mod_slices = slices[:self._stack_axis] + slices[self._stack_axis + 1:]

        if isinstance(modno, int):
            if modno < 0:
                modno += self._nmodules
            return self._get_single_mod(modno, mod_slices)
        elif modno == slice(None, None):
            return self._get_all_mods(mod_slices)
        else:
            raise Exception(
                "VirtualStack can only slice a single module or all modules"
            )

    def _get_single_mod(self, modno, mod_slices):
        try:
            mod_data = self._data[modno]
        except KeyError:
            if modno >= self._nmodules:
                raise IndexError(modno)
            mod_data = np.full(self._mod_shape, self._fillvalue, self.dtype)
            self._data[modno] = mod_data

        # Now slice the module data as requested
        return mod_data[mod_slices]

    def _get_all_mods(self, mod_slices):
        new_data = {modno: self._get_single_mod(modno, mod_slices)
                    for modno in self._data}
        new_mod_shape = list(new_data.values())[0].shape
        return StackView(new_data, self._nmodules, new_mod_shape, self.dtype,
                         self._fillvalue)

    def asarray(self):
        """Copy this data into a real numpy array
        Don't do this until necessary - the point of using VirtualStack is to
        avoid copying the data unnecessarily.
        """
        start_shape = (self._nmodules,) + self._mod_shape
        arr = np.full(start_shape, self._fillvalue, dtype=self.dtype)
        for modno, data in self._data.items():
            arr[modno] = data
        return np.moveaxis(arr, 0, self._stack_axis)

    def squeeze(self, axis=None):
        """Drop axes of length 1 - see numpy.squeeze()"""
        if axis is None:
            slices = [0 if d == 1 else slice(None, None) for d in self.shape]
        elif isinstance(axis, (int, tuple)):
            if isinstance(axis, int):
                axis = (axis,)

            slices = [slice(None, None)] * self.ndim

            for ax in axis:
                try:
                    slices[ax] = 0
                except IndexError:
                    raise np.AxisError(
                        "axis {} is out of bounds for array of dimension {}"
                        .format(ax, self.ndim)
                    )
                if self.shape[ax] != 1:
                    raise ValueError("cannot squeeze out an axis with size != 1")
        else:
            raise TypeError("axis={!r} not supported".format(axis))
        return self[tuple(slices)]


class ImageAssemblerFactory(ABC):

    class BaseAssembler(_BaseProcessor, _RedisParserMixin):
        """Abstract ImageAssembler class.

        Attributes:
            _require_geom (bool): whether a Geometry is required to assemble
                the detector modules.
            _stack_only (bool): whether simply stack all modules seamlessly
                together.
            _assembler_type (GeomAssembler): Type of geometry assembler,
                which can be EXtra-foam or EXtra-geom.
            _geom_file (str): full path of the geometry file.
            _quad_position (list): (x, y) coordinates for the corners of 4
                quadrants.
            _geom: geometry instance in use.
            _out_array (numpy.ndarray): buffer to store the assembled modules.
        """
        def __init__(self):
            """Initialization."""
            super().__init__()

            self._require_geom = config['REQUIRE_GEOMETRY']
            self._stack_only = False
            self._assembler_type = None
            self._geom_file = None
            self._quad_position = None
            self._geom = None
            self._out_array = None

        def update(self):
            if self._require_geom:
                cfg = self._meta.hget_all(mt.GEOMETRY_PROC)

                assembler_type = GeomAssembler(int(cfg["assembler"]))
                stack_only = cfg["stack_only"] == 'True'
                if stack_only:
                    # ignore "assembler" setup
                    assembler_type = GeomAssembler.OWN

                geom_file = cfg["geometry_file"]
                quad_positions = json.loads(cfg["quad_positions"],
                                            encoding='utf8')

                # reload geometry if any of the following 4 fields changed
                if stack_only != self._stack_only or \
                        assembler_type != self._assembler_type or \
                        geom_file != self._geom_file or \
                        quad_positions != self._quad_position:

                    self._stack_only = stack_only
                    self._assembler_type = assembler_type
                    self._geom_file = geom_file
                    self._quad_position = quad_positions

                    self._geom = None  # reset first
                    self._load_geometry(geom_file, quad_positions)

                    if not stack_only:
                        logger.info(f"Loaded geometry from {geom_file} with "
                                    f"quadrant positions {quad_positions}")

        @abstractmethod
        def _get_modules_bridge(self, data, src):
            """Get modules data from bridge."""
            pass

        @abstractmethod
        def _get_modules_file(self, data, src):
            """Get modules data from file."""
            pass

        def _load_geometry(self, filepath, quad_positions):
            """Load geometry from file.

            Required for modular detectors which must be assembled with
            a geometry.

            If the assembler type is not defined, it uses EXtra-geom by default.

            :param str filepath: path of the geometry file.
            :param tuple quad_positions: quadrant coordinates.
            """
            raise NotImplementedError

        def _assemble(self, modules):
            """Assemble modules data into assembled image data.

            :param array-like modules: modules data. shape = (memory cells,
                modules, y, x) for pulse-resolved detectors and (y, x) for
                train-resolved detectors.

            :return numpy.ndarray assembled: assembled detector image(s).
                shape = (memory cells, y, x) for pulse-resolved detectors
                and (y, x) for train resolved detectors.
            """
            image_dtype = config["SOURCE_PROC_IMAGE_DTYPE"]
            if self._geom is not None:
                n_modules = modules.shape[1]
                if n_modules == 1:
                    # single module operation
                    return modules.astype(image_dtype).squeeze(axis=1)

                n_pulses = modules.shape[0]
                if self._out_array is None or self._out_array.shape[0] != n_pulses:
                    self._out_array = self._geom.output_array_for_position_fast(
                        extra_shape=(n_pulses, ), dtype=image_dtype)

                try:
                    self._geom.position_all_modules(modules, out=self._out_array)
                # EXtra-foam raises ValueError while EXtra-geom raises
                # AssertionError if the shape of the output array does not
                # match the expected one, e.g. after a change of quadrant
                # positions during runtime.
                except (ValueError, AssertionError):
                    # recreate the output array
                    self._out_array = self._geom.output_array_for_position_fast(
                        extra_shape=(n_pulses, ), dtype=image_dtype)
                    self._geom.position_all_modules(modules, out=self._out_array)

                return self._out_array

            # temporary workaround for Pulse resolved JungFrau without geometry
            if config["DETECTOR"] == "JungFrauPR":
                shape = modules.shape
                # Stacking modules vertically along y axis.
                return modules.reshape(shape[0], -1, shape[-1])

            # For train-resolved detector, assembled is a reference
            # to the array data received from the pyzmq. This array data
            # is only readable since the data is owned by a pointer in
            # the zmq message (it is not copied). However, other data
            # like data['metadata'] is writeable.
            # FIXME: why once a while this takes a few ms???
            return modules.astype(image_dtype)

        @profiler("Image Assembler")
        def process(self, data):
            """Override."""
            meta = data['meta']
            raw = data['raw']
            catalog = data["catalog"]

            src = catalog.main_detector
            src_type = meta[src]['source_type']
            try:
                if src_type == DataSource.FILE:
                    modules_data = self._get_modules_file(raw, src)
                elif src_type == DataSource.BRIDGE:
                    modules_data = self._get_modules_bridge(raw, src)
                else:
                    raise ValueError(f"Unknown source type: {src_type}")

                # Remove raw detector data since we do not want to serialize
                # it and send around.
                raw[src] = None

            except (ValueError, IndexError, KeyError) as e:
                raise AssemblingError(repr(e))

            shape = modules_data.shape
            ndim = len(shape)
            try:
                n_modules = config["NUMBER_OF_MODULES"]
                module_shape = config["MODULE_SHAPE"]

                # check module shape
                # (BaslerCamera has module shape (-1, -1))
                if module_shape[0] > 0 and shape[-2:] != module_shape:
                    raise ValueError(f"Expected module shape {module_shape}, "
                                     f"but get {shape[-2:]} instead!")

                # check number of modules
                if ndim >= 3 and shape[-3] != n_modules:
                    n_modules_actual = shape[-3]
                    if config["DETECTOR"] != "JungFrauPR":
                        # allow single module operation
                        if n_modules_actual != 1:
                            raise ValueError(f"Expected {n_modules} modules, but get "
                                             f"{n_modules_actual} instead!")
                    elif n_modules_actual > 2:
                        raise ValueError(f"Expected 1 or 2 modules, but get "
                                         f"{n_modules_actual} instead!")

                # check number of memory cells
                if ndim == 4 and not shape[0]:
                    raise ValueError("Number of memory cells is zero!")

            except ValueError as e:
                raise AssemblingError(e)

            data['assembled'] = {
                'data': self._assemble(modules_data),
            }
            # Assign the global train ID once the main detector was
            # successfully assembled.
            raw["META timestamp.tid"] = meta[src]["tid"]

    class AgipdImageAssembler(BaseAssembler):
        def _get_modules_bridge(self, data, src):
            """Override.

            Should work for both raw and calibrated data, according to DSSC.

            - calibrated, "image.data", (modules, x, y, memory cells)
            - raw, "image.data", (modules, x, y, memory cells)
            -> (memory cells, modules, y, x)
            """
            modules_data = data[src]
            if modules_data.shape[1] == config["MODULE_SHAPE"][1]:
                # Reshaping could have already been done upstream (e.g.
                # at the PipeToZeroMQ device), if not:
                #   (modules, fs, ss, pulses) -> (pulses, modules, ss, fs)
                #   (modules, x, y, memory cells) -> (memory cells, modules, y, x)
                return np.transpose(modules_data, (3, 0, 2, 1))
            # (memory cells, modules, y, x)
            return modules_data

        def _get_modules_file(self, data, src):
            """Override.

            In the file, the data is separated into arrays of different
            modules. The layout of data for each module is:
            - calibrated, (memory cells, x, y)
            - raw, (memory cells, 2, x, y)

            - calibrated, "image.data", (memory cells, modules, y, x)
            - raw, "image.data", (memory cell, 2, modules, y, x)
                [:, 0, ...] -> data
                [:, 1, ...] -> gain
            -> (memory cells, modules, y, x)
            """
            modules_data = stack_detector_data(
                data[src], src.split(' ')[1], real_array=False)

            dtype = modules_data.dtype
            if dtype == _IMAGE_DTYPE:
                return modules_data

            if dtype == _RAW_IMAGE_DTYPE:
                return modules_data[:, 0, ...]

            raise AssemblingError(f"Unknown detector data type: {dtype}!")

        def _load_geometry(self, filename, quad_positions):
            """Override."""
            if self._assembler_type == GeomAssembler.OWN or self._stack_only:
                from ...geometries import AGIPD_1MGeometryFast

                if self._stack_only:
                    self._geom = AGIPD_1MGeometryFast()
                else:
                    try:
                        self._geom = AGIPD_1MGeometryFast.from_crystfel_geom(
                            filename)
                    except (ImportError, ModuleNotFoundError, OSError) as e:
                        raise AssemblingError(e)
            else:
                from extra_geom import AGIPD_1MGeometry

                try:
                    self._geom = AGIPD_1MGeometry.from_crystfel_geom(filename)
                except (ImportError, ModuleNotFoundError, OSError) as e:
                    raise AssemblingError(e)

    class LpdImageAssembler(BaseAssembler):
        def _get_modules_bridge(self, data, src):
            """Override.

            Should work for both raw and calibrated data, according to DSSC.

            - calibrated, "image.data", (modules, x, y, memory cells)
            - raw, "image.data", (modules, x, y, memory cells)
            -> (memory cells, modules, y, x)
            """
            return np.moveaxis(np.moveaxis(data[src], 3, 0), 3, 2)

        def _get_modules_file(self, data, src):
            """Override.

            In the file, the data is separated into arrays of different
            modules. The layout of data for each module is:
            - calibrated, (memory cells, x, y)
            - raw, (memory cells, 1, x, y)

            - calibrated, "image.data", (memory cells, modules, y, x)
            - raw, "image.data", (memory cell, 1, modules, y, x)
            -> (memory cells, modules, y, x)
            """
            modules_data = stack_detector_data(
                data[src], src.split(' ')[1], real_array=True)

            dtype = modules_data.dtype
            if dtype == _IMAGE_DTYPE:
                return modules_data
            if dtype == _RAW_IMAGE_DTYPE:
                return modules_data.squeeze(axis=1)

            raise AssemblingError(f"Unknown detector data type: {dtype}!")

        def _load_geometry(self, filename, quad_positions):
            """Override."""
            if self._assembler_type == GeomAssembler.OWN \
                    or self._stack_only:
                from ...geometries import LPD_1MGeometryFast

                if self._stack_only:
                    self._geom = LPD_1MGeometryFast()
                else:
                    self._geom = LPD_1MGeometryFast.from_h5_file_and_quad_positions(
                        filename, quad_positions)
            else:
                from extra_geom import LPD_1MGeometry

                try:
                    self._geom = LPD_1MGeometry.from_h5_file_and_quad_positions(
                        filename, quad_positions)
                except OSError as e:
                    raise AssemblingError(e)

    class DsscImageAssembler(BaseAssembler):

        def _get_modules_bridge(self, data, src):
            """Override.

            In the file, the data is separated into arrays of different
            modules. The layout of data for each module is:
            - calibrated, (memory cells, x, y)
            - raw, (memory cells, 1, x, y)

            - calibrated, "image.data", (modules, x, y, memory cells)
            - raw, "image.data", (modules, x, y, memory cells)
            -> (memory cells, modules, y, x)
            """
            return np.moveaxis(np.moveaxis(data[src], 3, 0), 3, 2)

        def _get_modules_file(self, data, src):
            """Override.

            - calibrated, "image.data", (memory cells, modules, y, x)
            - raw, "image.data", (memory cell, 1, modules, y, x)
            -> (memory cells, modules, y, x)
            """
            modules_data = stack_detector_data(
                data[src], src.split(' ')[1], real_array=False)

            dtype = modules_data.dtype
            if dtype == _IMAGE_DTYPE:
                return modules_data
            if dtype == _RAW_IMAGE_DTYPE:
                return modules_data.squeeze(axis=1)

            raise AssemblingError(f"Unknown detector data type: {dtype}!")

        def _load_geometry(self, filename, quad_positions):
            """Override."""
            if self._assembler_type == GeomAssembler.OWN \
                    or self._stack_only:
                from ...geometries import DSSC_1MGeometryFast

                if self._stack_only:
                    self._geom = DSSC_1MGeometryFast()
                else:
                    self._geom = DSSC_1MGeometryFast.from_h5_file_and_quad_positions(
                        filename, quad_positions)
            else:
                from extra_geom import DSSC_1MGeometry

                try:
                    self._geom = DSSC_1MGeometry.from_h5_file_and_quad_positions(
                        filename, quad_positions)
                except OSError as e:
                    raise AssemblingError(e)

    class JungFrauImageAssembler(BaseAssembler):
        def _get_modules_bridge(self, data, src):
            """Override.

            Calibrated data only.

            - calibrated, "data.adc", (y, x, modules)
            - raw, "data.adc", TODO
            -> (y, x)
            """
            modules_data = data[src]
            if modules_data.shape[-1] == 1:
                return modules_data.squeeze(axis=-1)
            else:
                raise NotImplementedError("Number of modules > 1")

        def _get_modules_file(self, data, src):
            """Override.

            - calibrated, "data.adc", (modules, y, x)
            - raw, "data.adc", (modules, y, x)
            -> (y, x)
            """
            modules_data = data[src]
            if modules_data.shape[0] == 1:
                return modules_data.squeeze(axis=0)
            else:
                raise NotImplementedError("Number of modules > 1")

    class JungFrauPulseResolvedImageAssembler(BaseAssembler):
        def _get_modules_bridge(self, data, src):
            """Override.

            Calibrated data only.

            - calibrated, "data.adc", TODO
            - raw, "data.adc", TODO
            -> (memory cells, modules, y, x)
            """
            modules_data = data[src]
            shape = modules_data.shape
            ndim = len(shape)
            if ndim == 3:
                # (y, x, memory cells) -> (memory cells, 1 module, y, x)
                return np.moveaxis(modules_data, -1, 0)[:, np.newaxis, ...]
            # (modules, y, x, memory cells) -> (memory cells, modules, y, x)
            return np.moveaxis(modules_data, -1, 0)

        def _get_modules_file(self, data, src):
            """Override.

            -> (memory cells, modules, y, x)
            """
            # modules_data = data[src_name]["data.adc"]
            # shape = modules_data.shape
            # ndim = len(shape)
            # if ndim == 3:
            #     # (pusles, y, x) -> (pulses, 1 module, y, x)
            #     return modules_data[:, np.newaxis, :]
            # # (pulses, modules, y, x,) -> (pulses, modules, y, x)
            # return modules_data
            raise NotImplementedError

    class EPix100ImageAssembler(BaseAssembler):
        def _get_modules_bridge(self, data, src):
            """Override.

            - calibrated, "data.image", (y, x, 1)
            - raw, "data.image.data", (1, y, x)
            -> (y, x)
            """
            img_data = data[src]
            dtype = img_data.dtype

            if dtype == _IMAGE_DTYPE:
                return img_data.squeeze(axis=-1)

            # raw data of ePix100 has an unexpected dtype int16
            if dtype == np.int16:
                return img_data.squeeze(axis=0)

            raise AssemblingError(f"Unknown detector data type: {dtype}!")

        def _get_modules_file(self, data, src):
            """Override.

            - calibrated, "data.image.pixels", (y, x)
            - raw, "data.image.pixels", (y, x)
            -> (y, x)
            """
            return data[src]

    class FastCCDImageAssembler(BaseAssembler):
        def _get_modules_bridge(self, data, src):
            """Override.

            - calibrated, "data.image", (y, x, 1)
            - raw, "data.image.data", (y, x)
            -> (y, x)
            """
            img_data = data[src]
            dtype = img_data.dtype

            if dtype == _IMAGE_DTYPE:
                return img_data.squeeze(axis=-1)

            if dtype == _RAW_IMAGE_DTYPE:
                return img_data

            raise AssemblingError(f"Unknown detector data type: {dtype}!")

        def _get_modules_file(self, data, src):
            """Override.

            - calibrated, "data.image.pixels", (y, x)
            - raw, "data.image.pixels", (y, x)
            -> (y, x)
            """
            return data[src]

    class BaslerCameraImageAssembler(BaseAssembler):
        # TODO: remove BaslerCamera from detector
        #       make a category for BaslerCamera.
        def _get_modules_bridge(self, data, src):
            """Override.

            - raw, "data.image.data", (y, x)
            -> (y, x)
            """
            # (y, x)
            return data[src]

        def _get_modules_file(self, data, src):
            """Override.

            -> (y, x)
            """
            raise NotImplementedError

    @classmethod
    def create(cls, detector):
        if detector == 'AGIPD':
            return cls.AgipdImageAssembler()

        if detector == 'LPD':
            return cls.LpdImageAssembler()

        if detector == 'DSSC':
            return cls.DsscImageAssembler()

        if detector == 'JungFrau':
            return cls.JungFrauImageAssembler()

        if detector == 'FastCCD':
            return cls.FastCCDImageAssembler()

        if detector == 'ePix100':
            return cls.EPix100ImageAssembler()

        if detector == 'BaslerCamera':
            return cls.BaslerCameraImageAssembler()

        if detector == 'JungFrauPR':
            return cls.JungFrauPulseResolvedImageAssembler()

        raise NotImplementedError(f"Unknown detector type {detector}!")
