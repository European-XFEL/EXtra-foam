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
from extra_data.stacking import StackView

from .base_processor import _RedisParserMixin
from ..exceptions import AssemblingError
from ...config import config, GeomAssembler, DataSource
from ...database import SourceCatalog
from ...geometries import load_geometry, maybe_mask_asic_edges
from ...ipc import process_logger as logger


_IMAGE_DTYPE = config['SOURCE_PROC_IMAGE_DTYPE']
_RAW_IMAGE_DTYPE = config['SOURCE_RAW_IMAGE_DTYPE']
_TRAIN_ID = SourceCatalog.TRAIN_ID


def _maybe_squeeze_to_image(arr):
    """Try to squeeze an array to a 2D image."""
    if arr.ndim == 2:
        return arr

    try:
        return np.squeeze(arr, axis=0)
    except ValueError:
        try:
            return np.squeeze(arr, axis=-1)
        except ValueError:
            raise ValueError(f"Array with shape {arr.shape} cannot be "
                             f"squeezed to a 2D image")


def _stack_detector_modules(data, src, modules, *,
                            pulse_resolved=True, memory_cell_last=False):
    """Stack detector modules.

    It should be used for detectors like, JungFrau, ePix100, etc. For
    AGIPD, LPD and DSSC, stack_detector_data should be used.

    :param dict data: raw data.
    :param str src: source name.
    :param list modules: a list of module indices. The module indices do
        not need to be continuous or be monotonically increasing.
    :param bool pulse_resolved: whether the detector is pulse-resolved.
    :param bool memory_cell_last: whether memory cell is the last dimension.
    """
    if isinstance(data, np.ndarray):
        # This happens when the source name if one of the modules. if the
        # source name neither contains "*" nor is one of the modules,
        # KeyError would have been raised.
        raise ValueError("Not found multi-module data when stacking "
                         "detector modules")

    if not data:
        raise ValueError("No data")

    base_name, ppt = src.split(" ")
    prefix, suffix = base_name.split("*")

    dtypes, shapes = set(), set()
    modno_arrays = {}
    for i, modno in enumerate(modules):
        try:
            array = data[f"{prefix}{modno}{suffix}"][ppt]
        except KeyError:
            continue

        if not pulse_resolved:
            # single-module train-resolved detector data may have the
            # shape (1, y, x), (y, x, 1)
            array = _maybe_squeeze_to_image(array)
        elif memory_cell_last:
            # (y, x, memory cells) -> (memory cells, y x)
            array = np.moveaxis(array, -1, 0)

        dtypes.add(array.dtype)
        shapes.add(array.shape)
        modno_arrays[i] = array

    if len(dtypes) > 1:
        raise ValueError(f"Modules have mismatched dtypes: {dtypes}")

    if len(shapes) > 1:
        raise ValueError(f"Modules have mismatched shapes: {shapes}")

    dtype = dtypes.pop()
    shape = shapes.pop()
    stack = StackView(
        modno_arrays, len(modules), shape, dtype,
        fillvalue=np.nan, stack_axis=-3
    )

    return stack


class ImageAssemblerFactory(ABC):

    class BaseAssembler(_RedisParserMixin):
        """Abstract ImageAssembler class.

        Attributes:
            _stack_only (bool): whether simply stack all modules seamlessly
                together.
            _mask_tile (bool): whether to mask the edges of tiles of each
                module if applicable.
            _mask_asic (bool): whether to mask the edges of asics of each
                module if applicable.
            _assembler_type (GeomAssembler): Type of geometry assembler,
                which can be EXtra-foam or EXtra-geom.
            _geom_file (str): full path of the geometry file.
            _coordinates (list): (x, y) coordinates for the corners of 4
                quadrants for detectors like AGIPD, LPD and DSSC; (x, y)
                coordinates for the corners of all modules for detectors like
                JungFrau.
            _geom: geometry instance in use.
            _out_array (numpy.ndarray): buffer to store the assembled modules.
        """
        def __init__(self):
            """Initialization."""
            super().__init__()

            self._detector = config["DETECTOR"]
            self._n_modules = config["NUMBER_OF_MODULES"]
            self._module_shape = config["MODULE_SHAPE"]
            self._require_geom = config['REQUIRE_GEOMETRY']

            self._stack_only = False
            self._mask_tile = False
            self._mask_asic = False
            self._assembler_type = None
            self._geom_file = None
            self._coordinates = None
            self._geom = None
            self._out_array = None

        @property
        def geometry(self):
            return self._geom

        def update(self, cfg, *, mask_tile=False, mask_asic=False):
            if mask_tile != self._mask_tile:
                self._mask_tile = mask_tile
                if mask_tile:
                    # Reset the out array when mask_tile is switched from
                    # False to True. Otherwise, edge pixels from the
                    # previous train will remain there forever as the
                    # "mask_tile" here is actually called
                    # "ignore_tile_edge" in the corresponding function.
                    self._out_array = None

            if mask_asic != self._mask_asic:
                self._mask_asic = mask_asic
                if mask_asic:
                    # similar as the reason in "mask_tile"
                    self._out_array = None

            if not self._require_geom:
                return

            assembler_type = GeomAssembler(int(cfg["assembler"]))
            stack_only = cfg["stack_only"] == 'True'
            geom_file = cfg["geometry_file"]
            coordinates = json.loads(cfg["coordinates"])

            # reload geometry if any of the following 4 fields changed
            if stack_only != self._stack_only or \
                    geom_file != self._geom_file or \
                    coordinates != self._coordinates or \
                    assembler_type != self._assembler_type:
                self._load_geometry(
                    stack_only, geom_file, coordinates, assembler_type)

                if not stack_only:
                    logger.info(f"Loaded geometry from {geom_file} with "
                                f"quadrant/module positions {coordinates}")
                else:
                    logger.info(f"Loaded stack-only geometry with "
                                f"{self._n_modules} modules")

        @abstractmethod
        def _get_modules_bridge(self, data, src, modules):
            """Get modules data from bridge."""
            pass

        @abstractmethod
        def _get_modules_file(self, data, src, modules):
            """Get modules data from file."""
            pass

        def _load_geometry(self, stack_only, filepath, coordinates, assembler_type):
            """Load geometry from file.

            Required for modular detectors which must be assembled with
            a geometry.
            """
            try:
                self._geom = load_geometry(config["DETECTOR"],
                                           stack_only=stack_only,
                                           filepath=filepath,
                                           coordinates=coordinates,
                                           n_modules=self._n_modules,
                                           assembler=assembler_type)

                self._stack_only = stack_only
                self._geom_file = filepath
                self._coordinates = coordinates
                self._assembler_type = assembler_type

            except Exception as e:
                raise AssemblingError(f"[Geometry] {e}")

        def _assemble(self, modules):
            """Assemble modules data into assembled image data.

            :param array-like modules: modules data. shape = (memory cells,
                modules, y, x) for pulse-resolved detectors or (y, x) for
                train-resolved detectors.

            :return numpy.ndarray assembled: assembled detector image(s).
                shape = (memory cells, y, x) for pulse-resolved detectors
                and (y, x) for train resolved detectors.
            """
            if modules.ndim == 4:
                # single module operation (for all 1M detectors and JungFrau)
                if modules.shape[1] == 1:
                    sm = modules.astype(_IMAGE_DTYPE).squeeze(axis=1)
                    if self._mask_asic:
                        maybe_mask_asic_edges(sm, self._detector)
                    return sm

                n_pulses = modules.shape[0]
                if self._out_array is None or self._out_array.shape[0] != n_pulses:
                    self._out_array = self._geom.output_array_for_position_fast(
                        extra_shape=(n_pulses, ), dtype=_IMAGE_DTYPE)
            else:  # modules.ndim == 3
                if self._out_array is None:
                    self._out_array = self._geom.output_array_for_position_fast(
                        dtype=_IMAGE_DTYPE)

            try:
                self._geom.position_all_modules(modules,
                                                out=self._out_array,
                                                ignore_tile_edge=self._mask_tile,
                                                ignore_asic_edge=self._mask_asic)
            # EXtra-foam raises ValueError while EXtra-geom raises
            # AssertionError if the shape of the output array does not
            # match the expected one, e.g. after a change of quadrant
            # positions during runtime.
            except (ValueError, AssertionError):
                # recreate the output array
                if modules.ndim == 4:
                    n_pulses = modules.shape[0]
                    self._out_array = self._geom.output_array_for_position_fast(
                            extra_shape=(n_pulses, ), dtype=_IMAGE_DTYPE)
                else:  # modules.ndim == 3
                    self._out_array = self._geom.output_array_for_position_fast(
                        dtype=_IMAGE_DTYPE)

                self._geom.position_all_modules(modules,
                                                out=self._out_array,
                                                ignore_tile_edge=self._mask_tile,
                                                ignore_asic_edge=self._mask_asic)

            return self._out_array

        def _preprocess(self, image):
            """Preprocess single image data.

            :param array-like image: image data. shape = (y, x).

            :return numpy.ndarray: processed image data
            """
            # For train-resolved detector, assembled is a reference
            # to the array data received from the pyzmq. This array data
            # is only readable since the data is owned by a pointer in
            # the zmq message (it is not copied). However, other data
            # like data['metadata'] is writeable.
            image = image.astype(_IMAGE_DTYPE)
            if self._mask_asic:
                maybe_mask_asic_edges(image, self._detector)
            return image

        def process(self, data):
            """Override."""
            meta = data['meta']
            raw = data['raw']
            catalog = data["catalog"]

            src = catalog.main_detector
            modules = catalog.get_modules(src)
            src_type = meta[src]['source_type']
            try:
                if src_type == DataSource.FILE:
                    modules_data = self._get_modules_file(raw, src, modules)
                else:
                    modules_data = self._get_modules_bridge(raw, src, modules)

                # Remove raw detector data since we do not want to serialize
                # it and send around.
                raw[src] = None

            except (ValueError, IndexError, KeyError) as e:
                raise AssemblingError(str(e))

            shape = modules_data.shape
            ndim = len(shape)
            n_modules = self._n_modules
            module_shape = self._module_shape

            # check module shape (BaslerCamera has module shape (-1, -1))
            if module_shape[0] > 0 and shape[-2:] != module_shape:
                raise AssemblingError(f"Expected module shape {module_shape}, "
                                      f"but get {shape[-2:]} instead!")

            # check number of modules
            if ndim >= 3 and shape[-3] != n_modules:
                n_modules_actual = shape[-3]
                # allow single module operation
                if n_modules_actual != 1:
                    raise AssemblingError(f"Expected {n_modules} modules, but "
                                          f"get {n_modules_actual} instead!")

            # check number of memory cells
            if not shape[0]:
                # only happens when ndim == 4
                raise AssemblingError(f"Number of memory cells is zero!")

            if ndim == 2:
                data['assembled'] = {'data': self._preprocess(modules_data)}
            else:
                data['assembled'] = {'data': self._assemble(modules_data)}

            # Assign the global train ID once the main detector was
            # successfully assembled.
            raw[_TRAIN_ID] = meta[src]["train_id"]

    class AgipdImageAssembler(BaseAssembler):
        def _get_modules_bridge(self, data, src, modules):
            """Override.

            Should work for both raw and calibrated data, according to DSSC.

            - calibrated, "image.data", (modules, x, y, memory cells)
            - raw, "image.data", (modules, x, y, memory cells)
            -> (memory cells, modules, y, x)
            """
            modules_data = data[src]
            if modules_data.shape[1] == self._module_shape[1]:
                # Reshaping could have already been done upstream (e.g.
                # at the PipeToZeroMQ device), if not:
                #   (modules, fs, ss, pulses) -> (pulses, modules, ss, fs)
                #   (modules, x, y, memory cells) -> (memory cells, modules, y, x)
                return np.transpose(modules_data, (3, 0, 2, 1))
            # (memory cells, modules, y, x)
            return modules_data

        def _get_modules_file(self, data, src, modules):
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

    class LpdImageAssembler(BaseAssembler):
        def _get_modules_bridge(self, data, src, modules):
            """Override.

            Should work for both raw and calibrated data, according to DSSC.

            - calibrated, "image.data", (modules, x, y, memory cells)
            - raw, "image.data", (modules, x, y, memory cells)
            -> (memory cells, modules, y, x)
            """
            return np.moveaxis(np.moveaxis(data[src], 3, 0), 3, 2)

        def _get_modules_file(self, data, src, modules):
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

    class DsscImageAssembler(BaseAssembler):

        def _get_modules_bridge(self, data, src, modules):
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

        def _get_modules_file(self, data, src, modules):
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

    class JungFrauImageAssembler(BaseAssembler):
        def _get_modules_bridge(self, data, src, modules):
            """Override.

            Calibrated data only.

            Single module:
            - calibrated, "data.adc", (y, x, memory cells)
            - raw, "data.adc", TODO

            Stacked module:
            - calibrated, "data.adc", (modules, y, x, memory cells)
            - raw, "data.adc", TODO

            -> (memory cells, modules, y, x)
            """
            # TODO: deal with modules received separately
            modules_data = data[src]

            if self._n_modules == 1:
                #   (y, x, memory cells) -> (memory cells, 1, y, x)
                return np.expand_dims(np.moveaxis(modules_data, -1, 0), axis=1)

            if isinstance(modules_data, np.ndarray):
                if modules_data.shape[1] == self._module_shape[0]:
                    # Reshaping could have already been done upstream (e.g.
                    # at the PipeToZeroMQ device), if not:
                    #   (modules, y, x, memory cells) ->
                    #   (memory cells, modules, y, x)
                    modules_data = np.moveaxis(modules_data, -1, 0)
                # (memory cells, modules, y, x)
                return modules_data

            # modules data arrives without being stacked
            return _stack_detector_modules(modules_data, src, modules,
                                           memory_cell_last=True)

        def _get_modules_file(self, data, src, modules):
            """Override.

            Single module:
            - calibrated, "data.adc", (memory cells, y, x)
            Note: no extra axis like AGIPD, LPD, etc.
            - raw, "data.adc", (memory cells, y, x)

            -> (memory cells, modules, y, x)
            """
            if self._n_modules == 1:
                # (memory cells, y, x) -> (memory cells, modules, y, x)
                return np.expand_dims(data[src], axis=1)
            return _stack_detector_modules(data[src], src, modules)

    class EPix100ImageAssembler(BaseAssembler):
        def _get_modules_bridge(self, data, src, modules):
            """Override.

            - calibrated, "data.image", (y, x, 1)
            - raw, "data.image.data", (1, y, x)
            -> (y, x)
            """
            modules_data = data[src]

            if self._n_modules == 1:
                return _maybe_squeeze_to_image(modules_data)

            if isinstance(modules_data, np.ndarray):
                # assume the stacked array has the shape: (modules, y, x)
                return modules_data
            return _stack_detector_modules(modules_data, src, modules,
                                           pulse_resolved=False)

        def _get_modules_file(self, data, src, modules):
            """Override.

            - calibrated, "data.image.pixels", (y, x)
            - raw, "data.image.pixels", (y, x)
            -> (y, x)
            """
            modules_data = data[src]
            if self._n_modules == 1:
                return modules_data
            return _stack_detector_modules(modules_data, src, modules,
                                           pulse_resolved=False)

    class FastCCDImageAssembler(BaseAssembler):
        def _get_modules_bridge(self, data, src, modules):
            """Override.

            - calibrated, "data.image", (y, x, 1)
            - raw, "data.image.data", (y, x)
            -> (y, x)
            """
            return _maybe_squeeze_to_image(data[src])

        def _get_modules_file(self, data, src, modules):
            """Override.

            - calibrated, "data.image.pixels", (y, x)
            - raw, "data.image.pixels", (y, x)
            -> (y, x)
            """
            return data[src]

    class BaslerCameraImageAssembler(BaseAssembler):
        def _get_modules_bridge(self, data, src, modules):
            """Override.

            - raw, "data.image.data", (y, x)
            -> (y, x)
            """
            # (y, x)
            return data[src]

        def _get_modules_file(self, data, src, modules):
            """Override.

            - raw, "data.image.pixels", (y, x)
            -> (y, x)
            """
            return data[src]

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

        raise NotImplementedError(f"Unknown detector type {detector}!")
