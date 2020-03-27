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


class ImageAssemblerFactory(ABC):

    class BaseAssembler(_BaseProcessor, _RedisParserMixin):
        """Abstract ImageAssembler class.

        Attributes:
            _require_geom (bool): whether a Geometry is required to assemble
                the detector modules.
            _stack_only (bool): whether simply stack all modules seamlessly
                together.
            _mask_tile (bool): whether to mask the tile of each module
                if applicable.
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
            self._mask_tile = False
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
                geom_file = cfg["geometry_file"]
                quad_positions = json.loads(cfg["quad_positions"],
                                            encoding='utf8')

                image_proc_cfg = self._meta.hget_all(mt.IMAGE_PROC)
                mask_tile = image_proc_cfg["mask_tile"] == 'True'
                if mask_tile != self._mask_tile:
                    self._mask_tile = mask_tile
                    if mask_tile:
                        # Reset the out array when mask_tile is switched from
                        # False to True. Otherwise, edge pixels from the
                        # previous train will remain there forever as the
                        # "mask_tile" here is actually called
                        # "ignore_tile_edge" in the corresponding function.
                        self._out_array = None

                # reload geometry if any of the following 4 fields changed
                if stack_only != self._stack_only or \
                        assembler_type != self._assembler_type or \
                        geom_file != self._geom_file or \
                        quad_positions != self._quad_position:

                    self._stack_only = stack_only
                    self._assembler_type = assembler_type
                    self._quad_position = quad_positions

                    self._geom = None  # reset first
                    self._load_geometry(geom_file, quad_positions)
                    # caveat: if _load_geometry raises, _geom_file will not
                    #         be set. Therefore, _load_geometry will raise
                    #         AssemblingError in the next train.
                    self._geom_file = geom_file

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
                    self._geom.position_all_modules(modules,
                                                    out=self._out_array,
                                                    ignore_tile_edge=self._mask_tile)
                # EXtra-foam raises ValueError while EXtra-geom raises
                # AssertionError if the shape of the output array does not
                # match the expected one, e.g. after a change of quadrant
                # positions during runtime.
                except (ValueError, AssertionError):
                    # recreate the output array
                    self._out_array = self._geom.output_array_for_position_fast(
                        extra_shape=(n_pulses, ), dtype=image_dtype)
                    self._geom.position_all_modules(modules,
                                                    out=self._out_array,
                                                    ignore_tile_edge=self._mask_tile)

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
            if self._assembler_type == GeomAssembler.OWN:
                from ...geometries import AGIPD_1MGeometryFast

                if self._stack_only:
                    self._geom = AGIPD_1MGeometryFast()
                else:
                    try:
                        # catch any exceptions here since it loads the CFEL
                        # geometry file with a CFEL function
                        self._geom = AGIPD_1MGeometryFast.from_crystfel_geom(
                            filename)
                    except Exception as e:
                        raise AssemblingError(e)
            else:
                from ...geometries import AGIPD_1MGeometry

                try:
                    # catch any exceptions here since it loads the CFEL
                    # geometry file with a CFEL function
                    self._geom = AGIPD_1MGeometry.from_crystfel_geom(filename)
                except Exception as e:
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
            if self._assembler_type == GeomAssembler.OWN:
                from ...geometries import LPD_1MGeometryFast

                if self._stack_only:
                    self._geom = LPD_1MGeometryFast()
                else:
                    try:
                        self._geom = LPD_1MGeometryFast.from_h5_file_and_quad_positions(
                            filename, quad_positions)
                    except (OSError, KeyError) as e:
                        raise AssemblingError(f"[Geometry] {e}")
            else:
                from ...geometries import LPD_1MGeometry

                try:
                    self._geom = LPD_1MGeometry.from_h5_file_and_quad_positions(
                        filename, quad_positions)
                except (OSError, KeyError) as e:
                    raise AssemblingError(f"[Geometry] {e}")

    class DsscImageAssembler(BaseAssembler):

        def _get_modules_bridge(self, data, src):
            """Override.

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
                data[src], src.split(' ')[1], real_array=False)

            dtype = modules_data.dtype
            if dtype == _IMAGE_DTYPE:
                return modules_data
            if dtype == _RAW_IMAGE_DTYPE:
                return modules_data.squeeze(axis=1)

            raise AssemblingError(f"Unknown detector data type: {dtype}!")

        def _load_geometry(self, filename, quad_positions):
            """Override."""
            if self._assembler_type == GeomAssembler.OWN:
                from ...geometries import DSSC_1MGeometryFast

                if self._stack_only:
                    self._geom = DSSC_1MGeometryFast()
                else:
                    try:
                        self._geom = DSSC_1MGeometryFast.from_h5_file_and_quad_positions(
                            filename, quad_positions)
                    except (OSError, KeyError) as e:
                        raise AssemblingError(f"[Geometry] {e}")
            else:
                from ...geometries import DSSC_1MGeometry

                try:
                    self._geom = DSSC_1MGeometry.from_h5_file_and_quad_positions(
                        filename, quad_positions)
                except (OSError, KeyError) as e:
                    raise AssemblingError(f"[Geometry] {e}")

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

            - calibrated, "data.adc", (y, x, memory cells)
            - raw, "data.adc", (memory cells, y, x)
            -> (memory cells, modules, y, x)
            """
            modules_data = data[src]
            shape = modules_data.shape
            dtype = modules_data.dtype

            ndim = len(shape)
            if ndim == 3:
                if dtype == _IMAGE_DTYPE:
                    # (y, x, memory cells) -> (memory cells, 1 module, y, x)
                    return np.moveaxis(modules_data, -1, 0)[:, np.newaxis, ...]

                if dtype == _RAW_IMAGE_DTYPE:
                    # (memory cells, y, x) -> (memory cells, 1 module, y, x)
                    return modules_data[:, np.newaxis, ...]

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
