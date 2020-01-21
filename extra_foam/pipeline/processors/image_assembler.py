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

from karabo_data import stack_detector_data
from karabo_data.geometry2 import (
    AGIPD_1MGeometry, DSSC_1MGeometry, LPD_1MGeometry
)
from .base_processor import _BaseProcessor, _RedisParserMixin
from ..exceptions import AssemblingError
from ...config import config, DataSource
from ...database import Metadata as mt
from ...ipc import process_logger as logger
from ...utils import profiler


_IMAGE_DTYPE = config['SOURCE_PROC_IMAGE_DTYPE']
_RAW_IMAGE_DTYPE = config['SOURCE_RAW_IMAGE_DTYPE']


class ImageAssemblerFactory(ABC):
    class BaseAssembler(_BaseProcessor, _RedisParserMixin):
        """Abstract ImageAssembler class.

        Attributes:
        """
        def __init__(self):
            """Initialization."""
            super().__init__()

            self._geom_file = None
            self._quad_position = None
            self._geom = None
            self._out_array = None
            self._n_images = None

        def update(self):
            if config['REQUIRE_GEOMETRY']:
                geom_cfg = self._meta.hget_all(mt.GEOMETRY_PROC)

                with_geometry = geom_cfg["with_geometry"] == 'True'
                if with_geometry:
                    geom_file = geom_cfg["geometry_file"]
                    quad_positions = json.loads(geom_cfg["quad_positions"],
                                                encoding='utf8')
                    if geom_file != self._geom_file or \
                            quad_positions != self._quad_position:
                        self.load_geometry(geom_file, quad_positions)
                        self._geom_file = geom_file
                        self._quad_position = quad_positions
                        logger.info(f"Loaded geometry from {geom_file} with "
                                    f"quadrant positions {quad_positions}")
                else:
                    # when only a single module is required or we only want to
                    # assemble modules seamlessly together.
                    self._geom = None

        @abstractmethod
        def _get_modules_bridge(self, data, src):
            """Get modules data from bridge."""
            pass

        @abstractmethod
        def _get_modules_file(self, data, src):
            """Get modules data from file."""
            pass

        def load_geometry(self, filepath, quad_positions):
            """Load geometry from file.

            :param str filepath: path of the geometry file.
            :param tuple quad_positions: quadrant coordinates.
            """
            pass

        def _modules_to_assembled(self, modules):
            """Convert modules data to assembled image data."""
            image_dtype = config["SOURCE_PROC_IMAGE_DTYPE"]
            if self._geom is not None:
                # karabo_data interface
                n_images = (modules.shape[0], )
                if self._out_array is None or self._n_images != n_images:
                    self._n_images = n_images
                    self._out_array = self._geom.output_array_for_position_fast(
                        extra_shape=self._n_images, dtype=image_dtype)
                try:
                    assembled, centre = self._geom.position_all_modules(
                        modules, out=self._out_array)
                except Exception:  # raise AssertionError if shape changes
                    # recreate the output array
                    self._out_array = self._geom.output_array_for_position_fast(
                        extra_shape=self._n_images, dtype=image_dtype)
                    assembled, centre = self._geom.position_all_modules(
                        modules, out=self._out_array)

                return assembled

            # for Pulse resolved JungFrau without geometry
            shape = modules.shape
            if config["DETECTOR"] == "JungFrauPR":
                # Stacking modules vertically along y axis.
                return modules.reshape(shape[0], -1, shape[-1])
            elif modules.ndim == 4:
                raise AssemblingError(
                    "Assembling modules without geometry is not supported!")

            # For train-resolved detector, assembled is a reference
            # to the array data received from the pyzmq. This array data
            # is only readable since the data is owned by a pointer in
            # the zmq message (it is not copied). However, other data
            # like data['metadata'] is writeable.
            return modules.astype(image_dtype, copy=True)

        @profiler("Image Assembler")
        def process(self, data):
            """Assemble the image data.

            :returns assembled: assembled detector image data.
            """
            meta = data['meta']
            raw = data['raw']

            src = data["catalog"].main_detector
            if src not in meta:
                raise AssemblingError(
                    f"{config['DETECTOR']} source <{src}> not found!")
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

                # BaslerCamera has module shape (-1, -1)
                if module_shape[0] > 0 and shape[-2:] != module_shape:
                    raise ValueError(f"Expected module shape {module_shape}, "
                                     f"but get {shape[-2:]} instead!")
                elif ndim >= 3 and shape[-3] != n_modules:
                    if config["DETECTOR"] != "JungFrauPR":
                        raise ValueError(f"Expected {n_modules} modules, but get "
                                         f"{shape[0]} instead!")
                    elif shape[-3] > 2:
                        raise ValueError(f"Expected 1 or 2 modules, but get "
                                         f"{shape[0]} instead!")
                elif ndim == 4 and not shape[0]:
                    raise ValueError("Number of memory cells is zero!")

            except ValueError as e:
                raise AssemblingError(e)

            data['assembled'] = {
                'data': self._modules_to_assembled(modules_data),
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

            - calibrated, "image.data", (memory cells, modules, y, x)
            - raw, "image.data", (memory cell, 2, modules, y, x)
                [:, 0, ...] -> data
                [:, 1, ...] -> gain
            """
            modules_data = stack_detector_data(data[src], src.split(' ')[1])
            dtype = modules_data.dtype

            if dtype == _IMAGE_DTYPE:
                return modules_data

            if dtype == _RAW_IMAGE_DTYPE:
                return modules_data[:, 0, ...]

            raise AssemblingError(f"Unknown detector data type: {dtype}!")

        def load_geometry(self, filename, quad_positions):
            """Override."""
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
            """
            # (modules, x, y, memory cells) -> (memory cells, modules, y, x)
            return np.moveaxis(np.moveaxis(data[src], 3, 0), 3, 2)

        def _get_modules_file(self, data, src):
            """Override.

            - calibrated, "image.data", (memory cells, modules, y, x)
            - raw, "image.data", (memory cell, 1, modules, y, x)
            """
            modules_data = stack_detector_data(data[src], src.split(' ')[1])
            dtype = modules_data.dtype

            if dtype == _IMAGE_DTYPE:
                return modules_data

            if dtype == _RAW_IMAGE_DTYPE:
                return np.squeeze(modules_data, axis=1)

            raise AssemblingError(f"Unknown detector data type: {dtype}!")

        def load_geometry(self, filename, quad_positions):
            """Override."""
            try:
                self._geom = LPD_1MGeometry.from_h5_file_and_quad_positions(
                    filename, quad_positions)
            except OSError as e:
                raise AssemblingError(e)

    class DsscImageAssembler(BaseAssembler):
        @profiler("Prepare Module Data")
        def _get_modules_bridge(self, data, src):
            """Override.

            - calibrated, "image.data", (modules, x, y, memory cells)
            - raw, "image.data", (modules, x, y, memory cells)
            """
            # (modules, x, y, memory cells) -> (memory cells, modules, y, x)
            return np.moveaxis(np.moveaxis(data[src], 3, 0), 3, 2)

        @profiler("Prepare Module Data")
        def _get_modules_file(self, data, src):
            """Override.

            - calibrated, "image.data", (memory cells, modules, y, x)
            - raw, "image.data", (memory cell, 1, modules, y, x)
            """
            modules_data = stack_detector_data(data[src], src.split(' ')[1])
            dtype = modules_data.dtype

            if dtype == _IMAGE_DTYPE:
                return modules_data

            if dtype == _RAW_IMAGE_DTYPE:
                return np.squeeze(modules_data, axis=1)

            raise AssemblingError(f"Unknown detector data type: {dtype}!")

        def load_geometry(self, filename, quad_positions):
            """Override."""
            try:
                self._geom = DSSC_1MGeometry.from_h5_file_and_quad_positions(
                    filename, quad_positions)
            # FIXME: OSError?
            except Exception as e:
                raise AssemblingError(e)

    class JungFrauImageAssembler(BaseAssembler):
        def _get_modules_bridge(self, data, src):
            """Override.

            Calibrated data only.

            - calibrated, "data.adc", (y, x, modules)
            - raw, "data.adc", TODO
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
            """Override."""
            # modules_data = data[src_name]["data.adc"]
            # shape = modules_data.shape
            # ndim = len(shape)
            # if ndim == 3:
            #     # (pusles, y, x) -> (pulses, 1 module, y, x)
            #     return modules_data[:, np.newaxis, :]
            # # (pulses, modules, y, x,) -> (pulses, modules, y, x)
            # return modules_data
            raise NotImplementedError

    class FastCCDImageAssembler(BaseAssembler):
        def _get_modules_bridge(self, data, src):
            """Override.

            - calibrated, "data.image", (y, x, 1)
            - raw, "data.image.data", (y, x)
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
            """
            return data[src]

    class BaslerCameraImageAssembler(BaseAssembler):
        # TODO: remove BaslerCamera from detector
        #       make a category for BaslerCamera.
        def _get_modules_bridge(self, data, src):
            """Override.

            - raw, "data.image.data", (y, x)
            """
            # (y, x)
            return data[src]

        def _get_modules_file(self, data, src):
            """Override."""
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

        if detector == 'BaslerCamera':
            return cls.BaslerCameraImageAssembler()

        if detector == 'JungFrauPR':
            return cls.JungFrauPulseResolvedImageAssembler()

        raise NotImplementedError(f"Unknown detector type {detector}!")
