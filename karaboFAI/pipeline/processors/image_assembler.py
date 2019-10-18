"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

ImageAssemblers for different detectors.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from abc import ABC, abstractmethod
import re

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
from ...utils import profiler


class ImageAssemblerFactory(ABC):
    class BaseAssembler(_BaseProcessor, _RedisParserMixin):
        """Abstract ImageAssembler class.

        Attributes:
            _source_name (str): detector data source name.
        """
        def __init__(self):
            """Initialization."""
            super().__init__()

            self._source_name = None

            self._geom_file = None
            self._quad_position = None
            self._geom = None
            self._out_array = None
            self._n_images = None

        def update(self):
            srcs = self._meta.get_all_data_sources(config["DETECTOR"])

            if srcs:
                self._source_name = srcs[-1].name

            if config['REQUIRE_GEOMETRY']:
                geom_cfg = self._meta.get_all(mt.GEOMETRY_PROC)
                geom_file = geom_cfg["geometry_file"]
                quad_positions = json.loads(geom_cfg["quad_positions"],
                                            encoding='utf8')
                if geom_file != self._geom_file or \
                        quad_positions != self._quad_position:
                    self.load_geometry(geom_file, quad_positions)
                    self._geom_file = geom_file
                    self._quad_position = quad_positions
                    print(f"Loaded geometry from {geom_file}")

        @abstractmethod
        def _get_modules_bridge(self, data, src_name):
            """Get modules data from bridge."""
            pass

        @abstractmethod
        def _get_modules_file(self, data, src_name):
            """Get modules data from file."""
            pass

        def _clear_det_data(self, data, src_name):
            """Delete detector related data from the raw data."""
            try:
                del data[src_name]
            except KeyError:
                # stream multi-module detector data from files
                devices = [dev for dev in data.keys()]
                for device in devices:
                    if re.search(r'/DET/(\d+)CH', device):
                        del data[device]

        def load_geometry(self, filepath, quad_positions):
            """Load geometry from file.

            :param str filepath: path of the geometry file.
            :param tuple quad_positions: quadrant coordinates.
            """
            pass

        def _modules_to_assembled(self, modules):
            """Convert modules data to assembled image data."""
            image_dtype = config["IMAGE_DTYPE"]
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
            # For train-resolved detector, assembled is a reference
            # to the array data received from the pyzmq. This array data
            # is only readable since the data is owned by a pointer in
            # the zmq message (it is not copied). However, other data
            # like data['metadata'] is writeable.
            return modules.astype(image_dtype)

        @profiler("Image Assembler")
        def process(self, data):
            """Assemble the image data.

            :returns assembled: assembled detector image data.
            """
            src_name = self._source_name
            src_type = data['source_type']
            raw = data['raw']

            try:
                if src_type == DataSource.FILE:
                    modules_data = self._get_modules_file(raw, src_name)
                elif src_type == DataSource.BRIDGE:
                    modules_data = self._get_modules_bridge(raw, src_name)
                else:
                    raise ValueError(f"Unknown source type: {src_type}")

                self._clear_det_data(raw, src_name)
            except (ValueError, IndexError, KeyError) as e:
                raise AssemblingError(e)

            shape = modules_data.shape
            ndim = len(shape)
            try:
                n_modules = config["NUMBER_OF_MODULES"]
                module_shape = config["MODULE_SHAPE"]
                # BaslerCamera has module shape [-1, -1]
                if module_shape[0] > 0 and list(shape[-2:]) != module_shape:
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

            assembled = self._modules_to_assembled(modules_data)
            data['assembled'] = assembled

    class AgipdImageAssembler(BaseAssembler):
        def _get_modules_bridge(self, data, src_name):
            """Overload."""
            # (memory cells, modules, y, x)
            modules_data = data[src_name]["image.data"]
            if modules_data.shape[1] == config["MODULE_SHAPE"][1]:
                # online-calibrated data, if reshaping not done upstream
                # (modules, fs, ss, pulses) -> (pulses, modules, ss, fs)
                return np.transpose(modules_data, (3, 0, 2, 1))
            return modules_data

        def _get_modules_file(self, data, src_name):
            """Overload."""
            # (memory cells, modules, y, x)
            return stack_detector_data(data, "image.data", only="AGIPD")

        def load_geometry(self, filename, quad_positions):
            """Overload."""
            try:
                self._geom = AGIPD_1MGeometry.from_crystfel_geom(filename)
            except (ImportError, ModuleNotFoundError, OSError) as e:
                raise AssemblingError(e)

    class LpdImageAssembler(BaseAssembler):
        def _get_modules_bridge(self, data, src_name):
            """Overload."""
            # (modules, x, y, memory cells) -> (memory cells, modules, y, x)
            return np.moveaxis(
                np.moveaxis(data[src_name]["image.data"], 3, 0), 3, 2)

        def _get_modules_file(self, data, src_name):
            """Overload."""
            # (memory cells, modules, y, x)
            return stack_detector_data(data, "image.data", only="LPD")

        def load_geometry(self, filename, quad_positions):
            """Overload."""
            try:
                self._geom = LPD_1MGeometry.from_h5_file_and_quad_positions(
                    filename, quad_positions)
            except OSError as e:
                raise AssemblingError(e)

    class JungFrauImageAssembler(BaseAssembler):
        def _get_modules_bridge(self, data, src_name):
            """Overload."""
            modules_data = data[src_name]["data.adc"]
            if modules_data.shape[-1] == 1:
                # (y, x, modules) -> (y, x)
                return modules_data.squeeze(axis=-1)
            else:
                raise NotImplementedError("Number of modules > 1")

        def _get_modules_file(self, data, src_name):
            """Overload."""
            modules_data = data[src_name]['data.adc']
            if modules_data.shape[0] == 1:
                # (modules, y, x) -> (y, x)
                return modules_data.squeeze(axis=0)
            else:
                raise NotImplementedError("Number of modules > 1")

    class JungFrauPulseResolvedImageAssembler(BaseAssembler):
        def _get_modules_bridge(self, data, src_name):
            """Overload."""
            modules_data = data[src_name]["data.adc"]
            shape = modules_data.shape
            ndim = len(shape)
            if ndim == 3:
                # (y, x, memory cells) -> (memory cells, 1 module, y, x)
                return np.moveaxis(modules_data, -1, 0)[:, np.newaxis, ...]
            # (modules, y, x, memory cells) -> (memory cells, modules, y, x)
            return np.moveaxis(modules_data, -1, 0)

        def _get_modules_file(self, data, src_name):
            """Overload."""
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
        def _get_modules_bridge(self, data, src_name):
            """Overload."""
            # (y, x, 1) -> (y, x)
            return data[src_name]["data.image"].squeeze(axis=-1)

        def _get_modules_file(self, data, src_name):
            """Overload."""
            # (y, x)
            return data[src_name]['data.image.pixels']

    class BaslerCameraImageAssembler(BaseAssembler):
        def _get_modules_bridge(self, data, src_name):
            """Overload."""
            # (y, x)
            return data[src_name]["data.image.data"]

        def _get_modules_file(self, data, src_name):
            """Overload."""
            raise NotImplementedError

    class DsscImageAssembler(BaseAssembler):
        @profiler("Prepare Module Data")
        def _get_modules_bridge(self, data, src_name):
            """Overload."""
            # (memory cells, modules, y, x)
            return np.moveaxis(
                np.moveaxis(data[src_name]["image.data"], 3, 0), 3, 2)

        @profiler("Prepare Module Data")
        def _get_modules_file(self, data, src_name):
            """Overload."""
            modules_data = stack_detector_data(data, "image.data", only="DSSC")

            dtype = modules_data.dtype

            # calibrated data
            if dtype == np.float32:
                # (memory cells, modules, y, x)
                return modules_data

            # raw data
            if dtype == np.uint16:
                # (memory cell, 1, modules, y, x) -> (memory cells, modules, y, x)
                return np.squeeze(modules_data, axis=1)

            raise AssemblingError(f"Unknown detector data type: {dtype}!")

        def load_geometry(self, filename, quad_positions):
            """Overload."""
            try:
                self._geom = DSSC_1MGeometry.from_h5_file_and_quad_positions(
                        filename, quad_positions)
            # FIXME: OSError?
            except Exception as e:
                raise AssemblingError(e)

    @classmethod
    def create(cls, detector):
        if detector == 'AGIPD':
            return cls.AgipdImageAssembler()

        if detector == 'LPD':
            return cls.LpdImageAssembler()

        if detector == 'JungFrau':
            return cls.JungFrauImageAssembler()

        if detector == 'FastCCD':
            return cls.FastCCDImageAssembler()

        if detector == 'BaslerCamera':
            return cls.BaslerCameraImageAssembler()

        if detector == 'DSSC':
            return cls.DsscImageAssembler()

        if detector == 'JungFrauPR':
            return cls.JungFrauPulseResolvedImageAssembler()

        raise NotImplementedError(f"Unknown detector type {detector}!")
