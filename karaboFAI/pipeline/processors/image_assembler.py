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

import json
import numpy as np
from h5py import File

from karabo_data import stack_detector_data
from karabo_data.geometry import LPDGeometry
from karabo_data.geometry2 import AGIPD_1MGeometry, DSSC_1MGeometry

from .base_processor import CompositeProcessor, _RedisParserMixin
from ..exceptions import AssemblingError
from ...config import config, DataSource
from ...metadata import Metadata as mt
from ...utils import profiler


class ImageAssemblerFactory(ABC):
    class BaseAssembler(CompositeProcessor, _RedisParserMixin):
        """Abstract ImageAssembler class.

        Attributes:
            _detector_source_name (str): detector data source name.
            _source_type (DataSource): detector data source type.
        """
        def __init__(self):
            """Initialization."""
            super().__init__()

            self._detector_source_name = None
            self._source_type = None

            self._geom_file = None
            self._quad_position = None
            self._geom = None

        def update(self):
            ds_cfg = self._meta.get_all(mt.DATA_SOURCE)
            self._detector_source_name = ds_cfg["detector_source_name"]
            self._source_type = DataSource(int(ds_cfg["source_type"]))

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

        def load_geometry(self, filepath, quad_positions):
            """Load geometry from file.

            :param str filepath: path of the geometry file.
            :param tuple quad_positions: quadrant coordinates.
            """
            pass

        def _modules_to_assembled(self, modules):
            """Convert modules data to assembled image data."""
            if self._geom is not None:
                # karabo_data interface
                assembled, centre = self._geom.position_all_modules(modules)
                return assembled

            # For train-resolved detector, assembled is a reference
            # to the array data received from the pyzmq. This array data
            # is only readable since the data is owned by a pointer in
            # the zmq message (it is not copied). However, other data
            # like data['metadata'] is writeable.
            return np.copy(modules)

        @profiler("Image Assembler")
        def process(self, data):
            """Assemble the image data.

            :returns assembled: assembled detector image data.
            """
            src_name = self._detector_source_name
            src_type = self._source_type

            raw = data['raw']

            try:
                if src_type == DataSource.FILE:
                    modules_data = self._get_modules_file(raw, src_name)
                elif src_type == DataSource.BRIDGE:
                    modules_data = self._get_modules_bridge(raw, src_name)
                else:
                    raise ValueError(f"Unknown source type: {src_type}")
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
                    raise ValueError(f"Expected {n_modules} modules, but get "
                                     f"{shape[0]} instead!")
                elif ndim == 4 and not shape[0]:
                    raise ValueError("Number of memory cells is zero!")

            except ValueError as e:
                raise AssemblingError(e)

            # assembled is a temporary item which will be need in ImageProcessor
            data['assembled'] = self._modules_to_assembled(modules_data)

    class AgipdImageAssembler(BaseAssembler):
        def _get_modules_bridge(self, data, src_name):
            """Overload."""
            # (memory cells, modules, y, x)
            return data[src_name]["image.data"]

        def _get_modules_file(self, data, src_name):
            """Overload."""
            # (memory cells, modules, y, x)
            return stack_detector_data(data, "image.data")

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
            return stack_detector_data(data, "image.data")

        def load_geometry(self, filename, quad_positions):
            """Overload."""
            try:
                with File(filename, 'r') as f:
                    self._geom = LPDGeometry.from_h5_file_and_quad_positions(
                        f, quad_positions)
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
            # (memory cells, modules, y, x)
            modules_data = stack_detector_data(data, "image.data")
            # ndim = len(modules_data.shape)
            # Raw files contain (memory cell, 1 , modules, y, x)
            # if ndim == 5:
            #     modules_data = np.squeeze(modules_data, axis=1)
            return modules_data

        def load_geometry(self, filename, quad_positions):
            """Overload."""
            try:
                self._geom = DSSC_1MGeometry.from_h5_file_and_quad_positions(
                        filename, quad_positions)
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

        raise NotImplementedError(f"Unknown detector type {detector}!")
