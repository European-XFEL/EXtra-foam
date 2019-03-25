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

import numpy as np
from h5py import File

from karabo_data import stack_detector_data
from karabo_data.geometry import LPDGeometry
from karabo_data.geometry2 import AGIPD_1MGeometry

from ..config import DataSource


class ImageAssemblerFactory(ABC):
    class BaseAssembler:
        """Abstract ImageAssembler class.

        Attributes:
            source_name (str): detector data source name.
            source_type (DataSource): detector data source type.
            pulse_id_range (tuple): (min, max) pulse ID to be processed.
                (int, int)
        """
        _modules = 1
        _module_shape = None

        def __init__(self):
            """Initialization."""
            self.source_name = None
            self.source_type = None

            self._geom = None

            self.pulse_id_range = (None, None)

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
            return True, ""

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

        def assemble(self, data):
            """Assembled the image data.

            :returns assembled: assembled detector image data.
            """
            src_name = self.source_name
            src_type = self.source_type

            try:
                if src_type == DataSource.FILES:
                    modules_data = self._get_modules_file(data, src_name)
                elif src_type == DataSource.BRIDGE:
                    modules_data = self._get_modules_bridge(data, src_name)
                else:
                    raise ValueError(f"Unknown source type: {src_type}")
            except (ValueError, IndexError, KeyError):
                raise

            shape = modules_data.shape
            ndim = len(shape)
            if shape[-2:] != self._module_shape:
                raise ValueError(f"Expected module shape {self._module_shape}, "
                                 f"but get {shape[-2:]} instead!")
            elif ndim >= 3 and shape[-3] != self._modules:
                raise ValueError(f"Expected {self._modules} modules, but get"
                                 f"{shape[0]} instead!")
            elif ndim == 4 and not shape[0]:
                raise ValueError("Number of memory cells is zero!")

            assembled = self._modules_to_assembled(modules_data)
            if assembled.ndim == 3:
                assembled = assembled[slice(*self.pulse_id_range)]

            return assembled

    class AgipdImageAssembler(BaseAssembler):
        _modules = 16
        _module_shape = (512, 128)

        def _get_modules_bridge(self, data, src_name):
            """Overload."""
            # (memory cells, modules, y, x)
            return data[src_name]["image.data"]

        def _get_modules_file(self, data, src_name):
            """Overload."""
            # (memory cells, modules, y, x)
            return stack_detector_data(data, "image.data", only="AGIPD")

        def load_geometry(self, filename, quad_positions):
            """Overload."""
            try:
                self._geom = AGIPD_1MGeometry.from_crystfel_geom(filename)
                return True, f"Created Geometry from '{filename}'"
            except (ImportError, ModuleNotFoundError, OSError) as e:
                info = f"Failed to create Geometry from '{filename}'\n" + repr(e)
                return False, info

    class LpdImageAssembler(BaseAssembler):
        _modules = 16
        _module_shape = (256, 256)

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
                with File(filename, 'r') as f:
                    self._geom = LPDGeometry.from_h5_file_and_quad_positions(
                        f, quad_positions)
                return True, f"Created Geometry from '{filename}'"
            except OSError as e:
                info = f"Failed to create Geometry from '{filename}'\n" + repr(e)
                return False, info

    class JungFrauImageAssembler(BaseAssembler):
        _modules = 1
        _module_shape = (512, 1024)

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
        _modules = 1
        _module_shape = (1934, 960)

        def _get_modules_bridge(self, data, src_name):
            """Overload."""
            # (y, x, 1) -> (y, x)
            return data[src_name]["data.image"].squeeze(axis=-1)

        def _get_modules_file(self, data, src_name):
            """Overload."""
            # (y, x)
            return data[src_name]['data.image.pixels']

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

        raise NotImplementedError(f"Unknown detector type {detector}!")
