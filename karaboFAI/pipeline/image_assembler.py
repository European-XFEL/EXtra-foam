"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

ImageAssembler.

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
        """Abstract ImageAssembler class

        Attributes:
            _source_name (str): detector data source name.
            _source_type (DataSource): detector data source type.
            _calibrated (bool): True for calibrated data and False for
                raw data.
            _from_file (bool): True for data streamed from files and
                False for data from the online ZMQ bridge.
            pulse_id_range (tuple): (min, max) pulse ID to be processed.
                (int, int)
        """
        def __init__(self):
            """Initialization."""
            self._source_name = None

            self._source_type = None
            self._from_file = None
            self._calibrated = None

            self._geom = None

            self.pulse_id_range = None

        @property
        def source_name(self):
            return self._source_name

        @source_name.setter
        def source_name(self, name):
            self._source_name = name

        @property
        def source_type(self):
            return self._source_type

        @source_type.setter
        def source_type(self, v):
            self._source_type = v
            if v == DataSource.CALIBRATED_FILES:
                self._from_file = True
                self._calibrated = True
            elif v == DataSource.CALIBRATED_BRIDGE:
                self._from_file = False
                self._calibrated = True
            elif v == DataSource.RAW_FILES:
                self._from_file = True
                self._calibrated = False
            elif v == DataSource.RAW_BRIDGE:
                self._from_file = False
                self._calibrated = False
            else:
                raise ValueError("Unknown data source type!")

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
            src_name = self._source_name
            from_file = self._from_file
            calibrated = self._calibrated

            data, meta = data

            if from_file:
                # get the train ID of the first metadata
                tid = next(iter(meta.values()))["timestamp.tid"]
                modules_data = self._get_modules_file(data, src_name)
            else:
                tid = meta[src_name]["timestamp.tid"]
                modules_data = self._get_modules_bridge(data, src_name)

            assembled = self._modules_to_assembled(modules_data)

            if assembled is not None and assembled.ndim == 3:
                assembled = assembled[slice(*self.pulse_id_range)]

            return tid, assembled

    class AgipdImageAssembler(BaseAssembler):
        def _get_modules_bridge(self, data, src_name):
            """Overload."""
            return data[src_name]["image.data"]

        def _get_modules_file(self, data, src_name):
            """Overload."""
            return stack_detector_data(data, "image.data", only="AGIPD")

        def load_geometry(self, filename, quad_positions):
            """Overload."""
            try:
                self._geom = AGIPD_1MGeometry.from_crystfel_geom(filename)
                return True, f"Created Geometry from '{filename}'"
            except (ImportError, ModuleNotFoundError, OSError) as e:
                info = f"Failed to create Geometry from '{filename}'\n" + str(e)
                return False, info

    class LpdImageAssembler(BaseAssembler):
        def _get_modules_bridge(self, data, src_name):
            """Overload."""
            # (modules, x, y, memory cells) -> (memory cells, modules, y, x)
            return np.moveaxis(
                np.moveaxis(data[src_name]["image.data"], 3, 0), 3, 2)

        def _get_modules_file(self, data, src_name):
            """Overload."""
            return stack_detector_data(data, "image.data", only="LPD")

        def load_geometry(self, filename, quad_positions):
            """Overload."""
            try:
                with File(filename, 'r') as f:
                    self._geom = LPDGeometry.from_h5_file_and_quad_positions(
                        f, quad_positions)
                return True, f"Created Geometry from '{filename}'"
            except OSError as e:
                info = f"Failed to create Geometry from '{filename}'\n" + str(e)
                return False, info

    class JungFrauImageAssembler(BaseAssembler):
        def _get_modules_bridge(self, data, src_name):
            """Overload."""
            # (y, x, modules) -> (y, x)
            modules_data = data[src_name]["data.adc"]
            if modules_data.shape[-1] == 1:
                return modules_data.squeeze(axis=-1)
            else:
                raise NotImplementedError

        def _get_modules_file(self, data, src_name):
            """Overload."""
            # (modules, y, x) -> (y, x)
            modules_data = data[src_name]['data.adc']
            if modules_data.shape[0] == 1:
                return modules_data.squeeze(axis=0)
            else:
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

        raise NotImplementedError
