"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import unittest
from unittest.mock import MagicMock, patch, PropertyMock
import copy
import os
import random
import tempfile
import json

import pytest

import numpy as np
import h5py

from extra_foam.pipeline.processors.image_assembler import (
    _IMAGE_DTYPE, _RAW_IMAGE_DTYPE, ImageAssemblerFactory
)
from extra_foam.pipeline.exceptions import AssemblingError
from extra_foam.config import GeomAssembler, config, DataSource
from extra_foam.database import SourceCatalog, SourceItem


def _check_assembled_result(data, src):
    # test the module keys have been deleted
    assert data['raw'][src] is None

    assembled = data['assembled']['data']
    assert 3 == assembled.ndim
    assembled_shape = assembled.shape
    assert 4 == assembled_shape[0]
    assert assembled_shape[1] >= 1024
    assert assembled_shape[2] >= 1024


def _check_single_module_result(data, src, module_shape):
    # test the module keys have been deleted
    assert data['raw'][src] is None

    assembled = data['assembled']['data']
    assert 3 == assembled.ndim
    assembled_shape = assembled.shape
    assert 4 == assembled_shape[0]
    assert assembled_shape[-2:] == module_shape


_tmp_cfg_dir = tempfile.mkdtemp()


def setup_module(module):
    from extra_foam import config
    module._backup_ROOT_PATH = config.ROOT_PATH
    config.ROOT_PATH = _tmp_cfg_dir


def teardown_module(module):
    os.rmdir(_tmp_cfg_dir)
    from extra_foam import config
    config.ROOT_PATH = module._backup_ROOT_PATH


class TestAgipdAssembler:
    @classmethod
    def setup_class(cls):
        config.load('AGIPD', random.choice(['SPB', 'MID']))

        cls._geom_file = config["GEOMETRY_FILE"]
        cls._quad_positions = config["QUAD_POSITIONS"]

    @classmethod
    def teardown_class(cls):
        os.remove(config.config_file)

    def setup_method(self, method):
        self._assembler = ImageAssemblerFactory.create("AGIPD")
        # we do not test using GeomAssembler.EXTRA_GEOM for AGIPD
        self._assembler._load_geometry(
            False, self._geom_file, self._quad_positions, GeomAssembler.OWN)

    def _create_catalog(self, src_name, key_name):
        catalog = SourceCatalog()
        src = f'{src_name} {key_name}'
        catalog.add_item(SourceItem('AGIPD', src_name, [], key_name, slice(None, None), None))
        return src, catalog

    def testInvalidGeometryFile(self):
        # test file does not exist
        with pytest.raises(AssemblingError):
            self._assembler._load_geometry(False, "abc", self._quad_positions, GeomAssembler.OWN)

        # test invalid file
        with tempfile.NamedTemporaryFile() as fp:
            with pytest.raises(AssemblingError):
                self._assembler._load_geometry(
                    False, fp.name, self._quad_positions, GeomAssembler.OWN)

    def testGeneral(self):
        # Note: this test does not need to repeat for each detector
        key_name = 'image.data'

        # test required fields in metadata

        src, catalog = self._create_catalog('SPB_DET_AGIPD1M-1/DET/*CH0:xtdf', key_name)
        data = {
            'catalog': catalog,
            'meta': {
                src: {
                    'train_id': 10001,
                }
            },
            'raw': {
                src: {
                    'SPB_DET_AGIPD1M-1/DET/11CH0:xtdf':
                        {key_name: np.ones((4, 512, 128), dtype=_IMAGE_DTYPE)}}
            },
        }

        with pytest.raises(KeyError, match="source_type"):
            self._assembler.process(copy.deepcopy(data))

        data['meta'][src]["source_type"] = DataSource.FILE
        self._assembler.process(data)
        assert 10001 == data['raw']['META timestamp.tid']
        assert data['raw'][src] is None

    def testAssembleFileCal(self):
        self._runAssembleFileTest((4, 512, 128), _IMAGE_DTYPE)

    def testAssembleFileRaw(self):
        self._runAssembleFileTest((4, 2, 512, 128), _RAW_IMAGE_DTYPE)

    def _runAssembleFileTest(self, shape, dtype):
        key_name = 'image.data'
        src, catalog = self._create_catalog('SPB_DET_AGIPD1M-1/DET/*CH0:xtdf', key_name)

        data = {
            'catalog': catalog,
            'meta': {
                src: {
                    'train_id': 10001,
                    'source_type': DataSource.FILE,
                }
            },
            'raw': {
                src: {
                    'SPB_DET_AGIPD1M-1/DET/11CH0:xtdf':
                        {key_name: np.ones(shape, dtype=dtype)},
                    'SPB_DET_AGIPD1M-1/DET/7CH0:xtdf':
                        {key_name: np.ones(shape, dtype=dtype)},
                    'SPB_DET_AGIPD1M-1/DET/8CH0:xtdf':
                        {key_name: np.ones(shape, dtype=dtype)},
                    'SPB_DET_AGIPD1M-1/DET/3CH0:xtdf':
                        {key_name: np.ones(shape, dtype=dtype)},
                }
            },
        }

        self._assembler.process(data)
        _check_assembled_result(data, src)

    def testAssembleBridge(self):
        key_name = 'image.data'
        src, catalog = self._create_catalog('SPB_DET_AGIPD1M-1/CAL/APPEND_CORRECTED', key_name)

        data = {
            'catalog': catalog,
            'meta': {
                src: {
                    'train_id': 10001,
                    'source_type': DataSource.BRIDGE,
                }
            },
            'raw': {
                src: np.ones((4, 16, 100, 100), dtype=_IMAGE_DTYPE),
            },
        }

        with pytest.raises(AssemblingError, match='Expected module shape'):
            self._assembler.process(data)

        with pytest.raises(AssemblingError, match='modules, but'):
            data['raw'][src] = np.ones((4, 12, 512, 128), dtype=_IMAGE_DTYPE)
            self._assembler.process(data)

        with pytest.raises(AssemblingError, match='Number of memory cells'):
            data['raw'][src] = np.ones((0, 16, 512, 128), dtype=_IMAGE_DTYPE)
            self._assembler.process(data)

        # (modules, fs, ss, memory cells)
        data['raw'][src] = np.ones((16, 128, 512, 4), dtype=_IMAGE_DTYPE)
        self._assembler.process(data)
        _check_assembled_result(data, src)

        # (memory cells, modules, ss, fs)
        data['raw'][src] = np.ones((4, 16, 512, 128), dtype=_IMAGE_DTYPE)
        self._assembler.process(data)
        _check_assembled_result(data, src)

        # test single module

        # (modules, fs, ss, memory cells)
        data['raw'][src] = np.ones((1, 128, 512, 4), dtype=_IMAGE_DTYPE)
        self._assembler.process(data)
        _check_single_module_result(data, src, config["MODULE_SHAPE"])

        # (memory cells, modules, ss, fs)
        data['raw'][src] = np.ones((4, 1, 512, 128), dtype=_IMAGE_DTYPE)
        self._assembler.process(data)
        _check_single_module_result(data, src, config["MODULE_SHAPE"])


class TestLpdAssembler:
    @classmethod
    def setup_class(cls):
        config.load('LPD', 'FXE')

        cls._geom_file = config["GEOMETRY_FILE"]
        cls._quad_positions = config["QUAD_POSITIONS"]

    @classmethod
    def teardown_class(cls):
        os.remove(config.config_file)

    def setup_method(self, method):
        self._assembler = ImageAssemblerFactory.create("LPD")

    def _load_geometry(self, assembler_type, stack_only=False):
        self._assembler._load_geometry(
            stack_only, self._geom_file, self._quad_positions, assembler_type)

    def _create_catalog(self, src_name, key_name):
        catalog = SourceCatalog()
        src = f'{src_name} {key_name}'
        catalog.add_item(SourceItem('LPD', src_name, [], key_name, slice(None, None), None))
        return src, catalog

    @pytest.mark.parametrize("assembler_type", [GeomAssembler.EXTRA_GEOM, GeomAssembler.OWN])
    def testInvalidGeometryFile(self, assembler_type):
        # test file does not exist
        with pytest.raises(AssemblingError):
            self._assembler._load_geometry(False, "abc", self._quad_positions, assembler_type)

        # test invalid file (file signature not found)
        with tempfile.TemporaryFile() as fp:
            with pytest.raises(AssemblingError):
                self._assembler._load_geometry(False, fp, self._quad_positions, assembler_type)

        # test invalid h5 file (component not found)
        with tempfile.NamedTemporaryFile() as fp:
            fph5 = h5py.File(fp.name, 'r+')
            with pytest.raises(AssemblingError):
                self._assembler._load_geometry(
                    False, fp.name, self._quad_positions, assembler_type)

    @patch('extra_foam.ipc.ProcessLogger.info')
    def testUpdate(self, info):
        from extra_foam.geometries import LPD_1MGeometryFast
        from extra_foam.geometries import LPD_1MGeometry

        geom_cfg = {
            'stack_only': 'False',
            'assembler': '2',
            'geometry_file': self._geom_file,
            'coordinates': json.dumps([[0, 1], [1, 1], [1, 0], [0, 0]]),
        }

        # Note: this test does not need to repeat for each detector
        assembler = self._assembler

        assert assembler._assembler_type is None
        assembler._stack_only = True
        assembler.update(geom_cfg)
        assert isinstance(assembler._geom, LPD_1MGeometry)
        assert assembler._coordinates == [[0, 1], [1, 1], [1, 0], [0, 0]]
        assert assembler._geom_file == self._geom_file
        assert assembler._assembler_type == GeomAssembler.EXTRA_GEOM
        assert not assembler._stack_only

        # test assembler switching
        geom_cfg.update({'assembler': 1})
        assembler.update(geom_cfg)
        assert isinstance(assembler._geom, LPD_1MGeometryFast)
        geom_cfg.update({'assembler': 2})
        assembler.update(geom_cfg)
        assert isinstance(assembler._geom, LPD_1MGeometry)

        # test file and quad position change
        assembler._load_geometry = MagicMock()
        geom_cfg.update({'geometry_file': '/New/File'})
        assembler.update(geom_cfg)
        assembler._load_geometry.assert_called_once()
        assembler._load_geometry.reset_mock()
        geom_cfg.update({'coordinates':  json.dumps([[1, 1], [1, 1], [1, 0], [0, 0]])})
        assembler.update(geom_cfg)
        assembler._load_geometry.assert_called_once()
        assembler._load_geometry.reset_mock()

        # test stack only change
        info.reset_mock()
        assembler._load_geometry = MagicMock()
        geom_cfg.update({'stack_only': 'True'})
        assembler.update(geom_cfg)
        assembler._load_geometry.assert_called_once()
        assembler._load_geometry.reset_mock()
        info.assert_called_once()
        info.reset_mock()
        geom_cfg.update({'stack_only': 'False'})
        assembler.update(geom_cfg)
        assembler._load_geometry.assert_called_once()
        assembler._load_geometry.reset_mock()
        info.assert_called_once()

        # test mask tile switching
        assembler._out_array = np.ones((2, 2)).astype(_IMAGE_DTYPE)  # any value except None
        assembler.update(geom_cfg, mask_tile=True)
        assert assembler._mask_tile
        assert assembler._out_array is None
        assembler._out_array = np.ones((2, 2)).astype(_IMAGE_DTYPE)  # any value except None
        assembler.update(geom_cfg,  mask_tile=False)
        assert not assembler._mask_tile
        assert assembler._out_array is not None

    @pytest.mark.parametrize("assembler_type", [GeomAssembler.EXTRA_GEOM, GeomAssembler.OWN])
    def testAssembleFileCal(self, assembler_type):
        self._load_geometry(assembler_type)
        self._runAssembleFileTest((4, 256, 256), _IMAGE_DTYPE)

    @pytest.mark.parametrize("assembler_type", [GeomAssembler.EXTRA_GEOM, GeomAssembler.OWN])
    def testAssembleFileRaw(self, assembler_type):
        self._load_geometry(assembler_type)
        self._runAssembleFileTest((4, 1, 256, 256), _RAW_IMAGE_DTYPE)

    def _runAssembleFileTest(self, shape, dtype):
        key_name = 'image.data'
        src, catalog = self._create_catalog('FXE_DET_LPD1M-1/DET/*CH0:xtdf', key_name)

        data = {
            'catalog': catalog,
            'meta': {
                src: {
                    'train_id': 10001,
                    'source_type': DataSource.FILE,
                }
            },
            'raw': {
                src: {
                    'FXE_DET_LPD1M-1/DET/11CH0:xtdf':
                        {key_name: np.ones(shape, dtype=dtype)},
                    'FXE_DET_LPD1M-1/DET/7CH0:xtdf':
                        {key_name: np.ones(shape, dtype=dtype)},
                    'FXE_DET_LPD1M-1/DET/8CH0:xtdf':
                        {key_name: np.ones(shape, dtype=dtype)},
                    'FXE_DET_LPD1M-1/DET/3CH0:xtdf':
                        {key_name: np.ones(shape, dtype=dtype)},
                }
            },
        }

        self._assembler.process(data)
        _check_assembled_result(data, src)

    @pytest.mark.parametrize("assembler_type", [GeomAssembler.EXTRA_GEOM, GeomAssembler.OWN])
    def testAssembleBridge(self, assembler_type):
        self._load_geometry(assembler_type)

        key_name = 'image.data'
        src, catalog = self._create_catalog('FXE_DET_LPD1M-1/CAL/APPEND_CORRECTED', key_name)

        data = {
            'catalog': catalog,
            'meta': {
                src: {
                    'train_id': 10001,
                    'source_type': DataSource.BRIDGE,
                }
            },
            'raw': {
                src: np.ones((16, 100, 100, 4), dtype=_IMAGE_DTYPE)
            },
        }

        with pytest.raises(AssemblingError, match='Expected module shape'):
            self._assembler.process(data)

        with pytest.raises(AssemblingError, match='modules, but'):
            data['raw'][src] = np.ones((15, 256, 256, 4), dtype=_IMAGE_DTYPE)
            self._assembler.process(data)

        with pytest.raises(AssemblingError, match='Number of memory cells'):
            data['raw'][src] = np.ones((16, 256, 256, 0), dtype=_IMAGE_DTYPE)
            self._assembler.process(data)

        # (modules, x, y, memory cells)
        data['raw'][src] = np.ones((16, 256, 256, 4), dtype=_IMAGE_DTYPE)
        self._assembler.process(data)
        _check_assembled_result(data, src)

        # test single module
        data['raw'][src] = np.ones((1, 256, 256, 4), dtype=_IMAGE_DTYPE)
        self._assembler.process(data)
        _check_single_module_result(data, src, config["MODULE_SHAPE"])

    @pytest.mark.parametrize("assembler_type", [GeomAssembler.EXTRA_GEOM, GeomAssembler.OWN])
    def testOutArray(self, assembler_type):
        self._load_geometry(assembler_type)

        key_name = 'image.data'
        src, catalog = self._create_catalog('FXE_DET_LPD1M-1/CAL/APPEND_CORRECTED', key_name)

        data = {
            'catalog': catalog,
            'meta': {
                src: {
                    'train_id': 10001,
                    'source_type': DataSource.BRIDGE,
                }
            },
            'raw': {
                src: np.ones((16, 256, 256, 4), dtype=_IMAGE_DTYPE)
            },
        }
        self._assembler.process(data)

        assembled_shape = data['assembled']['data'].shape
        assert self._assembler._out_array.shape == assembled_shape
        assert _IMAGE_DTYPE == self._assembler._out_array.dtype

        # Test number of pulses change on the fly
        data['raw'][src] = np.ones((16, 256, 256, 10), dtype=_IMAGE_DTYPE)
        self._assembler.process(data)
        assembled_shape = data['assembled']['data'].shape
        assert self._assembler._out_array.shape == assembled_shape

        # test quad_positions (geometry) change on the fly
        quad_positions = [list(v) for v in self._quad_positions]
        quad_positions[0][1] += 2  # modify the quad positions
        quad_positions[3][0] -= 4
        quad_positions = [tuple(v) for v in quad_positions]
        self._assembler._load_geometry(False, self._geom_file, tuple(quad_positions), assembler_type)
        data['raw'][src] = np.ones((16, 256, 256, 10), dtype=_IMAGE_DTYPE)
        self._assembler.process(data)
        assembled_shape_old = assembled_shape
        assembled_shape = data['assembled']['data'].shape
        assert assembled_shape_old != assembled_shape
        assert self._assembler._out_array.shape == assembled_shape
        # change the geometry back
        self._assembler._load_geometry(False, self._geom_file, self._quad_positions, assembler_type)

    @pytest.mark.parametrize("assembler_type", [GeomAssembler.EXTRA_GEOM, GeomAssembler.OWN])
    def testAssembleDtype(self, assembler_type):
        self._load_geometry(assembler_type)

        key_name = 'image.data'
        src, catalog = self._create_catalog('FXE_DET_LPD1M-1/CAL/APPEND_CORRECTED', key_name)

        data = {
            'catalog': catalog,
            'meta': {
                src: {
                    'train_id': 10001,
                    'source_type': DataSource.BRIDGE,
                }
            },
            'raw': {
                src: np.ones((16, 256, 256, 4), dtype=np.float64)
            },
        }
        # dtype conversion float64 -> float32 throws TypeError (extra_geom)
        with pytest.raises(TypeError):
            self._assembler.process(data)

        data['raw'][src] = np.ones((16, 256, 256, 4), dtype=np.int16)
        if assembler_type == GeomAssembler.EXTRA_GEOM:
            self._assembler.process(data)

            assembled_dtype = data['assembled']['data'].dtype
            assert _IMAGE_DTYPE == assembled_dtype
        else:
            # C++ implementation does not allow any implicit type conversion
            with pytest.raises(TypeError):
                self._assembler.process(data)


class TestDSSCAssembler:
    @classmethod
    def setup_class(cls):
        config.load('DSSC', 'SCS')

        cls._geom_file = config["GEOMETRY_FILE"]
        cls._quad_positions = config["QUAD_POSITIONS"]

    @classmethod
    def teardown_class(cls):
        os.remove(config.config_file)

    def setup_method(self, method):
        self._assembler = ImageAssemblerFactory.create("DSSC")

    def _load_geometry(self, assembler_type, stack_only=False):
        self._assembler._load_geometry(
            stack_only, self._geom_file, self._quad_positions, assembler_type)

    def _create_catalog(self, src_name, key_name):
        catalog = SourceCatalog()
        src = f'{src_name} {key_name}'
        catalog.add_item(SourceItem('DSSC', src_name, [], key_name, slice(None, None), None))
        return src, catalog

    @pytest.mark.parametrize("assembler_type", [GeomAssembler.EXTRA_GEOM, GeomAssembler.OWN])
    def testInvalidGeometryFile(self, assembler_type):
        # test file does not exist
        with pytest.raises(AssemblingError):
            self._assembler._load_geometry(False, "abc", self._quad_positions, assembler_type)

        # test invalid file (file signature not found)
        with tempfile.TemporaryFile() as fp:
            with pytest.raises(AssemblingError):
                self._assembler._load_geometry(False, fp, self._quad_positions, assembler_type)

        # test invalid h5 file (component not found)
        with tempfile.NamedTemporaryFile() as fp:
            fph5 = h5py.File(fp.name, 'r+')
            with pytest.raises(AssemblingError):
                self._assembler._load_geometry(False, fp.name, self._quad_positions, assembler_type)

    @pytest.mark.parametrize("assembler_type", [GeomAssembler.EXTRA_GEOM, GeomAssembler.OWN])
    def testAssembleFileCal(self, assembler_type):
        self._load_geometry(assembler_type)
        self._runAssembleFileTest((4, 128, 512), _IMAGE_DTYPE)

    @pytest.mark.parametrize("assembler_type", [GeomAssembler.EXTRA_GEOM, GeomAssembler.OWN])
    def testAssembleFileRaw(self, assembler_type):
        self._load_geometry(assembler_type)
        self._runAssembleFileTest((4, 1, 128, 512), _RAW_IMAGE_DTYPE)

    def _runAssembleFileTest(self, shape, dtype):
        key_name = 'image.data'
        src, catalog = self._create_catalog('SCS_DET_DSSC1M-1/DET/*CH0:xtdf', key_name)

        data = {
            'catalog': catalog,
            'meta': {
                src: {
                    'train_id': 10001,
                    'source_type': DataSource.FILE,
                }
            },
            'raw': {
                src: {
                    'SCS_DET_DSSC1M-1/DET/11CH0:xtdf':
                        {key_name: np.ones(shape, dtype=dtype)},
                    'SCS_DET_DSSC1M-1/DET/7CH0:xtdf':
                        {key_name: np.ones(shape, dtype=dtype)},
                    'SCS_DET_DSSC1M-1/DET/8CH0:xtdf':
                        {key_name: np.ones(shape, dtype=dtype)},
                    'SCS_DET_DSSC1M-1/DET/3CH0:xtdf':
                        {key_name: np.ones(shape, dtype=dtype)},
                }
            },
        }

        self._assembler.process(data)
        _check_assembled_result(data, src)

        # test invalid data type
        data['raw'][src] = np.ones((4, 1, 128, 512), dtype=np.float64)
        with pytest.raises(AssemblingError):
            self._assembler.process(data)

        data['raw'][src] = np.ones((4, 1, 128, 512), dtype=np.int64)
        with pytest.raises(AssemblingError):
            self._assembler.process(data)

    @pytest.mark.parametrize("assembler_type", [GeomAssembler.EXTRA_GEOM, GeomAssembler.OWN])
    def testAssembleBridge(self, assembler_type):
        self._load_geometry(assembler_type)

        key_name = 'image.data'
        src, catalog = self._create_catalog('SCS_CDIDET_DSSC/CAL/APPEND_CORRECTED', key_name)

        data = {
            'catalog': catalog,
            'meta': {
                src: {
                    'train_id': 10001,
                    'source_type': DataSource.BRIDGE,
                }
            },
            'raw': {
                src: np.ones((16, 100, 100, 4), dtype=_IMAGE_DTYPE),
            },
        }

        with pytest.raises(AssemblingError, match='Expected module shape'):
            self._assembler.process(data)

        with pytest.raises(AssemblingError, match='modules, but'):
            data['raw'][src] = np.ones((15, 512, 128, 4), dtype=_IMAGE_DTYPE)
            self._assembler.process(data)

        with pytest.raises(AssemblingError, match='Number of memory cells'):
            data['raw'][src] = np.ones((16, 512, 128, 0), dtype=_IMAGE_DTYPE)
            self._assembler.process(data)

        # (modules, x, y, memory cells)
        data['raw'][src] = np.ones((16, 512, 128, 4), dtype=_IMAGE_DTYPE)
        self._assembler.process(data)
        _check_assembled_result(data, src)

        # test single module
        data['raw'][src] = np.ones((1, 512, 128, 4), dtype=_IMAGE_DTYPE)
        self._assembler.process(data)
        _check_single_module_result(data, src, config["MODULE_SHAPE"])

    @pytest.mark.parametrize("assembler_type", [GeomAssembler.EXTRA_GEOM, GeomAssembler.OWN])
    def testOutArray(self, assembler_type):
        self._load_geometry(assembler_type)

        key_name = 'image.data'
        src, catalog = self._create_catalog('SCS_CDIDET_DSSC/CAL/APPEND_CORRECTED', key_name)

        data = {
            'catalog': catalog,
            'meta': {
                src: {
                    'train_id': 10001,
                    'source_type': DataSource.BRIDGE,
                }
            },
            'raw': {
                src: np.ones((16, 512, 128, 4), dtype=_IMAGE_DTYPE)
            },
        }
        self._assembler.process(data)
        assembled_shape = data['assembled']['data'].shape
        np.testing.assert_array_equal(self._assembler._out_array.shape, assembled_shape)
        assert _IMAGE_DTYPE == self._assembler._out_array.dtype

        # Test output array shape change on the fly
        data['raw'][src] = np.ones((16, 512, 128, 10), dtype=_IMAGE_DTYPE)
        self._assembler.process(data)
        assembled_shape = data['assembled']['data'].shape
        np.testing.assert_array_equal(self._assembler._out_array.shape, assembled_shape)

    @pytest.mark.parametrize("assembler_type", [GeomAssembler.EXTRA_GEOM, GeomAssembler.OWN])
    def testAssembleDtype(self, assembler_type):
        self._load_geometry(assembler_type)

        key_name = 'image.data'
        src, catalog = self._create_catalog('SCS_CDIDET_DSSC/CAL/APPEND_CORRECTED', key_name)

        data = {
            'catalog': catalog,
            'meta': {
                src: {
                    'train_id': 10001,
                    'source_type': DataSource.BRIDGE,
                }
            },
            'raw': {
                src: np.ones((16, 512, 128, 4), dtype=np.float64)
            },
        }
        # dtype conversion float64 -> float32 throws TypeError (extra_geom)
        with pytest.raises(TypeError):
            self._assembler.process(data)

        data['raw'][src] = np.ones((16, 512, 128, 4), dtype=np.int16)
        if assembler_type == GeomAssembler.EXTRA_GEOM:
            self._assembler.process(data)

            assembled_dtype = data['assembled']['data'].dtype
            assert _IMAGE_DTYPE == assembled_dtype
        else:
            # C++ implementation does not allow any implicit type conversion
            with pytest.raises(TypeError):
                self._assembler.process(data)


class TestJungfrauAssembler(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        config.load('JungFrau', 'FXE')

    @classmethod
    def tearDownClass(cls):
        os.remove(config.config_file)

    def setUp(self):
        self._assembler = ImageAssemblerFactory.create("JungFrau")

    def _create_catalog(self, src_name, key_name):
        catalog = SourceCatalog()
        src = f'{src_name} {key_name}'
        catalog.add_item(SourceItem('JungFrau', src_name, [], key_name, None, None))
        return src, catalog

    def testAssembleFile(self):
        key_name = 'data.adc'
        src, catalog = self._create_catalog('FXE_XAD_JF1M/DET/RECEIVER-1:daqOutput', key_name)

        data = {
            'catalog': catalog,
            'meta': {
                src: {
                    'train_id': 10001,
                    'source_type': DataSource.FILE,
                }
            },
            'raw': {
                src: np.ones((1, 512, 1024))
            },
        }

        self._assembler.process(data)
        self.assertIsNone(data['raw'][src])

        with self.assertRaisesRegex(AssemblingError, 'Expected module shape'):
            data['raw'][src] = np.ones((1, 100, 100))
            self._assembler.process(data)

        with self.assertRaisesRegex(NotImplementedError, "Use 'JungFrauPR'"):
            data['raw'][src] = np.ones((2, 512, 1024))
            self._assembler.process(data)

    def testAssembleBridge(self):
        key_name = 'data.adc'
        src, catalog = self._create_catalog('FXE_XAD_JF1M/DET/RECEIVER-1:display', key_name)

        data = {
            'catalog': catalog,
            'meta': {
                src: {
                    'train_id': 10001,
                    'source_type': DataSource.BRIDGE,
                }
            },
            'raw': {
                src: np.ones((512, 1024, 1))
            },
        }
        self._assembler.process(data)
        self.assertIsNone(data['raw'][src])

        data['raw'][src] = np.ones((100, 100, 1))
        with self.assertRaisesRegex(AssemblingError, 'Expected module shape'):
            self._assembler.process(data)

        data['raw'][src] = np.ones((512, 1024, 2))
        with self.assertRaisesRegex(NotImplementedError, "Use 'JungFrauPR'"):
            self._assembler.process(data)


class TestJungfrauPulseResolvedAssembler(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        config.load('JungFrauPR', 'FXE')

    @classmethod
    def tearDownClass(cls):
        os.remove(config.config_file)

    def setUp(self):
        self._assembler = ImageAssemblerFactory.create("JungFrauPR")

    def _create_catalog(self, src_name, key_name, n_modules=1):
        catalog = SourceCatalog()
        src = f'{src_name} {key_name}'
        modules = [] if n_modules == 1 else list(range(1, n_modules+1))
        catalog.add_item(SourceItem('JungFrauPR', src_name, modules, key_name, None, None))
        return src, catalog

    def _check6ModuleResult(self, data, src):
        # test the module keys have been deleted
        assert data['raw'][src] is None

        assembled = data['assembled']['data']
        assert 3 == assembled.ndim
        assembled_shape = assembled.shape
        assert assembled_shape == (16, 1536, 2048)
        assert _IMAGE_DTYPE == assembled.dtype

    def testAssembleFileCal(self):
        # 16 is the number of memory cells
        self._runAssembleFileTest((16, 512, 1024), _IMAGE_DTYPE)

    def testAssembleFileRaw(self):
        self._runAssembleFileTest((16, 512, 1024), _RAW_IMAGE_DTYPE)

    def _runAssembleFileTest(self, shape, dtype):
        self._assembler._n_modules = 1

        key_name = 'data.adc'
        src, catalog = self._create_catalog(
            'FXE_XAD_JF1M/DET/RECEIVER-1:daqOutput', key_name)

        data = {
            'catalog': catalog,
            'meta': {
                src: {
                    'train_id': 10001,
                    'source_type': DataSource.FILE,
                }
            },
            'raw': {
                src: np.ones(shape, dtype=dtype)
            },
        }

        self._assembler.process(data)
        self.assertIsNone(data['raw'][src])
        assembled = data['assembled']['data']
        self.assertTupleEqual(assembled.shape, shape)
        assert _IMAGE_DTYPE == assembled.dtype

        with self.assertRaisesRegex(AssemblingError, 'Expected module shape'):
            data['raw'][src] = np.ones((16, 100, 100), dtype=dtype)
            self._assembler.process(data)

    def testAssembleFile6ModulesCal(self):
        self._runAssembleFileTest((16, 512, 1024), _IMAGE_DTYPE)

    def testAssembleFile6ModulesRaw(self):
        self._runAssembleFileTest((16, 512, 1024), _RAW_IMAGE_DTYPE)

    def _runAssembleFile6ModulesTest(self, shape, dtype):
        self._assembler._n_modules = 6
        self._assembler._load_geometry(True, None, None, GeomAssembler.OWN)

        key_name = 'data.adc'
        src, catalog = self._create_catalog(
            'FXE_XAD_JF1M/DET/RECEIVER-*:daqOutput', key_name, n_modules=6)

        data = {
            'catalog': catalog,
            'meta': {
                src: {
                    'train_id': 10001,
                    'source_type': DataSource.FILE,
                }
            },
            'raw': {
                src: {
                    'FXE_XAD_JF1M/DET/RECEIVER-1:daqOutput':
                        {key_name: np.ones(shape, dtype=dtype)},
                    'FXE_XAD_JF1M/DET/RECEIVER-2:daqOutput':
                        {key_name: np.ones(shape, dtype=dtype)},
                    'FXE_XAD_JF1M/DET/RECEIVER-3:daqOutput':
                        {key_name: np.ones(shape, dtype=dtype)},
                    'FXE_XAD_JF1M/DET/RECEIVER-4:daqOutput':
                        {key_name: np.ones(shape, dtype=dtype)},
                }
            },
        }

        self._assembler.process(data)
        self._check6ModuleResult(data, src)

    def testAssembleBridge(self):
        self._assembler._n_modules = 1

        key_name = 'data.adc'
        src, catalog = self._create_catalog('FXE_XAD_JF1M/DET/RECEIVER-1:display', key_name)

        data = {
            'catalog': catalog,
            'meta': {
                src: {
                    'train_id': 10001,
                    'source_type': DataSource.BRIDGE,
                }
            },
            'raw': {
                src: np.ones((2, 512, 1024, 16))
            },
        }

        with pytest.raises(AssemblingError, match='1 modules, but'):
            self._assembler.process(data)

        # (y, x, memory cells)
        data['raw'][src] = np.ones((512, 1024, 16))
        self._assembler.process(data)
        self.assertIsNone(data['raw'][src])
        assembled = data['assembled']['data']
        self.assertTupleEqual(assembled.shape, (16, 512, 1024))

    def testAssembleBridge6ModulesCal(self):
        self._runAssembleBridge6ModulesTest((512, 1024, 16), _IMAGE_DTYPE)

    def testAssembleBridge6ModulesRaw(self):
        # TODO
        pass

    def _runAssembleBridge6ModulesTest(self, shape, dtype):
        self._assembler._n_modules = 6
        self._assembler._load_geometry(True, None, None, GeomAssembler.OWN)

        key_name = "data.adc"
        src, catalog = self._create_catalog(
            'FXE_XAD_JF1M/DET/RECEIVER-*:display', key_name, n_modules=6)

        data = {
            'catalog': catalog,
            'meta': {
                src: {
                    'train_id': 10001,
                    'source_type': DataSource.BRIDGE,
                }
            },
            'raw': {
                src: {
                    'FXE_XAD_JF1M/DET/RECEIVER-1:display':
                        {key_name: np.ones(shape, dtype=dtype)},
                    'FXE_XAD_JF1M/DET/RECEIVER-2:display':
                        {key_name: np.ones(shape, dtype=dtype)},
                    'FXE_XAD_JF1M/DET/RECEIVER-3:display':
                        {key_name: np.ones(shape, dtype=dtype)},
                    'FXE_XAD_JF1M/DET/RECEIVER-6:display':
                        {key_name: np.ones(shape, dtype=dtype)},
                }
            },
        }

        self._assembler.process(data)
        self._check6ModuleResult(data, src)

    def testAssembleBridge6ModulesStacked(self):
        self._assembler._n_modules = 6
        self._assembler._load_geometry(True, None, None, GeomAssembler.OWN)

        key_name = 'data.adc'
        src, catalog = self._create_catalog('FXE_XAD_JF1M/DET/RECEIVER', key_name)

        data = {
            'catalog': catalog,
            'meta': {
                src: {
                    'train_id': 10001,
                    'source_type': DataSource.BRIDGE,
                }
            },
            'raw': {
                src: np.ones((6, 512, 1024, 16), dtype=_IMAGE_DTYPE),
            },
        }

        # (modules, y, x, memory cells)
        self._assembler.process(data)
        self._check6ModuleResult(data, src)

        # (memory cells, modules, y, x)
        data['raw'][src] = np.ones((16, 6, 512, 1024), dtype=_IMAGE_DTYPE)
        self._assembler.process(data)
        self._check6ModuleResult(data, src)

        with pytest.raises(AssemblingError, match='Expected module shape'):
            data['raw'][src] = np.ones((6, 100, 100, 16), dtype=_IMAGE_DTYPE)
            self._assembler.process(data)

        with pytest.raises(AssemblingError, match='modules, but'):
            data['raw'][src] = np.ones((2, 512, 1024, 2), dtype=_IMAGE_DTYPE)
            self._assembler.process(data)


class TestFastccdAssembler(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        config.load('FastCCD', 'SCS')

    @classmethod
    def tearDownClass(cls):
        os.remove(config.config_file)

    def setUp(self):
        self._assembler = ImageAssemblerFactory.create("FastCCD")

    def _create_catalog(self, src_name, key_name):
        catalog = SourceCatalog()
        src = f'{src_name} {key_name}'
        catalog.add_item(SourceItem('FastCCD', src_name, [], key_name, None, None))
        return src, catalog

    def testAssembleFile(self):
        key_name = 'data.image.pixels'
        src, catalog = self._create_catalog('SCS_CDIDET_FCCD2M/DAQ/FCCD:daqOutput', key_name)

        data = {
            'catalog': catalog,
            'meta': {
                src: {
                    'train_id': 10001,
                    'source_type': DataSource.FILE,
                }
            },
            'raw': {
                src: np.ones((1934, 960))
            },
        }
        self._assembler.process(data)
        self.assertIsNone(data['raw'][src])

        with self.assertRaisesRegex(AssemblingError, 'Expected module shape'):
            data['raw'][src] = np.ones((100, 100))
            self._assembler.process(data)

    def testAssembleBridgeCal(self):
        self._runAssembleBridgeTest((1934, 960, 1), _IMAGE_DTYPE, "data.image")

    def testAssembleBridgeRaw(self):
        self._runAssembleBridgeTest((1934, 960), _RAW_IMAGE_DTYPE, "data.image.data")

    def _runAssembleBridgeTest(self, shape, dtype, key):
        src, catalog = self._create_catalog('SCS_CDIDET_FCCD2M/DAQ/FCCD:display', key)

        data = {
            'catalog': catalog,
            'meta': {
                src: {
                    'train_id': 10001,
                    'source_type': DataSource.BRIDGE,
                }
            },
            'raw': {
                src: np.ones(shape, dtype=dtype)
            },
        }

        self._assembler.process(data)
        self.assertIsNone(data['raw'][src])

        with self.assertRaisesRegex(AssemblingError, 'Expected module shape'):
            data['raw'][src] = np.ones((100, 100, 1), dtype=dtype)
            self._assembler.process(data)


class TestEPix100Assembler(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        config.load('ePix100', 'MID')

    @classmethod
    def tearDownClass(cls):
        os.remove(config.config_file)

    def setUp(self):
        self._assembler = ImageAssemblerFactory.create("ePix100")

    def _create_catalog(self, src_name, key_name, n_modules=1):
        catalog = SourceCatalog()
        src = f'{src_name} {key_name}'
        modules = [] if n_modules == 1 else list(range(1, n_modules+1))
        catalog.add_item(SourceItem('ePix100', src_name, modules, key_name, None, None))
        return src, catalog

    def _check4ModuleResult(self, data, src):
        # test the module keys have been deleted
        assert data['raw'][src] is None

        assembled = data['assembled']['data']
        assert assembled.ndim == 2
        assembled_shape = assembled.shape
        assert assembled_shape == (1416, 1536)
        assert _IMAGE_DTYPE == assembled.dtype

    def testAssembleFileCal(self):
        self._runAssembleFileTest((708, 768), _IMAGE_DTYPE)

    def testAssembleFileRaw(self):
        self._runAssembleFileTest((708, 768), np.int16)

    def _runAssembleFileTest(self, shape, dtype):
        key_name = 'data.image.pixels'
        src, catalog = self._create_catalog('MID_EXP_EPIX-1/DET/RECEIVER:daqOutput', key_name)

        data = {
            'catalog': catalog,
            'meta': {
                src: {
                    'train_id': 10001,
                    'source_type': DataSource.FILE,
                }
            },
            'raw': {
                src: np.ones(shape, dtype=dtype),
            },
        }

        self._assembler.process(data)
        self.assertIsNone(data['raw'][src])
        assembled = data['assembled']['data']
        self.assertTupleEqual(assembled.shape, shape)
        assert _IMAGE_DTYPE == assembled.dtype

        with self.assertRaisesRegex(AssemblingError, 'Expected module shape'):
            data['raw'][src] = np.ones((100, 100), dtype=dtype)
            self._assembler.process(data)

    def testAssembleFile4ModulesCal(self):
        self._runAssembleFile4ModulesTest((708, 768), _IMAGE_DTYPE)

    def testAssembleFile4ModulesRaw(self):
        self._runAssembleFile4ModulesTest((708, 768), np.int16)

    def _runAssembleFile4ModulesTest(self, shape, dtype):
        self._assembler._n_modules = 4
        self._assembler._load_geometry(True, None, None, GeomAssembler.OWN)

        key_name = 'data.image.pixels'
        src, catalog = self._create_catalog(
            'MID_EXP_EPIX-*/DET/RECEIVER:daqOutput', key_name, n_modules=4)

        data = {
            'catalog': catalog,
            'meta': {
                src: {
                    'train_id': 10001,
                    'source_type': DataSource.FILE,
                }
            },
            'raw': {
                src: {
                    'MID_EXP_EPIX-1/DET/RECEIVER:daqOutput':
                        {key_name: np.ones(shape, dtype=dtype)},
                    'MID_EXP_EPIX-2/DET/RECEIVER:daqOutput':
                        {key_name: np.ones(shape, dtype=dtype)},
                },
            }
        }

        self._assembler.process(data)
        self._check4ModuleResult(data, src)

    def testAssembleBridgeCal(self):
        self._runAssembleBridgeTest((708, 768, 1), _IMAGE_DTYPE, "data.image")

    def testAssembleBridgeRaw(self):
        self._runAssembleBridgeTest((1, 708, 768), np.int16, "data.image.data")

    def _runAssembleBridgeTest(self, shape, dtype, key):
        self._assembler._n_modules = 1
        src, catalog = self._create_catalog('MID_EXP_EPIX-1/DET/RECEIVER:daqOutput', key)

        data = {
            'catalog': catalog,
            'meta': {
                src: {
                    'train_id': 10001,
                    'source_type': DataSource.BRIDGE,
                }
            },
            'raw': {
                src: np.ones(shape, dtype=dtype)
            },
        }

        self._assembler.process(data)
        self.assertIsNone(data['raw'][src])

        with self.assertRaisesRegex(AssemblingError, 'Expected module shape'):
            if dtype == _IMAGE_DTYPE:
                data['raw'][src] = np.ones((100, 100, 1), dtype=dtype)
            else:
                data['raw'][src] = np.ones((1, 100, 100), dtype=np.int16)
            self._assembler.process(data)

    def testAssembleBridge4ModulesCal(self):
        self._runAssembleBridge4ModulesTest((708, 768, 1), _IMAGE_DTYPE, "data.image")

    def testAssembleBridge4ModulesRaw(self):
        self._runAssembleBridge4ModulesTest((1, 708, 768), np.int16, "data.image.data")

    def _runAssembleBridge4ModulesTest(self, shape, dtype, key_name):
        self._assembler._n_modules = 4
        self._assembler._load_geometry(True, None, None, GeomAssembler.OWN)

        src, catalog = self._create_catalog(
            'MID_EXP_EPIX-*/DET/RECEIVER:daqOutput', key_name, n_modules=4)

        data = {
            'catalog': catalog,
            'meta': {
                src: {
                    'train_id': 10001,
                    'source_type': DataSource.BRIDGE,
                }
            },
            'raw': {
                src: {
                    'MID_EXP_EPIX-1/DET/RECEIVER:daqOutput':
                        {key_name: np.ones(shape, dtype=dtype)},
                    'MID_EXP_EPIX-2/DET/RECEIVER:daqOutput':
                        {key_name: np.ones(shape, dtype=dtype)},
                }
            },
        }

        self._assembler.process(data)
        self._check4ModuleResult(data, src)

    def testAssembleBridge4ModulesStacked(self):
        self._assembler._n_modules = 4
        self._assembler._load_geometry(True, None, None, GeomAssembler.OWN)

        src, catalog = self._create_catalog(
            'MID_EXP_EPIX/DET/RECEIVER', "data.image", n_modules=4)

        data = {
            'catalog': catalog,
            'meta': {
                src: {
                    'train_id': 10001,
                    'source_type': DataSource.BRIDGE,
                }
            },
            'raw': {
                src: np.ones((4, 708, 768), dtype=_IMAGE_DTYPE),
            },
        }

        self._assembler.process(data)
        self._check4ModuleResult(data, src)

        with pytest.raises(AssemblingError, match='Expected module shape'):
            data['raw'][src] = np.ones((4, 100, 100), dtype=_IMAGE_DTYPE)
            self._assembler.process(data)

        with pytest.raises(AssemblingError, match='modules, but'):
            data['raw'][src] = np.ones((2, 708, 768), dtype=_IMAGE_DTYPE)
            self._assembler.process(data)


class TestBaslerCameraAssembler(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        config.load('BaslerCamera', 'SCS')

    @classmethod
    def tearDownClass(cls):
        os.remove(config.config_file)

    def setUp(self):
        self._assembler = ImageAssemblerFactory.create("BaslerCamera")

    def _create_catalog(self, src_name, key_name):
        catalog = SourceCatalog()
        src = f'{src_name} {key_name}'
        catalog.add_item(SourceItem('BaslerCamera', src_name, [], key_name, None, None))
        return src, catalog

    def testAssembleBridge(self):
        key_name = 'data.image.data'
        src, catalog = self._create_catalog("baslercamera_module", key_name)

        data = {
            'catalog': catalog,
            'meta': {
                src: {
                    'train_id': 10001,
                    'source_type': DataSource.BRIDGE,
                }
            },
            'raw': {
                src: np.ones((1024, 1024))
            },
        }

        self._assembler.process(data)
        self.assertIsNone(data['raw'][src])
