"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import unittest
import copy
import os
import random
import tempfile

import pytest

import numpy as np

from extra_foam.pipeline.processors.image_assembler import (
    _IMAGE_DTYPE, _RAW_IMAGE_DTYPE, ImageAssemblerFactory
)
from extra_foam.pipeline.processors.tests import _BaseProcessorTest
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


_tmp_cfg_dir = tempfile.mkdtemp()


def setup_module(module):
    from extra_foam import config
    module._backup_ROOT_PATH = config.ROOT_PATH
    config.ROOT_PATH = _tmp_cfg_dir


def teardown_module(module):
    os.rmdir(_tmp_cfg_dir)
    from extra_foam import config
    config.ROOT_PATH = module._backup_ROOT_PATH


class TestAgipdAssembler(unittest.TestCase, _BaseProcessorTest):
    @classmethod
    def setUpClass(cls):
        config.load('AGIPD', random.choice(['SPB', 'MID']))

        cls._geom_file = config["GEOMETRY_FILE"]
        cls._quad_positions = config["QUAD_POSITIONS"]

    @classmethod
    def tearDownClass(cls):
        os.remove(config.config_file)

    def setUp(self):
        self._assembler = ImageAssemblerFactory.create("AGIPD")
        self._assembler._load_geometry(self._geom_file, self._quad_positions)

    def _create_catalog(self, src_name, key_name):
        catalog = SourceCatalog()
        src = f'{src_name} {key_name}'
        catalog.add_item(SourceItem('AGIPD', src_name, [], key_name, slice(None, None), None))
        return src, catalog

    def testGeneral(self):
        # Note: this test does not need to repeat for each detector
        key_name = 'image.data'
        src, catalog = self._create_catalog('SPB_DET_AGIPD1M-1/DET/*CH0:xtdf', key_name)

        # test detector sources not found
        data = {
            'catalog': catalog,
            'meta': {},
            'raw': {},
        }

        with self.assertRaisesRegex(AssemblingError, "AGIPD source"):
            self._assembler.process(data)

        # test required fields in metadata

        data = {
            'catalog': catalog,
            'meta': {
                src: {
                    'tid': 10001,
                }
            },
            'raw': {
                src: {
                    'SPB_DET_AGIPD1M-1/DET/11CH0:xtdf':
                        {key_name: np.ones((4, 512, 128), dtype=_IMAGE_DTYPE)}}
            },
        }

        with self.assertRaisesRegex(KeyError, "source_type"):
            self._assembler.process(copy.deepcopy(data))

        data['meta'][src]["source_type"] = DataSource.FILE
        self._assembler.process(data)
        self.assertEqual(10001, data['raw']['META timestamp.tid'])
        self.assertIsNone(data['raw'][src])

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
                    'tid': 10001,
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

        with self.assertRaisesRegex(AssemblingError, 'Expected module shape'):
            data = {
                'catalog': catalog,
                'meta': {
                    src: {
                        'tid': 10001,
                        'source_type': DataSource.BRIDGE,
                    }
                },
                'raw': {
                    src: np.ones((4, 16, 100, 100), dtype=_IMAGE_DTYPE),
                },
            }
            self._assembler.process(data)

        with self.assertRaisesRegex(AssemblingError, 'modules, but'):
            data['raw'][src] = np.ones((4, 12, 512, 128), dtype=_IMAGE_DTYPE)
            self._assembler.process(data)

        with self.assertRaisesRegex(AssemblingError, 'Number of memory cells'):
            data['raw'][src] = np.ones((0, 16, 512, 128), dtype=_IMAGE_DTYPE)
            self._assembler.process(data)

        expected_shape = (4, 16, 512, 128)
        # (modules, fs, ss, memory cells)
        data['raw'][src] = np.ones((16, 128, 512, 4), dtype=_IMAGE_DTYPE)
        self._assembler.process(data)
        _check_assembled_result(data, src)

        # (memory cells, modules, ss, fs)
        data['raw'][src] = np.ones((4, 16, 512, 128), dtype=_IMAGE_DTYPE)
        self._assembler.process(data)
        _check_assembled_result(data, src)


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
        self._load_geometry(GeomAssembler.EXTRA_GEOM)

    def _load_geometry(self, assembler_type):
        self._assembler._assembler_type = assembler_type
        self._assembler._load_geometry(self._geom_file, self._quad_positions)

    def _create_catalog(self, src_name, key_name):
        catalog = SourceCatalog()
        src = f'{src_name} {key_name}'
        catalog.add_item(SourceItem('LPD', src_name, [], key_name, slice(None, None), None))
        return src, catalog

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
                    'tid': 10001,
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

        with pytest.raises(AssemblingError, match='Expected module shape'):
            data = {
                'catalog': catalog,
                'meta': {
                    src: {
                        'tid': 10001,
                        'source_type': DataSource.BRIDGE,
                    }
                },
                'raw': {
                    src: np.ones((16, 100, 100, 4), dtype=_IMAGE_DTYPE)
                },
            }
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

    @pytest.mark.parametrize("assembler_type", [GeomAssembler.EXTRA_GEOM, GeomAssembler.OWN])
    def testOutArray(self, assembler_type):
        self._load_geometry(assembler_type)

        key_name = 'image.data'
        src, catalog = self._create_catalog('FXE_DET_LPD1M-1/CAL/APPEND_CORRECTED', key_name)

        data = {
            'catalog': catalog,
            'meta': {
                src: {
                    'tid': 10001,
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
        self._assembler._load_geometry(self._geom_file, tuple(quad_positions))
        data['raw'][src] = np.ones((16, 256, 256, 10), dtype=_IMAGE_DTYPE)
        self._assembler.process(data)
        assembled_shape_old = assembled_shape
        assembled_shape = data['assembled']['data'].shape
        assert assembled_shape_old != assembled_shape
        assert self._assembler._out_array.shape == assembled_shape
        # change the geometry back
        self._assembler._load_geometry(self._geom_file, self._quad_positions)

    @pytest.mark.parametrize("assembler_type", [GeomAssembler.EXTRA_GEOM, GeomAssembler.OWN])
    def testAssembleDtype(self, assembler_type):
        self._load_geometry(assembler_type)

        key_name = 'image.data'
        src, catalog = self._create_catalog('FXE_DET_LPD1M-1/CAL/APPEND_CORRECTED', key_name)

        data = {
            'catalog': catalog,
            'meta': {
                src: {
                    'tid': 10001,
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
        self._load_geometry(GeomAssembler.EXTRA_GEOM)

    def _load_geometry(self, assembler_type):
        self._assembler._assembler_type = assembler_type
        self._assembler._load_geometry(self._geom_file, self._quad_positions)

    def _create_catalog(self, src_name, key_name):
        catalog = SourceCatalog()
        src = f'{src_name} {key_name}'
        catalog.add_item(SourceItem('DSSC', src_name, [], key_name, slice(None, None), None))
        return src, catalog

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
                    'tid': 10001,
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

        with pytest.raises(AssemblingError, match='Expected module shape'):
            data = {
                'catalog': catalog,
                'meta': {
                    src: {
                        'tid': 10001,
                        'source_type': DataSource.BRIDGE,
                    }
                },
                'raw': {
                    src: np.ones((16, 100, 100, 4), dtype=_IMAGE_DTYPE),
                },
            }
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

    def testOutArray(self):
        key_name = 'image.data'
        src, catalog = self._create_catalog('SCS_CDIDET_DSSC/CAL/APPEND_CORRECTED', key_name)

        data = {
            'catalog': catalog,
            'meta': {
                src: {
                    'tid': 10001,
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

    def testAssembleDtype(self):
        key_name = 'image.data'
        src, catalog = self._create_catalog('SCS_CDIDET_DSSC/CAL/APPEND_CORRECTED', key_name)

        data = {
            'catalog': catalog,
            'meta': {
                src: {
                    'tid': 10001,
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
        self._assembler.process(data)
        assembled_dtype = data['assembled']['data'].dtype
        assert _IMAGE_DTYPE == assembled_dtype


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
                    'tid': 10001,
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

        with self.assertRaises(NotImplementedError):
            data['raw'][src] = np.ones((2, 512, 1024))
            self._assembler.process(data)

    def testAssembleBridge(self):
        key_name = 'data.adc'
        src, catalog = self._create_catalog('FXE_XAD_JF1M/DET/RECEIVER-1:display', key_name)

        data = {
            'catalog': catalog,
            'meta': {
                src: {
                    'tid': 10001,
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
        with self.assertRaises(NotImplementedError):
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

    def _create_catalog(self, src_name, key_name):
        catalog = SourceCatalog()
        src = f'{src_name} {key_name}'
        catalog.add_item(SourceItem('JungFrauPR', src_name, [], key_name, None, None))
        return src, catalog

    def testAssembleBridge(self):
        key_name = 'data.adc'
        src, catalog = self._create_catalog('FXE_XAD_JF1M/DET/RECEIVER-1:display', key_name)

        data = {
            'catalog': catalog,
            'meta': {
                src: {
                    'tid': 10001,
                    'source_type': DataSource.BRIDGE,
                }
            },
            'raw': {
                src: np.ones((512, 1024, 1))
            },
        }

        self._assembler.process(data)
        self.assertIsNone(data['raw'][src])

        assembled = data['assembled']['data']
        self.assertTupleEqual(assembled.shape, (1, 512, 1024))

        # test multi-frame (16 here), single module JungFrau received
        # in the old array shape.
        data['raw'][src] = np.ones((512, 1024, 16))

        temp = self._assembler._get_modules_bridge(data['raw'], src)
        self.assertTupleEqual(temp.shape, (16, 1, 512, 1024))

        self._assembler.process(data)
        assembled = data['assembled']['data']
        self.assertTupleEqual(assembled.shape, (16, 512, 1024))

        # test multi-frame (16 here), two-modules JungFrau in new array shape
        data['raw'][src] = np.ones((2, 512, 1024, 16))

        temp = self._assembler._get_modules_bridge(data['raw'], src)
        self.assertTupleEqual(temp.shape, (16, 2, 512, 1024))

        self._assembler.process(data)
        assembled = data['assembled']['data']
        self.assertTupleEqual(assembled.shape, (16, 1024, 1024))

        # test multi-frame, three-modules JungFrau
        with self.assertRaisesRegex(AssemblingError, 'Expected 1 or 2 module'):
            data['raw'][src] = np.ones((3, 512, 1024, 16))
            self._assembler.process(data)

        with self.assertRaisesRegex(AssemblingError, 'Expected module shape'):
            data['raw'][src] = np.ones((100, 100, 1))
            self._assembler.process(data)

    def testAssembleFile(self):
        pass


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
                    'tid': 10001,
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
                    'tid': 10001,
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
                    'tid': 10001,
                    'source_type': DataSource.BRIDGE,
                }
            },
            'raw': {
                src: np.ones((1024, 1024))
            },
        }

        self._assembler.process(data)
        self.assertIsNone(data['raw'][src])
