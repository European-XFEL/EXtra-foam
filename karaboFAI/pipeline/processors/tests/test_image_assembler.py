"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import copy
import unittest
from unittest.mock import patch
import os
import re
import tempfile

import numpy as np

from karaboFAI.pipeline.processors.image_assembler import ImageAssemblerFactory
from karaboFAI.pipeline.exceptions import AssemblingError
from karaboFAI.config import _Config, ConfigWrapper, config, DataSource
from karaboFAI.logger import logger

logger.setLevel('CRITICAL')


class TestAgipdAssembler(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _Config._filename = os.path.join(tempfile.mkdtemp(), "config.json")
        ConfigWrapper()   # ensure file
        config.load('AGIPD')

        cls._geom_file = config["GEOMETRY_FILE"]
        cls._quad_positions = config["QUAD_POSITIONS"]

    def setUp(self):
        self._assembler = ImageAssemblerFactory.create("AGIPD")
        self._assembler.load_geometry(self._geom_file, self._quad_positions)

    def testAssembleFile(self):
        key_name = 'image.data'

        data = {
            'raw': {
                # According to which criteria are the sample module #s chosen?
                'SPB_DET_AGIPD1M-1/DET/11CH0:xtdf':
                    {key_name: np.ones((4, 512, 128), dtype=np.float32)},
                'SPB_DET_AGIPD1M-1/DET/7CH0:xtdf':
                    {key_name: np.ones((4, 512, 128), dtype=np.float32)},
                'SPB_DET_AGIPD1M-1/DET/8CH0:xtdf':
                    {key_name: np.ones((4, 512, 128), dtype=np.float32)},
                'SPB_DET_AGIPD1M-1/DET/3CH0:xtdf':
                    {key_name: np.ones((4, 512, 128), dtype=np.float32)}
                },
            'meta': {'source_type': DataSource.FILE},
        }
        with self.assertRaises(AssemblingError):
            # no source name
            self._assembler.process(data)

        self._assembler._source_name = "AGIPD modules"
        with self.assertRaises(AssemblingError):
            # source name must end with 'xtdf'
            self._assembler.process(data)

        self._assembler._source_name = "SPB_DET_AGIPD1M-1/DET/*CH0:xtdf"
        self._assembler.process(data)
        # test the module keys have been deleted
        self.assertFalse(bool(data['raw']))

        self.assertEqual(3, data['assembled'].ndim)
        assembled_shape = data['assembled'].shape
        self.assertEqual(4, assembled_shape[0])
        self.assertGreater(assembled_shape[1], 1024)
        self.assertGreater(assembled_shape[2], 1024)

    @patch.dict(config._data, {"NUMBER_OF_MODULES": 16,
                               "MODULE_SHAPE": [512, 128]})
    def _check_result(self, data):
        # test the module keys have been deleted
        self.assertFalse(bool(data['raw']))
        self.assertEqual(3, data['assembled'].ndim)
        assembled_shape = data['assembled'].shape
        self.assertEqual(4, assembled_shape[0])
        self.assertGreater(assembled_shape[1], 1024)
        self.assertGreater(assembled_shape[2], 1024)

    def testAssembleBridge(self):
        src_name = 'detector_data'
        key_name = 'image.data'
        self._assembler._source_name = src_name

        with self.assertRaisesRegex(AssemblingError, 'Expected module shape'):
            data = {
                'raw': {
                    src_name: {key_name: np.ones((4, 16, 100, 100), dtype=np.float32)}
                },
                'meta': {'source_type': DataSource.BRIDGE}
            }
            self._assembler.process(data)

        with self.assertRaisesRegex(AssemblingError, 'modules, but'):
            data = {
                'raw': {
                    src_name: {key_name: np.ones((4, 12, 512, 128), dtype=np.float32)}
                },
                'meta': {'source_type': DataSource.BRIDGE}
            }
            self._assembler.process(data)

        with self.assertRaisesRegex(AssemblingError, 'Number of memory cells'):
            data = {
                'raw': {
                    src_name: {key_name: np.ones((0, 16, 512, 128), dtype=np.float32)}
                },
                'meta': {'source_type': DataSource.BRIDGE}
            }
            self._assembler.process(data)

        expected_shape = (4, 16, 512, 128)
        # (modules, fs, ss, memory cells)
        data = {
            'raw': {
                src_name: {key_name: np.ones((16, 128, 512, 4), dtype=np.float32)}
            },
            'meta': {'source_type': DataSource.BRIDGE}
        }
        self._assembler.process(data)
        self._check_result(data)

        # (memory cells, modules, ss, fs)
        data = {
            'raw': {
                src_name: {key_name: np.ones((4, 16, 512, 128), dtype=np.float32)}
            },
            'meta': {'source_type': DataSource.BRIDGE}
        }
        self._assembler.process(data)
        self._check_result(data)


class TestLpdAssembler(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _Config._filename = os.path.join(tempfile.mkdtemp(), "config.json")
        ConfigWrapper()   # ensure file
        config.load('LPD')

        cls._geom_file = config["GEOMETRY_FILE"]
        cls._quad_positions = config["QUAD_POSITIONS"]

    def setUp(self):
        self._assembler = ImageAssemblerFactory.create("LPD")
        self._assembler.load_geometry(self._geom_file, self._quad_positions)

    def testAssembleFile(self):
        key_name = 'image.data'

        data = {
            'raw': {
                'FXE_DET_LPD1M-1/DET/11CH0:xtdf':
                    {key_name: np.ones((4, 256, 256), dtype=np.float32)},
                'FXE_DET_LPD1M-1/DET/7CH0:xtdf':
                    {key_name: np.ones((4, 256, 256), dtype=np.float32)},
                'FXE_DET_LPD1M-1/DET/8CH0:xtdf':
                    {key_name: np.ones((4, 256, 256), dtype=np.float32)},
                'FXE_DET_LPD1M-1/DET/3CH0:xtdf':
                    {key_name: np.ones((4, 256, 256), dtype=np.float32)},
                # Non detector data source included in streaming from files.
                # To test stack_detector_data
                'XGM':{"data.intensitySa1TD": np.ones(60)}
            },
            'meta': {'source_type': DataSource.FILE}
        }

        with self.assertRaises(AssemblingError):
            # no source name
            self._assembler.process(data)

        self._assembler._source_name = "LPD modules"
        with self.assertRaises(AssemblingError):
            # source name must end with 'xtdf'
            self._assembler.process(data)

        self._assembler._source_name = "FXE_DET_LPD1M-1/DET/*CH0:xtdf"
        # Only LPD modules related keys
        module_keys = [key for key in data['raw'].keys()
                       if re.match(r"(.+)/DET/(.+):(.+)", key)]
        self._assembler.process(data)
        # test the module keys have been deleted
        self.assertFalse(any([key in data['raw'] for key in module_keys]))

        self.assertEqual(3, data['assembled'].ndim)
        assembled_shape = data['assembled'].shape
        self.assertEqual(4, assembled_shape[0])
        self.assertGreater(assembled_shape[1], 1024)
        self.assertGreater(assembled_shape[2], 1024)

    @patch.dict(config._data, {"NUMBER_OF_MODULES": 16,
                               "MODULE_SHAPE": [256, 256]})
    def testAssembleBridge(self):
        src_name = 'lpd_modules'
        key_name = 'image.data'
        self._assembler._source_name = src_name

        with self.assertRaisesRegex(AssemblingError, 'Expected module shape'):
            data = {
                'raw': {
                    src_name: {key_name: np.ones((16, 100, 100, 4), dtype=np.float32)}
                },
                'meta': {'source_type': DataSource.BRIDGE}
            }
            self._assembler.process(data)

        with self.assertRaisesRegex(AssemblingError, 'modules, but'):
            data = {
                'raw': {
                    src_name: {key_name: np.ones((15, 256, 256, 4), dtype=np.float32)}
                },
                'meta': {'source_type': DataSource.BRIDGE}
            }
            self._assembler.process(data)

        with self.assertRaisesRegex(AssemblingError, 'Number of memory cells'):
            data = {
                'raw': {
                    src_name: {key_name: np.ones((16, 256, 256, 0), dtype=np.float32)}
                },
                'meta': {'source_type': DataSource.BRIDGE}
            }
            self._assembler.process(data)

        # (modules, x, y, memory cells)
        data = {
            'raw': {
                src_name: {key_name: np.ones((16, 256, 256, 4), dtype=np.float32)}
            },
            'meta': {'source_type': DataSource.BRIDGE}
        }
        self._assembler.process(data)
        # test the module keys have been deleted
        self.assertFalse(bool(data['raw']))
        assembled = data["assembled"]
        self.assertEqual(3, data['assembled'].ndim)
        assembled_shape = data['assembled'].shape
        self.assertEqual(4, assembled_shape[0])
        self.assertGreater(assembled_shape[1], 1024)
        self.assertGreater(assembled_shape[2], 1024)

    def testOutArray(self):
        src_name = 'lpd_modules'
        key_name = 'image.data'
        self._assembler._source_name = src_name

        data = {
            'raw': {
                src_name: {key_name: np.ones((16, 256, 256, 4), dtype=np.float32)}
            },
            'meta': {'source_type': DataSource.BRIDGE}
        }
        self._assembler.process(data)
        assembled_shape = data['assembled'].shape

        self.assertTupleEqual(self._assembler._out_array.shape, assembled_shape)
        self.assertTupleEqual(self._assembler._n_images, (assembled_shape[0],))
        self.assertEqual(config["IMAGE_DTYPE"], self._assembler._out_array.dtype)

        # Test number of pulses change on the fly
        data = {
            'raw': {
                src_name: {key_name: np.ones((16, 256, 256, 10), dtype=np.float32)}
            },
            'meta': {'source_type': DataSource.BRIDGE}
        }
        self._assembler.process(data)
        assembled_shape = data['assembled'].shape
        self.assertTupleEqual(self._assembler._out_array.shape, assembled_shape)
        self.assertTupleEqual(self._assembler._n_images, (assembled_shape[0],))

        # test quad_positions (geometry) change on the fly
        quad_positions = copy.deepcopy(self._quad_positions)
        quad_positions[0][1] += 2  # modify the quad positions
        quad_positions[3][0] -= 4
        self._assembler.load_geometry(self._geom_file, quad_positions)
        data = {
            'raw': {
                src_name: {key_name: np.ones((16, 256, 256, 10), dtype=np.float32)}
            },
            'meta': {'source_type': DataSource.BRIDGE}
        }
        self._assembler.process(data)
        assembled_shape_old = assembled_shape
        assembled_shape = data['assembled'].shape
        self.assertNotEqual(assembled_shape_old, assembled_shape)
        self.assertTupleEqual(self._assembler._out_array.shape, assembled_shape)
        self.assertTupleEqual(self._assembler._n_images, (assembled_shape[0],))
        # change the geometry back
        self._assembler.load_geometry(self._geom_file, self._quad_positions)

    def testAssembleDtype(self):
        src_name = 'lpd_modules'
        key_name = 'image.data'
        self._assembler._source_name = src_name
        # dtype conversion float64 -> float32 throws TypeError (karabo_data)
        with self.assertRaises(TypeError):
            data = {
                'raw': {
                    src_name: {key_name: np.ones((16, 256, 256, 4), dtype=np.float64)}
                },
                'meta': {'source_type': DataSource.BRIDGE}
            }
            self._assembler.process(data)

        data = {
            'raw': {
                src_name: {key_name: np.ones((16, 256, 256, 4), dtype=np.int16)}
            },
            'meta': {'source_type': DataSource.BRIDGE}
        }
        self._assembler.process(data)
        assembled_dtype = data["assembled"].dtype
        self.assertEqual(config["IMAGE_DTYPE"], assembled_dtype)


class TestJungfrauAssembler(unittest.TestCase):
    def setUp(self):
        self._assembler = ImageAssemblerFactory.create("JungFrau")

    @patch.dict(config._data, {"NUMBER_OF_MODULES": 1,
                               "MODULE_SHAPE": [512, 1024]})
    def testAssembleFile(self):
        src_name = 'jungfrau_modules'
        key_name = 'data.adc'
        self._assembler._source_name = src_name

        data = {'raw': {src_name: {key_name: np.ones((1, 512, 1024))}},
                'meta': {'source_type': DataSource.FILE}}
        self._assembler.process(data)
        # test the module keys have been deleted
        self.assertFalse(bool(data['raw']))

        with self.assertRaisesRegex(AssemblingError, 'Expected module shape'):
            data = {'raw': {src_name: {key_name: np.ones((1, 100, 100))}},
                    'meta': {'source_type': DataSource.FILE}}
            self._assembler.process(data)

        with self.assertRaises(NotImplementedError):
            data = {'raw': {src_name: {key_name: np.ones((2, 512, 1024))}},
                    'meta': {'source_type': DataSource.FILE}}
            self._assembler.process(data)

    @patch.dict(config._data, {"NUMBER_OF_MODULES": 1,
                               "MODULE_SHAPE": [512, 1024]})
    def testAssembleBridge(self):
        src_name = 'jungfrau_modules'
        key_name = 'data.adc'
        self._assembler._source_name = src_name
        data = {'raw': {src_name: {key_name: np.ones((512, 1024, 1))}},
                'meta': {'source_type': DataSource.BRIDGE}}
        self._assembler.process(data)
        # test the module keys have been deleted
        self.assertFalse(bool(data['raw']))

        with self.assertRaisesRegex(AssemblingError, 'Expected module shape'):
            data = {'raw': {src_name: {key_name: np.ones((100, 100, 1))}},
                    'meta': {'source_type': DataSource.BRIDGE}}
            self._assembler.process(data)

        with self.assertRaises(NotImplementedError):
            data = {'raw': {src_name: {key_name: np.ones((512, 1024, 2))}},
                    'meta': {'source_type': DataSource.BRIDGE}}
            self._assembler.process(data)


class TestJungfrauPulseResolvedAssembler(unittest.TestCase):

    def setUp(self):
        self._assembler = ImageAssemblerFactory.create("JungFrauPR")

    @patch.dict(config._data, {"DETECTOR":"JungFrauPR",
                               "NUMBER_OF_MODULES": 2,
                               "MODULE_SHAPE": [512, 1024]})
    def testAssembleBridge(self):
        src_name = 'jungfrau_modules'
        key_name = 'data.adc'
        self._assembler._source_name = src_name
        data = {'raw': {src_name: {key_name: np.ones((512, 1024, 1))}},
                'meta': {'source_type': DataSource.BRIDGE}}
        self._assembler.process(data)
        # test the module keys have been deleted
        self.assertFalse(bool(data['raw']))

        assembled = data["assembled"]
        self.assertTupleEqual(assembled.shape, (1, 512, 1024))

        # test multi-frame (16 here), single module JungFrau received
        # in the old array shape.
        data = {'raw': {src_name: {key_name: np.ones((512, 1024, 16))}},
                'meta': {'source_type': DataSource.BRIDGE}}

        temp = self._assembler._get_modules_bridge(data['raw'], src_name)
        self.assertTupleEqual(temp.shape, (16, 1, 512, 1024))

        self._assembler.process(data)
        assembled = data["assembled"]
        self.assertTupleEqual(assembled.shape, (16, 512, 1024))

        # test multi-frame (16 here), two-modules JungFrau in new array shape
        data = {'raw': {src_name: {key_name: np.ones((2, 512, 1024, 16))}},
                'meta': {'source_type': DataSource.BRIDGE}}

        temp = self._assembler._get_modules_bridge(data['raw'], src_name)
        self.assertTupleEqual(temp.shape, (16, 2, 512, 1024))

        self._assembler.process(data)
        assembled = data["assembled"]
        self.assertTupleEqual(assembled.shape, (16, 1024, 1024))

        # test multi-frame, three-modules JungFrau
        with self.assertRaisesRegex(AssemblingError, 'Expected 1 or 2 module'):
            data = {'raw': {src_name: {key_name: np.ones((3, 512, 1024, 16))}},
                    'meta': {'source_type': DataSource.BRIDGE}}
            self._assembler.process(data)

        with self.assertRaisesRegex(AssemblingError, 'Expected module shape'):
            data = {'raw': {src_name: {key_name: np.ones((100, 100, 1))}},
                    'meta': {'source_type': DataSource.BRIDGE}}
            self._assembler.process(data)

    @patch.dict(config._data, {"DETECTOR":"JungFrauPR",
                               "NUMBER_OF_MODULES": 2,
                               "MODULE_SHAPE": [512, 1024]})
    def testAssembleFile(self):
        pass


class TestFastccdAssembler(unittest.TestCase):
    def setUp(self):
        self._assembler = ImageAssemblerFactory.create("FastCCD")

    @patch.dict(config._data, {"NUMBER_OF_MODULES": 1,
                               "MODULE_SHAPE": [1934, 960]})
    def testAssembleFile(self):
        src_name = 'fastccd_module'
        key_name = 'data.image.pixels'
        self._assembler._source_name = src_name

        data = {'raw': {src_name: {key_name: np.ones((1934, 960))}},
                'meta': {'source_type': DataSource.FILE}}
        self._assembler.process(data)
        # test the module keys have been deleted
        self.assertFalse(bool(data['raw']))

        with self.assertRaisesRegex(AssemblingError, 'Expected module shape'):
            data = {'raw': {src_name: {key_name: np.ones((100, 100))}},
                    'meta': {'source_type': DataSource.FILE}}
            self._assembler.process(data)

    @patch.dict(config._data, {"NUMBER_OF_MODULES": 1,
                               "MODULE_SHAPE": [1934, 960]})
    def testAssembleBridge(self):
        src_name = 'fastccd_module'
        key_name = 'data.image'
        self._assembler._source_name = src_name
        data = {'raw': {src_name: {key_name: np.ones((1934, 960, 1))}},
                'meta': {'source_type': DataSource.BRIDGE}}
        self._assembler.process(data)
        # test the module keys have been deleted
        self.assertFalse(bool(data['raw']))

        with self.assertRaisesRegex(AssemblingError, 'Expected module shape'):
            data = {'raw': {src_name: {key_name: np.ones((100, 100, 1))}},
                    'meta': {'source_type': DataSource.BRIDGE}}
            self._assembler.process(data)


class TestBaslerCameraAssembler(unittest.TestCase):
    def setUp(self):
        self._assembler = ImageAssemblerFactory.create("BaslerCamera")

    @patch.dict(config._data, {"NUMBER_OF_MODULES": 1,
                               "MODULE_SHAPE": [-1, -1]})
    def testAssembleBridge(self):
        src_name = 'baslercamera_module'
        key_name = 'data.image.data'
        self._assembler._source_name = src_name
        data = {'raw': {src_name: {key_name: np.ones((1024, 1024))}},
                'meta': {'source_type': DataSource.BRIDGE}}
        self._assembler.process(data)
        # test the module keys have been deleted
        self.assertFalse(bool(data['raw']))


class TestDSSCAssembler(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _Config._filename = os.path.join(tempfile.mkdtemp(), "config.json")
        ConfigWrapper()   # ensure file
        config.load('DSSC')
        config._data['REDIS_PORT'] = 6379

        cls._geom_file = config["GEOMETRY_FILE"]
        cls._quad_positions = config["QUAD_POSITIONS"]

    def setUp(self):
        self._assembler = ImageAssemblerFactory.create("DSSC")
        self._assembler.load_geometry(self._geom_file, self._quad_positions)

    def testAssembleFileCal(self):
        key_name = 'image.data'

        data = {
            'raw': {
                'SCS_DET_DSSC1M-1/DET/11CH0:xtdf':
                    {key_name: np.ones((4, 128, 512), dtype=np.float32)},
                'SCS_DET_DSSC1M-1/DET/7CH0:xtdf':
                    {key_name: np.ones((4, 128, 512), dtype=np.float32)},
                'SCS_DET_DSSC1M-1/DET/8CH0:xtdf':
                    {key_name: np.ones((4, 128, 512), dtype=np.float32)},
                'SCS_DET_DSSC1M-1/DET/3CH0:xtdf':
                    {key_name: np.ones((4, 128, 512), dtype=np.float32)},
            },
            'meta': {'source_type': DataSource.FILE}
        }

        self._assembler._source_name = "SCS_DET_DSSC1M-1/DET/*CH0:xtdf"
        self._assembler.process(data)
        # test the module keys have been deleted
        self.assertFalse(bool(data['raw']))

        self.assertEqual(3, data['assembled'].ndim)
        assembled_shape = data['assembled'].shape
        self.assertEqual(4, assembled_shape[0])
        self.assertGreater(assembled_shape[1], 1024)
        self.assertGreater(assembled_shape[2], 1024)

    def testAssembleFileRaw(self):
        key_name = 'image.data'

        data = {
            'raw': {
                'SCS_DET_DSSC1M-1/DET/11CH0:xtdf':
                    {key_name: np.ones((4, 1, 128, 512), dtype=np.uint16)},
            },
            'meta': {'source_type': DataSource.FILE}
        }

        with self.assertRaises(AssemblingError):
            # no source name
            self._assembler.process(data)

        self._assembler._source_name = "DSSC modules"
        with self.assertRaises(AssemblingError):
            # source name must end with 'xtdf'
            self._assembler.process(data)

        self._assembler._source_name = "SCS_DET_DSSC1M-1/DET/*CH0:xtdf"
        self._assembler.process(data)
        # test the module keys have been deleted
        self.assertFalse(bool(data['raw']))

        self.assertEqual(3, data['assembled'].ndim)
        assembled_shape = data['assembled'].shape
        self.assertEqual(4, assembled_shape[0])
        self.assertGreater(assembled_shape[1], 1024)
        self.assertGreater(assembled_shape[2], 1024)

        # test invalid data type
        data = {
            'raw': {
                'SCS_DET_DSSC1M-1/DET/11CH0:xtdf':
                    {key_name: np.ones((4, 1, 128, 512), dtype=np.float64)},
            },
            'meta': {'source_type': DataSource.FILE},
        }
        with self.assertRaises(AssemblingError):
            self._assembler.process(data)

        data = {
            'raw': {
                'SCS_DET_DSSC1M-1/DET/11CH0:xtdf':
                    {key_name: np.ones((4, 1, 128, 512), dtype=np.int64)},
            },
            'meta': {'source_type': DataSource.FILE},
        }
        with self.assertRaises(AssemblingError):
            self._assembler.process(data)

    @patch.dict(config._data, {"NUMBER_OF_MODULES": 16,
                               "MODULE_SHAPE": [128, 512]})
    def testAssembleBridge(self):
        src_name = 'dssc_modules'
        key_name = 'image.data'
        self._assembler._source_name = src_name

        with self.assertRaisesRegex(AssemblingError, 'Expected module shape'):
            data = {
                'raw': {
                    src_name: {key_name: np.ones((16, 100, 100, 4), dtype=np.float32)}},
                'meta': {'source_type': DataSource.BRIDGE},
            }
            self._assembler.process(data)

        with self.assertRaisesRegex(AssemblingError, 'modules, but'):
            data = {
                'raw': {
                    src_name: {key_name: np.ones((15, 512, 128, 4), dtype=np.float32)}},
                'meta': {'source_type': DataSource.BRIDGE},
            }
            self._assembler.process(data)

        with self.assertRaisesRegex(AssemblingError, 'Number of memory cells'):
            data = {
                'raw': {
                    src_name: {key_name: np.ones((16, 512, 128, 0), dtype=np.float32)}},
                'meta': {'source_type': DataSource.BRIDGE},
            }
            self._assembler.process(data)

        # (modules, x, y, memory cells)
        data = {
            'raw': {
                src_name: {key_name: np.ones((16, 512, 128, 4), dtype=np.float32)}},
            'meta': {'source_type': DataSource.BRIDGE},
        }
        self._assembler.process(data)
        # test the module keys have been deleted
        self.assertFalse(bool(data['raw']))

        self.assertEqual(3, data['assembled'].ndim)
        assembled_shape = data['assembled'].shape
        self.assertEqual(4, assembled_shape[0])
        self.assertGreater(assembled_shape[1], 1024)
        self.assertGreater(assembled_shape[2], 1024)

    def testOutArray(self):
        src_name = 'dssc_modules'
        key_name = 'image.data'
        self._assembler._source_name = src_name

        data = {
            'raw': {
                src_name: {key_name: np.ones((16, 512, 128, 4), dtype=np.float32)}},
            'meta': {'source_type': DataSource.BRIDGE},
        }
        self._assembler.process(data)
        assembled_shape = data['assembled'].shape

        np.testing.assert_array_equal(self._assembler._out_array.shape, assembled_shape)
        np.testing.assert_array_equal(self._assembler._n_images, assembled_shape[0])
        self.assertEqual(config["IMAGE_DTYPE"], self._assembler._out_array.dtype)

        # Test output array shape change on the fly
        data = {
            'raw': {
                src_name: {key_name: np.ones((16, 512, 128, 10), dtype=np.float32)}},
            'meta': {'source_type': DataSource.BRIDGE},
        }
        self._assembler.process(data)
        assembled_shape = data['assembled'].shape
        np.testing.assert_array_equal(self._assembler._out_array.shape, assembled_shape)
        np.testing.assert_array_equal(self._assembler._n_images, assembled_shape[0])

    def testAssembleDtype(self):
        self._assembler._source_type = DataSource.BRIDGE
        src_name = 'dssc_modules'
        key_name = 'image.data'
        self._assembler._source_name = src_name
        # dtype conversion float64 -> float32 throws TypeError (karabo_data)
        with self.assertRaises(TypeError):
            data = {
                'raw': {
                    src_name: {key_name: np.ones((16, 512, 128, 4), dtype=np.float64)}},
                'meta': {'source_type': DataSource.BRIDGE},
            }
            self._assembler.process(data)

        data = {
            'raw': {
                src_name: {key_name: np.ones((16, 512, 128, 4), dtype=np.int16)}},
            'meta': {'source_type': DataSource.BRIDGE},
        }
        self._assembler.process(data)
        assembled_dtype = data["assembled"].dtype
        self.assertEqual(config["IMAGE_DTYPE"], assembled_dtype)
