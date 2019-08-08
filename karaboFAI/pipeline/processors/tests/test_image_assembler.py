"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

Unittest for ImageAssembler.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import unittest
from unittest.mock import patch
import os
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
        # FIXME: we need a geometry for this test
        # self._assembler.load_geometry(self._geom_file, self._quad_positions)

    def testAssembleFile(self):
        pass

    @patch.dict(config._data, {"NUMBER_OF_MODULES": 16,
                               "MODULE_SHAPE": [512, 128]})
    def testAssembleBridge(self):
        self._assembler._source_type = DataSource.BRIDGE
        src_name = 'detector_data'
        key_name = 'image.data'
        self._assembler._detector_source_name = src_name

        with self.assertRaisesRegex(AssemblingError, 'Expected module shape'):
            data = {'raw': {
                src_name: {
                    key_name: np.ones((4, 16, 100, 100), dtype=np.float32)}}}
            self._assembler.process(data)

        with self.assertRaisesRegex(AssemblingError, 'modules, but'):
            data = {'raw': {
                src_name: {
                    key_name: np.ones((4, 12, 512, 128), dtype=np.float32)}}}
            self._assembler.process(data)

        with self.assertRaisesRegex(AssemblingError, 'Number of memory cells'):
            data = {'raw': {
                src_name: {
                    key_name: np.ones((0, 16, 512, 128), dtype=np.float32)}}}
            self._assembler.process(data)

        # TODO: add a test similar to the LPD one


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
        self._assembler._source_type = DataSource.FILE
        key_name = 'image.data'

        data = {'raw': {
            'FXE_DET_LPD1M-1/DET/11CH0:xtdf':
                {key_name: np.ones((4, 256, 256), dtype=np.float32)},
            'FXE_DET_LPD1M-1/DET/7CH0:xtdf':
                {key_name: np.ones((4, 256, 256), dtype=np.float32)},
            'FXE_DET_LPD1M-1/DET/8CH0:xtdf':
                {key_name: np.ones((4, 256, 256), dtype=np.float32)},
            'FXE_DET_LPD1M-1/DET/3CH0:xtdf':
                {key_name: np.ones((4, 256, 256), dtype=np.float32)},
        }}
        self._assembler.process(data)
        # test the module keys have been deleted
        self.assertFalse(bool(data['raw']))

        self.assertEqual(3, data['assembled'].ndim)
        assembled_shape = data['assembled'].shape
        self.assertEqual(4, assembled_shape[0])
        self.assertGreater(assembled_shape[1], 1024)
        self.assertGreater(assembled_shape[2], 1024)

    @patch.dict(config._data, {"NUMBER_OF_MODULES": 16,
                               "MODULE_SHAPE": [256, 256]})
    def testAssembleBridge(self):
        self._assembler._source_type = DataSource.BRIDGE
        src_name = 'lpd_modules'
        key_name = 'image.data'
        self._assembler._detector_source_name = src_name

        with self.assertRaisesRegex(AssemblingError, 'Expected module shape'):
            data = {'raw': {
                src_name: {
                    key_name: np.ones((16, 100, 100, 4), dtype=np.float32)}}}
            self._assembler.process(data)

        with self.assertRaisesRegex(AssemblingError, 'modules, but'):
            data = {'raw': {
                src_name: {
                    key_name: np.ones((15, 256, 256, 4), dtype=np.float32)}}}
            self._assembler.process(data)

        with self.assertRaisesRegex(AssemblingError, 'Number of memory cells'):
            data = {'raw': {
                src_name: {
                    key_name: np.ones((16, 256, 256, 0), dtype=np.float32)}}}
            self._assembler.process(data)

        # (modules, x, y, memory cells)
        data = {'raw': {
            src_name: {
                key_name: np.ones((16, 256, 256, 4), dtype=np.float32)}}}
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
        self._assembler._source_type = DataSource.BRIDGE
        src_name = 'lpd_modules'
        key_name = 'image.data'
        self._assembler._detector_source_name = src_name

        data = {'raw': {
            src_name: {
                key_name: np.ones((16, 256, 256, 4), dtype=np.float32)}}}
        self._assembler.process(data)
        assembled_shape = data['assembled'].shape

        np.testing.assert_array_equal(self._assembler._out_array.shape,
            assembled_shape)
        np.testing.assert_array_equal(
            self._assembler._extra_shape, assembled_shape[0])
        self.assertEqual(np.float32, self._assembler._out_array.dtype)

        # Test output array shape change on the fly
        data = {'raw': {
            src_name: {
                key_name: np.ones((16, 256, 256, 10), dtype=np.float32)}}}
        self._assembler.process(data)
        assembled_shape = data['assembled'].shape
        np.testing.assert_array_equal(self._assembler._out_array.shape,
            assembled_shape)
        np.testing.assert_array_equal(self._assembler._extra_shape,
                                      assembled_shape[0])

    def testAssembleDtype(self):
        self._assembler._source_type = DataSource.BRIDGE
        src_name = 'lpd_modules'
        key_name = 'image.data'
        self._assembler._detector_source_name = src_name
        # dtype conversion float64 -> float32 throws TypeError (karabo_data)
        with self.assertRaises(TypeError):
            data = {'raw': {
                src_name: {
                    key_name: np.ones((16, 256, 256, 4), dtype=np.float64)}}}
            self._assembler.process(data)

        data = {'raw': {
            src_name: {
                key_name: np.ones((16, 256, 256, 4), dtype=np.int16)}}}
        self._assembler.process(data)
        assembled_dtype = data["assembled"].dtype
        self.assertEqual(config["IMAGE_DTYPE"], assembled_dtype)


class TestJungfrauAssembler(unittest.TestCase):
    def setUp(self):
        self._assembler = ImageAssemblerFactory.create("JungFrau")

    @patch.dict(config._data, {"NUMBER_OF_MODULES": 1,
                               "MODULE_SHAPE": [512, 1024]})
    def testAssembleFile(self):
        self._assembler._source_type = DataSource.FILE
        src_name = 'jungfrau_modules'
        key_name = 'data.adc'
        self._assembler._detector_source_name = src_name

        data = {'raw': {src_name: {key_name: np.ones((1, 512, 1024))}}}
        self._assembler.process(data)
        # test the module keys have been deleted
        self.assertFalse(bool(data['raw']))

        with self.assertRaisesRegex(AssemblingError, 'Expected module shape'):
            data = {'raw': {src_name: {key_name: np.ones((1, 100, 100))}}}
            self._assembler.process(data)

        with self.assertRaises(NotImplementedError):
            data = {'raw': {src_name: {key_name: np.ones((2, 512, 1024))}}}
            self._assembler.process(data)

    @patch.dict(config._data, {"NUMBER_OF_MODULES": 1,
                               "MODULE_SHAPE": [512, 1024]})
    def testAssembleBridge(self):
        self._assembler._source_type = DataSource.BRIDGE
        src_name = 'jungfrau_modules'
        key_name = 'data.adc'
        self._assembler._detector_source_name = src_name
        data = {'raw': {src_name: {key_name: np.ones((512, 1024, 1))}}}
        self._assembler.process(data)
        # test the module keys have been deleted
        self.assertFalse(bool(data['raw']))

        with self.assertRaisesRegex(AssemblingError, 'Expected module shape'):
            data = {'raw': {src_name: {key_name: np.ones((100, 100, 1))}}}
            self._assembler.process(data)

        with self.assertRaises(NotImplementedError):
            data = {'raw': {src_name: {key_name: np.ones((512, 1024, 2))}}}
            self._assembler.process(data)


class TestFastccdAssembler(unittest.TestCase):
    def setUp(self):
        self._assembler = ImageAssemblerFactory.create("FastCCD")

    @patch.dict(config._data, {"NUMBER_OF_MODULES": 1,
                               "MODULE_SHAPE": [1934, 960]})
    def testAssembleFile(self):
        self._assembler._source_type = DataSource.FILE
        src_name = 'fastccd_module'
        key_name = 'data.image.pixels'
        self._assembler._detector_source_name = src_name

        data = {'raw': {src_name: {key_name: np.ones((1934, 960))}}}
        self._assembler.process(data)
        # test the module keys have been deleted
        self.assertFalse(bool(data['raw']))

        with self.assertRaisesRegex(AssemblingError, 'Expected module shape'):
            data = {'raw': {src_name: {key_name: np.ones((100, 100))}}}
            self._assembler.process(data)

    @patch.dict(config._data, {"NUMBER_OF_MODULES": 1,
                               "MODULE_SHAPE": [1934, 960]})
    def testAssembleBridge(self):
        self._assembler._source_type = DataSource.BRIDGE
        src_name = 'fastccd_module'
        key_name = 'data.image'
        self._assembler._detector_source_name = src_name
        data = {'raw': {src_name: {key_name: np.ones((1934, 960, 1))}}}
        self._assembler.process(data)
        # test the module keys have been deleted
        self.assertFalse(bool(data['raw']))

        with self.assertRaisesRegex(AssemblingError, 'Expected module shape'):
            data = {'raw': {src_name: {key_name: np.ones((100, 100, 1))}}}
            self._assembler.process(data)


class TestBaslerCameraAssembler(unittest.TestCase):
    def setUp(self):
        self._assembler = ImageAssemblerFactory.create("BaslerCamera")

    @patch.dict(config._data, {"NUMBER_OF_MODULES": 1,
                               "MODULE_SHAPE": [-1, -1]})
    def testAssembleBridge(self):
        self._assembler._source_type = DataSource.BRIDGE
        src_name = 'baslercamera_module'
        key_name = 'data.image.data'
        self._assembler._detector_source_name = src_name
        data = {'raw': {src_name: {key_name: np.ones((1024, 1024))}}}
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
        self._assembler._source_type = DataSource.FILE
        key_name = 'image.data'

        data = {'raw': {
            'SCS_DET_DSSC1M-1/DET/11CH0:xtdf':
                {key_name: np.ones((4, 128, 512), dtype=np.float32)},
            'SCS_DET_DSSC1M-1/DET/7CH0:xtdf':
                {key_name: np.ones((4, 128, 512), dtype=np.float32)},
            'SCS_DET_DSSC1M-1/DET/8CH0:xtdf':
                {key_name: np.ones((4, 128, 512), dtype=np.float32)},
            'SCS_DET_DSSC1M-1/DET/3CH0:xtdf':
                {key_name: np.ones((4, 128, 512), dtype=np.float32)},
        }}
        self._assembler.process(data)
        # test the module keys have been deleted
        self.assertFalse(bool(data['raw']))

        self.assertEqual(3, data['assembled'].ndim)
        assembled_shape = data['assembled'].shape
        self.assertEqual(4, assembled_shape[0])
        self.assertGreater(assembled_shape[1], 1024)
        self.assertGreater(assembled_shape[2], 1024)

    def testAssembleFileRaw(self):
        self._assembler._source_type = DataSource.FILE
        key_name = 'image.data'

        data = {'raw': {
            'SCS_DET_DSSC1M-1/DET/11CH0:xtdf':
                {key_name: np.ones((4, 1, 128, 512), dtype=np.uint16)},
        }}
        self._assembler.process(data)
        self.assertFalse(bool(data['raw']))

        self.assertEqual(3, data['assembled'].ndim)
        assembled_shape = data['assembled'].shape
        self.assertEqual(4, assembled_shape[0])
        self.assertGreater(assembled_shape[1], 1024)
        self.assertGreater(assembled_shape[2], 1024)

        # test invalid data type
        data = {'raw': {
            'SCS_DET_DSSC1M-1/DET/11CH0:xtdf':
                {key_name: np.ones((4, 1, 128, 512), dtype=np.float64)},
        }}
        with self.assertRaises(AssemblingError):
            self._assembler.process(data)

        data = {'raw': {
            'SCS_DET_DSSC1M-1/DET/11CH0:xtdf':
                {key_name: np.ones((4, 1, 128, 512), dtype=np.int64)},
        }}
        with self.assertRaises(AssemblingError):
            self._assembler.process(data)

    @patch.dict(config._data, {"NUMBER_OF_MODULES": 16,
                               "MODULE_SHAPE": [128, 512]})
    def testAssembleBridge(self):
        self._assembler._source_type = DataSource.BRIDGE
        src_name = 'dssc_modules'
        key_name = 'image.data'
        self._assembler._detector_source_name = src_name

        with self.assertRaisesRegex(AssemblingError, 'Expected module shape'):
            data = {'raw': {
                src_name: {
                    key_name: np.ones((16, 100, 100, 4), dtype=np.float32)}}}
            self._assembler.process(data)

        with self.assertRaisesRegex(AssemblingError, 'modules, but'):
            data = {'raw': {
                src_name: {
                    key_name: np.ones((15, 512, 128, 4), dtype=np.float32)}}}
            self._assembler.process(data)

        with self.assertRaisesRegex(AssemblingError, 'Number of memory cells'):
            data = {'raw': {
                src_name: {
                    key_name: np.ones((16, 512, 128, 0), dtype=np.float32)}}}
            self._assembler.process(data)

        # (modules, x, y, memory cells)
        data = {'raw': {
            src_name: {
                key_name: np.ones((16, 512, 128, 4), dtype=np.float32)}}}
        self._assembler.process(data)
        # test the module keys have been deleted
        self.assertFalse(bool(data['raw']))

        self.assertEqual(3, data['assembled'].ndim)
        assembled_shape = data['assembled'].shape
        self.assertEqual(4, assembled_shape[0])
        self.assertGreater(assembled_shape[1], 1024)
        self.assertGreater(assembled_shape[2], 1024)

    def testOutArray(self):
        self._assembler._source_type = DataSource.BRIDGE
        src_name = 'dssc_modules'
        key_name = 'image.data'
        self._assembler._detector_source_name = src_name

        data = {'raw': {
            src_name: {
                key_name: np.ones((16, 512, 128, 4), dtype=np.float32)}}}
        self._assembler.process(data)
        assembled_shape = data['assembled'].shape

        np.testing.assert_array_equal(self._assembler._out_array.shape,
            assembled_shape)
        np.testing.assert_array_equal(self._assembler._extra_shape,
                                      assembled_shape[0])

        # Test output array shape change on the fly
        data = {'raw': {
            src_name: {
                key_name: np.ones((16, 512, 128, 10), dtype=np.float32)}}}
        self._assembler.process(data)
        assembled_shape = data['assembled'].shape
        np.testing.assert_array_equal(self._assembler._out_array.shape,
            assembled_shape)
        np.testing.assert_array_equal(self._assembler._extra_shape,
                                      assembled_shape[0])

    def testAssembleDtype(self):
        self._assembler._source_type = DataSource.BRIDGE
        src_name = 'dssc_modules'
        key_name = 'image.data'
        self._assembler._detector_source_name = src_name
        # dtype conversion float64 -> float32 throws TypeError (karabo_data)
        with self.assertRaises(TypeError):
            data = {'raw': {
                src_name: {
                    key_name: np.ones((16, 512, 128, 4), dtype=np.float64)}}}
            self._assembler.process(data)

        data = {'raw': {
            src_name: {
                key_name: np.ones((16, 512, 128, 4), dtype=np.int16)}}}
        self._assembler.process(data)
        assembled_dtype = data["assembled"].dtype
        self.assertEqual(config["IMAGE_DTYPE"], assembled_dtype)
