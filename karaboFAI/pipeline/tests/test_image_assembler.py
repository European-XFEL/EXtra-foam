import unittest
from unittest.mock import patch
import numpy as np

from karaboFAI.pipeline.image_assembler import ImageAssemblerFactory
from karaboFAI.pipeline.exceptions import AssemblingError
from karaboFAI.config import config, DataSource


class TestAgipdAssembler(unittest.TestCase):
    def setUp(self):
        self._assembler = ImageAssemblerFactory.create("AGIPD")

    def testAssembleFile(self):
        pass

    @patch.dict(config._data, {"NUMBER_OF_MODULES": 16,
                               "MODULE_SHAPE": [512, 128]})
    def testAssembleBridge(self):
        self._assembler._source_type = DataSource.BRIDGE
        src_name = 'detector_data'
        key_name = 'image.data'
        self._assembler._detector_source_name = src_name
        data = {src_name: {key_name: np.ones((4, 16, 512, 128))}}
        self._assembler.assemble(data)

        with self.assertRaisesRegex(AssemblingError, 'Expected module shape'):
            data = {src_name: {key_name: np.ones((4, 16, 100, 100))}}
            self._assembler.assemble(data)

        with self.assertRaisesRegex(AssemblingError, 'modules, but'):
            data = {src_name: {key_name: np.ones((4, 12, 512, 128))}}
            self._assembler.assemble(data)

        with self.assertRaisesRegex(AssemblingError, 'Number of memory cells'):
            data = {src_name: {key_name: np.ones((0, 16, 512, 128))}}
            self._assembler.assemble(data)


class TestLpdAssembler(unittest.TestCase):

    def setUp(self):
        self._assembler = ImageAssemblerFactory.create("LPD")

    def testAssembleFile(self):
        pass

    @patch.dict(config._data, {"NUMBER_OF_MODULES": 16,
                               "MODULE_SHAPE": [256, 256]})
    def testAssembleBridge(self):
        self._assembler._source_type = DataSource.BRIDGE
        src_name = 'detector_data'
        key_name = 'image.data'
        self._assembler._detector_source_name = src_name
        data = {src_name: {key_name: np.ones((16, 256, 256, 4))}}
        self._assembler.assemble(data)

        with self.assertRaisesRegex(AssemblingError, 'Expected module shape'):
            data = {src_name: {key_name: np.ones((16, 100, 100, 4))}}
            self._assembler.assemble(data)

        with self.assertRaisesRegex(AssemblingError, 'modules, but'):
            data = {src_name: {key_name: np.ones((15, 256, 256, 4))}}
            self._assembler.assemble(data)

        with self.assertRaisesRegex(AssemblingError, 'Number of memory cells'):
            data = {src_name: {key_name: np.ones((16, 256, 256, 0))}}
            self._assembler.assemble(data)


class TestJungfrauAssembler(unittest.TestCase):
    def setUp(self):
        self._assembler = ImageAssemblerFactory.create("JungFrau")

    @patch.dict(config._data, {"NUMBER_OF_MODULES": 1,
                               "MODULE_SHAPE": [512, 1024]})
    def testAssembleFile(self):
        self._assembler._source_type = DataSource.FILE
        src_name = 'detector_data'
        key_name = 'data.adc'
        self._assembler._detector_source_name = src_name
        data = {src_name: {key_name: np.ones((1, 512, 1024))}}
        self._assembler.assemble(data)

        with self.assertRaisesRegex(AssemblingError, 'Expected module shape'):
            data = {src_name: {key_name: np.ones((1, 100, 100))}}
            self._assembler.assemble(data)

        with self.assertRaises(NotImplementedError):
            data = {src_name: {key_name: np.ones((2, 512, 1024))}}
            self._assembler.assemble(data)

    @patch.dict(config._data, {"NUMBER_OF_MODULES": 1,
                               "MODULE_SHAPE": [512, 1024]})
    def testAssembleBridge(self):
        self._assembler._source_type = DataSource.BRIDGE
        src_name = 'detector_data'
        key_name = 'data.adc'
        self._assembler._detector_source_name = src_name
        data = {src_name: {key_name: np.ones((512, 1024, 1))}}
        self._assembler.assemble(data)

        with self.assertRaisesRegex(AssemblingError, 'Expected module shape'):
            data = {src_name: {key_name: np.ones((100, 100, 1))}}
            self._assembler.assemble(data)

        with self.assertRaises(NotImplementedError):
            data = {src_name: {key_name: np.ones((512, 1024, 2))}}
            self._assembler.assemble(data)


class TestFastccdAssembler(unittest.TestCase):
    def setUp(self):
        self._assembler = ImageAssemblerFactory.create("FastCCD")

    @patch.dict(config._data, {"NUMBER_OF_MODULES": 1,
                               "MODULE_SHAPE": [1934, 960]})
    def testAssembleFile(self):
        self._assembler._source_type = DataSource.FILE
        src_name = 'detector_data'
        key_name = 'data.image.pixels'
        self._assembler._detector_source_name = src_name
        data = {src_name: {key_name: np.ones((1934, 960))}}
        self._assembler.assemble(data)

        with self.assertRaisesRegex(AssemblingError, 'Expected module shape'):
            data = {src_name: {key_name: np.ones((100, 100))}}
            self._assembler.assemble(data)

    @patch.dict(config._data, {"NUMBER_OF_MODULES": 1,
                               "MODULE_SHAPE": [1934, 960]})
    def testAssembleBridge(self):
        self._assembler._source_type = DataSource.BRIDGE
        src_name = 'detector_data'
        key_name = 'data.image'
        self._assembler._detector_source_name = src_name
        data = {src_name: {key_name: np.ones((1934, 960, 1))}}
        self._assembler.assemble(data)

        with self.assertRaisesRegex(AssemblingError, 'Expected module shape'):
            data = {src_name: {key_name: np.ones((100, 100, 1))}}
            self._assembler.assemble(data)
