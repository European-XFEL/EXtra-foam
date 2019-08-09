import os
import unittest
from unittest.mock import MagicMock
from threading import Thread
import tempfile
import time

from karaboFAI.config import _Config, config, ConfigWrapper, AnalysisType
from karaboFAI.gui import mkQApp
from karaboFAI.logger import logger
from karaboFAI.pipeline.processors.base_processor import (
    _BaseProcessor, ProcessingError, SharedProperty,
)
from karaboFAI.processes import wait_until_redis_shutdown
from karaboFAI.services import FAI

app = mkQApp()

logger.setLevel("CRITICAL")


class _DummyProcessor(_BaseProcessor):
    def __init__(self):
        super().__init__()
        self.analysis_type = AnalysisType.UNDEFINED

    def update(self):
        pass

    def process(self, processed):
        pass


class TestRedisParserMixin(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._proc = _DummyProcessor()

    def testStr2Tuple(self):
        self.assertTupleEqual((1.0, 2.0), self._proc.str2tuple('(1, 2)'))

    def testStr2List(self):
        self.assertListEqual([1.0, 2.0], self._proc.str2list('[1, 2]'))
        self.assertListEqual([1], self._proc.str2list('[1]'))
        self.assertListEqual([], self._proc.str2list('[]'))

    def testStr2Slice(self):
        self.assertEqual(slice(None, 2), self._proc.str2slice('[None, 2]'))
        self.assertEqual(slice(1, 10, 2), self._proc.str2slice('[1, 10, 2]'))


class TestBaseProcessor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # do not use the config file in the current computer
        _Config._filename = os.path.join(tempfile.mkdtemp(), "config.json")
        ConfigWrapper()   # ensure file
        config.load('LPD')

        cls.fai = FAI().init()
        cls.scheduler = cls.fai.scheduler

    @classmethod
    def tearDownClass(cls):
        cls.fai.terminate()

        wait_until_redis_shutdown()

    def testAnalysisType(self):
        self._proc1 = _DummyProcessor()
        self._proc2 = _DummyProcessor()
        self._proc3 = _DummyProcessor()

        with self.assertRaises(ProcessingError):
            self._proc1._update_analysis(1)

        self._proc1._update_analysis(AnalysisType.UNDEFINED)
        self._check_has_no_analysis(AnalysisType.UNDEFINED)

        # set new analysis type for proc2
        self._proc2._update_analysis(AnalysisType.PROJ_ROI1_SUB_ROI2)
        self.assertEqual(AnalysisType.UNDEFINED, self._proc1.analysis_type)
        self.assertEqual(AnalysisType.PROJ_ROI1_SUB_ROI2, self._proc2.analysis_type)
        self._check_has_analysis(AnalysisType.PROJ_ROI1_SUB_ROI2)

        # set new analysis type for proc1
        self._proc1._update_analysis(AnalysisType.AZIMUTHAL_INTEG)
        self._check_has_analysis(AnalysisType.AZIMUTHAL_INTEG)
        self.assertEqual(AnalysisType.PROJ_ROI1_SUB_ROI2, self._proc2.analysis_type)
        self.assertEqual(AnalysisType.AZIMUTHAL_INTEG, self._proc1.analysis_type)

        # unset analysis type for proc1
        self._proc1._update_analysis(AnalysisType.UNDEFINED)
        self.assertEqual(AnalysisType.UNDEFINED, self._proc1.analysis_type)
        self.assertEqual(AnalysisType.PROJ_ROI1_SUB_ROI2, self._proc2.analysis_type)
        self._check_has_analysis(AnalysisType.PROJ_ROI1_SUB_ROI2)
        self._check_has_no_analysis(AnalysisType.AZIMUTHAL_INTEG)

        # unset analysis type for proc2
        self._proc2._update_analysis(AnalysisType.UNDEFINED)
        self.assertEqual(AnalysisType.UNDEFINED, self._proc1.analysis_type)
        self.assertEqual(AnalysisType.UNDEFINED, self._proc2.analysis_type)
        self._check_has_no_analysis(AnalysisType.PROJ_ROI1_SUB_ROI2)
        self._check_has_no_analysis(AnalysisType.AZIMUTHAL_INTEG)

        # set same analysis type for proc1 and proc2
        self._proc1._update_analysis(AnalysisType.PROJ_ROI1_SUB_ROI2)
        self._proc2._update_analysis(AnalysisType.PROJ_ROI1_SUB_ROI2)
        self._check_has_analysis(AnalysisType.PROJ_ROI1_SUB_ROI2)
        self._proc2._update_analysis(AnalysisType.UNDEFINED)
        self._check_has_analysis(AnalysisType.PROJ_ROI1_SUB_ROI2)
        self._proc1._update_analysis(AnalysisType.UNDEFINED)
        self._check_has_no_analysis(AnalysisType.PROJ_ROI1_SUB_ROI2)

    def _check_has_analysis(self, analysis_type):
        self.assertTrue(self._proc1._has_analysis(analysis_type))
        self.assertTrue(self._proc2._has_analysis(analysis_type))
        # check with another processor
        self.assertTrue(self._proc3._has_analysis(analysis_type))

    def _check_has_no_analysis(self, analysis_type):
        self.assertFalse(self._proc1._has_analysis(analysis_type))
        self.assertFalse(self._proc2._has_analysis(analysis_type))
        self.assertFalse(self._proc3._has_analysis(analysis_type))
