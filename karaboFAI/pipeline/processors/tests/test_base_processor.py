import os
import unittest
from unittest.mock import MagicMock
from threading import Thread
import tempfile
import time

from karaboFAI.config import _Config, ConfigWrapper, AnalysisType
from karaboFAI.logger import logger
from karaboFAI.services import FAI
from karaboFAI.pipeline.data_model import ProcessedData
from karaboFAI.pipeline.processors.base_processor import (
    LeafProcessor, CompositeProcessor, ProcessingError,
    SharedProperty, StopCompositionProcessing
)


class _DummyLeafProcessor1(LeafProcessor):
    pass


class _DummyLeafProcessor2(LeafProcessor):
    def process(self, processed):
        raise StopCompositionProcessing


class _DummyLeafProcessor3(LeafProcessor):
    pass


class _DummyCompProcessor1(CompositeProcessor):
    param1 = SharedProperty()
    analysis_type = SharedProperty()


class _DummyCompProcessor2(CompositeProcessor):
    analysis_type = SharedProperty()


class TestProcessorComposition(unittest.TestCase):
    def setUp(self):
        self._leaf1 = _DummyLeafProcessor1()
        self._leaf2 = _DummyLeafProcessor2()
        self._leaf3 = _DummyLeafProcessor3()

        self._comp1 = _DummyCompProcessor1()
        self._comp2 = _DummyCompProcessor2()

        self._comp1.add(self._leaf1)
        self._comp1.add(self._leaf2)
        self._comp1.add(self._leaf3)

    def testRaises(self):
        with self.assertRaises(TypeError):
            self._comp1.add(self._comp2)

    def testComposition(self):
        self.assertIs(self._leaf1._parent, self._comp1)
        self.assertIs(self._leaf2._parent, self._comp1)

        self.assertListEqual([self._leaf1, self._leaf2, self._leaf3],
                             self._comp1._children)

        leaf3 = self._comp1.pop()
        self.assertListEqual([self._leaf1, self._leaf2], self._comp1._children)
        self.assertIs(leaf3, self._leaf3)

        leaf2 = self._comp1.pop()
        self.assertListEqual([self._leaf1], self._comp1._children)
        self.assertIs(leaf2, self._leaf2)

        self._comp1.remove(self._leaf1)
        self.assertListEqual([], self._comp1._children)

    def testSharedPropertyInitialization(self):
        self.assertEqual(0, len(self._comp1._params))
        # if a key does not exist, it will be assigned to None when visited
        self.assertEqual(None, self._comp1.param1)
        self.assertEqual(None, self._comp1._params['param1'])
        self.assertEqual(1, len(self._comp1._params))

    def testProcessInterface(self):
        self._leaf1.process = MagicMock()
        self._leaf1.run_once({})
        self._leaf1.process.assert_called_once_with({})
        self._leaf1.process.reset_mock()

        self._leaf2.process = MagicMock()
        self._comp1.process = MagicMock()
        self._comp1.run_once({})
        self._comp1.process.assert_called_once_with({})
        self._leaf1.process.assert_called_once_with({})
        self._leaf2.process.assert_called_once_with({})

    def testResetInterface(self):
        self._leaf1.reset = MagicMock()
        self._leaf1.reset_all()
        self._leaf1.reset.assert_called_once_with()
        self._leaf1.reset.reset_mock()

        self._leaf2.reset = MagicMock()
        self._comp1.reset = MagicMock()
        self._comp1.reset_all()
        self._leaf1.reset.assert_called_once_with()
        self._leaf2.reset.assert_called_once_with()
        self._comp1.reset.assert_called_once_with()

    def testSharedPropertyPropagation(self):
        self._comp1.param1 = 5
        self._comp1.run_once({})
        self.assertEqual(5, self._leaf1.param1)
        self.assertEqual(5, self._leaf2.param1)
        with self.assertRaises(AttributeError):
            # composition process was stopped by LeafProcessor2
            self._leaf3.param1

        def setter(obj, attr):
            for i in range(10):
                setattr(obj, attr, i)
                time.sleep(0.005)

        # if the process() method of a parent CompositionProcessor
        # takes some time, shared properties could be modified during
        # this period. After the processing is finished, the modified
        # shared properties will be passed to its children processors.
        # The following code tests this race condition!!!
        t = Thread(target=setter, args=(self._comp1, 'param1'))
        t.start()
        value = self._comp1.param1
        self._comp1.run_once({})
        self.assertEqual(value, self._leaf1.param1)
        self.assertEqual(value, self._leaf2.param1)
        t.join()

    def testStopProcessing(self):
        self._leaf3.process = MagicMock()

        self._comp1.run_once({})
        self.assertFalse(self._leaf3.process.called)


class TestRedisParserMixin(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._proc = _DummyLeafProcessor1()

    def testStr2Tuple(self):
        self.assertTupleEqual((1.0, 2.0), self._proc.str2tuple('(1, 2)'))

    def testStr2List(self):
        self.assertListEqual([1.0, 2.0], self._proc.str2list('[1, 2]'))
        self.assertListEqual([1], self._proc.str2list('[1]'))
        self.assertListEqual([], self._proc.str2list('[]'))


class TestBaseProcessor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        logger.setLevel("CRITICAL")

        # do not use the config file in the current computer
        _Config._filename = os.path.join(tempfile.mkdtemp(), "config.json")
        ConfigWrapper()   # ensure file

        fai = FAI('LPD')
        fai.init()

        cls.app = fai.app
        cls.gui = fai.gui
        cls.fai = fai
        cls.scheduler = fai.scheduler

    @classmethod
    def tearDownClass(cls):
        cls.gui.close()

    def testAnalysisType(self):
        self._comp1 = _DummyCompProcessor1()
        self._comp2 = _DummyCompProcessor2()
        self._leaf1 = _DummyLeafProcessor1()

        with self.assertRaises(ProcessingError):
            self._comp1._update_analysis(1)

        self._comp1._update_analysis(AnalysisType.UNDEFINED)
        self._check_has_no_analysis(AnalysisType.UNDEFINED)

        # set new analysis type for comp2
        self._comp2._update_analysis(AnalysisType.ROI)
        self.assertEqual(AnalysisType.UNDEFINED, self._comp1.analysis_type)
        self.assertEqual(AnalysisType.ROI, self._comp2.analysis_type)
        self._check_has_analysis(AnalysisType.ROI)

        # set new analysis type for comp1
        self._comp1._update_analysis(AnalysisType.PP_AZIMUTHAL_INTEG)
        self._check_has_analysis(AnalysisType.PP_AZIMUTHAL_INTEG)
        self.assertEqual(AnalysisType.ROI, self._comp2.analysis_type)
        self.assertEqual(AnalysisType.PP_AZIMUTHAL_INTEG, self._comp1.analysis_type)

        # unset analysis type for comp1
        self._comp1._update_analysis(AnalysisType.UNDEFINED)
        self.assertEqual(AnalysisType.UNDEFINED, self._comp1.analysis_type)
        self.assertEqual(AnalysisType.ROI, self._comp2.analysis_type)
        self._check_has_analysis(AnalysisType.ROI)
        self._check_has_no_analysis(AnalysisType.PP_AZIMUTHAL_INTEG)

        # unset analysis type for comp2
        self._comp2._update_analysis(AnalysisType.UNDEFINED)
        self.assertEqual(AnalysisType.UNDEFINED, self._comp1.analysis_type)
        self.assertEqual(AnalysisType.UNDEFINED, self._comp2.analysis_type)
        self._check_has_no_analysis(AnalysisType.ROI)
        self._check_has_no_analysis(AnalysisType.PP_AZIMUTHAL_INTEG)

        # set same analysis type for comp1 and comp2
        self._comp1._update_analysis(AnalysisType.ROI)
        self._comp2._update_analysis(AnalysisType.ROI)
        self._check_has_analysis(AnalysisType.ROI)
        self._comp2._update_analysis(AnalysisType.UNDEFINED)
        self._check_has_analysis(AnalysisType.ROI)
        self._comp1._update_analysis(AnalysisType.UNDEFINED)
        self._check_has_no_analysis(AnalysisType.ROI)

    def _check_has_analysis(self, analysis_type):
        self.assertTrue(self._comp2._has_analysis(analysis_type))
        self.assertTrue(self._comp2._has_analysis(analysis_type))
        self.assertTrue(self._leaf1._has_analysis(analysis_type))

    def _check_has_no_analysis(self, analysis_type):
        self.assertFalse(self._comp2._has_analysis(analysis_type))
        self.assertFalse(self._comp2._has_analysis(analysis_type))
        self.assertFalse(self._leaf1._has_analysis(analysis_type))
