import unittest

import numpy as np

from extra_foam.config import AnalysisType
from extra_foam.database import MetaProxy
from extra_foam.logger import logger
from extra_foam.pipeline.processors.base_processor import (
    _BaseProcessor, ProcessingError, SimpleSequence, SimpleVectorSequence,
    SimplePairSequence, _StatDataItem, OneWayAccuPairSequence
)
from extra_foam.processes import wait_until_redis_shutdown
from extra_foam.services import start_redis_server

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
        str2list = self._proc.str2list

        self.assertListEqual([1.0, 2.0], str2list('[1, 2]'))
        self.assertListEqual([1], str2list('[1]'))
        self.assertListEqual([], str2list('[]'))

    def testStr2Slice(self):
        self.assertEqual(slice(None, 2), self._proc.str2slice('[None, 2]'))
        self.assertEqual(slice(1, 10, 2), self._proc.str2slice('[1, 10, 2]'))


class TestBaseProcessor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        start_redis_server()

        cls._meta = MetaProxy()

    @classmethod
    def tearDownClass(cls):
        wait_until_redis_shutdown()

    def testUpdateAnalysisType(self):
        self._proc1 = _DummyProcessor()
        self._proc2 = _DummyProcessor()
        self._proc3 = _DummyProcessor()

        with self.assertRaises(ProcessingError):
            self._proc1._update_analysis(1)

        self._proc1._update_analysis(AnalysisType.UNDEFINED)
        self._check_has_no_analysis(AnalysisType.UNDEFINED)

        # set new analysis type for proc2
        self._proc2._update_analysis(AnalysisType.ROI_PROJ)
        self.assertEqual(AnalysisType.UNDEFINED, self._proc1.analysis_type)
        self.assertEqual(AnalysisType.ROI_PROJ, self._proc2.analysis_type)
        self._check_has_analysis(AnalysisType.ROI_PROJ)

        # set new analysis type for proc1
        self._proc1._update_analysis(AnalysisType.AZIMUTHAL_INTEG)
        self._check_has_analysis(AnalysisType.AZIMUTHAL_INTEG)
        self.assertEqual(AnalysisType.ROI_PROJ, self._proc2.analysis_type)
        self.assertEqual(AnalysisType.AZIMUTHAL_INTEG, self._proc1.analysis_type)

        # unset analysis type for proc1
        self._proc1._update_analysis(AnalysisType.UNDEFINED)
        self.assertEqual(AnalysisType.UNDEFINED, self._proc1.analysis_type)
        self.assertEqual(AnalysisType.ROI_PROJ, self._proc2.analysis_type)
        self._check_has_analysis(AnalysisType.ROI_PROJ)
        self._check_has_no_analysis(AnalysisType.AZIMUTHAL_INTEG)

        # unset analysis type for proc2
        self._proc2._update_analysis(AnalysisType.UNDEFINED)
        self.assertEqual(AnalysisType.UNDEFINED, self._proc1.analysis_type)
        self.assertEqual(AnalysisType.UNDEFINED, self._proc2.analysis_type)
        self._check_has_no_analysis(AnalysisType.ROI_PROJ)
        self._check_has_no_analysis(AnalysisType.AZIMUTHAL_INTEG)

        # set same analysis type for proc1 and proc2
        self._proc1._update_analysis(AnalysisType.ROI_PROJ)
        self._proc2._update_analysis(AnalysisType.ROI_PROJ)
        self._check_has_analysis(AnalysisType.ROI_PROJ)
        self._proc2._update_analysis(AnalysisType.UNDEFINED)
        self._check_has_analysis(AnalysisType.ROI_PROJ)
        self._proc1._update_analysis(AnalysisType.UNDEFINED)
        self._check_has_no_analysis(AnalysisType.ROI_PROJ)

    def _check_has_analysis(self, analysis_type):
        self.assertTrue(self._meta.has_analysis(analysis_type))
        self.assertTrue(self._meta.has_analysis(analysis_type))
        # check with another processor
        self.assertTrue(self._meta.has_analysis(analysis_type))

    def _check_has_no_analysis(self, analysis_type):
        self.assertFalse(self._meta.has_analysis(analysis_type))
        self.assertFalse(self._meta.has_analysis(analysis_type))
        self.assertFalse(self._meta.has_analysis(analysis_type))


class TestSequenceData(unittest.TestCase):
    def testSimpleSequence(self):
        MAX_LENGTH = 10000

        hist = SimpleSequence(max_len=MAX_LENGTH)
        self.assertEqual(0, len(hist))

        hist.append(3)
        hist.append(4)
        hist.extend([1, 2])
        ax = hist.data()
        np.testing.assert_array_almost_equal([3, 4, 1, 2], ax)

        # test Sequence protocol
        self.assertEqual(3, hist[0])
        self.assertEqual(2, hist[-1])
        with self.assertRaises(IndexError):
            hist[4]
        self.assertEqual(4, len(hist))

        self.assertEqual(1, hist.min)
        self.assertEqual(4, hist.max)
        self.assertTupleEqual((1, 4), hist.range)

        # test reset
        hist.reset()
        np.testing.assert_array_almost_equal([], hist.data())
        self.assertEqual(None, hist.min)
        self.assertEqual(None, hist.max)
        self.assertTupleEqual((None, None), hist.range)

        # ----------------------------
        # test when max length reached
        # ----------------------------

        overflow = 10
        for i in range(MAX_LENGTH + overflow):
            hist.append(i)
        ax = hist.data()
        self.assertEqual(MAX_LENGTH, len(ax))
        self.assertEqual(overflow, ax[0])
        self.assertEqual(MAX_LENGTH + overflow - 1, ax[-1])
        self.assertEqual(0, hist.min)
        self.assertEqual(MAX_LENGTH + overflow - 1, hist.max)
        self.assertTupleEqual((0, MAX_LENGTH + overflow - 1), hist.range)

    def testSimpleVectorSequence(self):
        MAX_LENGTH = 1000

        hist = SimpleVectorSequence(2, max_len=MAX_LENGTH)
        self.assertEqual(0, len(hist))
        self.assertTrue(hist.data().flags['C_CONTIGUOUS'])
        hist = SimpleVectorSequence(2, max_len=MAX_LENGTH, order='F')
        self.assertTrue(hist.data().flags['F_CONTIGUOUS'])

        with self.assertRaises(TypeError):
            hist.append(3)

        with self.assertRaises(TypeError):
            hist.append(None)

        with self.assertRaises(ValueError):
            hist.append(np.array([[1, 2], [3, 4]]))

        hist.append([3, 4])
        hist.append(np.array([1, 2]))
        ax = hist.data()
        np.testing.assert_array_almost_equal([[3, 4], [1, 2]], ax)

        # test Sequence protocol
        np.testing.assert_array_almost_equal([3, 4], hist[0])
        np.testing.assert_array_almost_equal([1, 2], hist[-1])
        with self.assertRaises(IndexError):
            hist[4]
        self.assertEqual(2, len(hist))

        # test reset
        hist.reset()
        self.assertEqual(0, len(hist.data()))

        # ----------------------------
        # test when max length reached
        # ----------------------------

        overflow = 10
        for i in range(MAX_LENGTH + overflow):
            hist.append([i, i])
        ax = hist.data()
        self.assertEqual(MAX_LENGTH, len(ax))
        np.testing.assert_array_almost_equal([overflow, overflow], ax[0])
        np.testing.assert_array_almost_equal([MAX_LENGTH + overflow - 1,
                                              MAX_LENGTH + overflow - 1], ax[-1])

    def testSimplePairSequence(self):
        MAX_LENGTH = 10000

        hist = SimplePairSequence(max_len=MAX_LENGTH)
        self.assertEqual(0, len(hist))

        hist.append((3, 200))
        hist.append((4, 220))
        ax, ay = hist.data()
        np.testing.assert_array_almost_equal([3, 4], ax)
        np.testing.assert_array_almost_equal([200, 220], ay)

        # test Sequence protocol
        self.assertTupleEqual((3, 200), hist[0])
        self.assertTupleEqual((4, 220), hist[-1])
        with self.assertRaises(IndexError):
            hist[2]
        self.assertEqual(2, len(hist))

        # test reset
        hist.reset()
        ax, ay = hist.data()
        np.testing.assert_array_almost_equal([], ax)
        np.testing.assert_array_almost_equal([], ay)

        # ----------------------------
        # test when max length reached
        # ----------------------------

        overflow = 10
        for i in range(MAX_LENGTH + overflow):
            hist.append((i, i))
        ax, ay = hist.data()
        self.assertEqual(MAX_LENGTH, len(ax))
        self.assertEqual(MAX_LENGTH, len(ay))
        self.assertEqual(overflow, ax[0])
        self.assertEqual(overflow, ay[0])
        self.assertEqual(MAX_LENGTH + overflow - 1, ax[-1])
        self.assertEqual(MAX_LENGTH + overflow - 1, ay[-1])

    def testOneWayAccuPairSequence(self):
        MAX_LENGTH = 600

        # resolution is missing
        with self.assertRaises(TypeError):
            OneWayAccuPairSequence()
        # resolution == 0
        with self.assertRaises(ValueError):
            OneWayAccuPairSequence(0.0)
        # resolution < 0
        with self.assertRaises(ValueError):
            OneWayAccuPairSequence(-1)

        hist = OneWayAccuPairSequence(0.1, max_len=MAX_LENGTH, min_count=2)
        self.assertEqual(0, len(hist))

        # distance between two adjacent data > resolution
        hist.append((1, 0.3))
        hist.append((2, 0.4))
        ax, ay = hist.data()
        np.testing.assert_array_equal([], ax)
        np.testing.assert_array_equal([], ay.avg)
        np.testing.assert_array_equal([], ay.min)
        np.testing.assert_array_equal([], ay.max)
        np.testing.assert_array_equal([], ay.count)

        hist.append((2.02, 0.5))
        ax, ay = hist.data()
        np.testing.assert_array_equal([2.01], ax)
        np.testing.assert_array_equal([0.45], ay.avg)
        np.testing.assert_array_almost_equal([0.425], ay.min)
        np.testing.assert_array_almost_equal([0.475], ay.max)
        np.testing.assert_array_equal([2], ay.count)

        hist.append((2.10, 0.6))
        ax, ay = hist.data()
        np.testing.assert_array_equal([2.04], ax)
        np.testing.assert_array_equal([0.5], ay.avg)
        np.testing.assert_array_almost_equal([0.4591751709536137], ay.min)
        np.testing.assert_array_almost_equal([0.5408248290463863], ay.max)
        np.testing.assert_array_equal([3], ay.count)

        # new point
        hist.append((2.31, 1))
        hist.append((2.41, 2))
        ax, ay = hist.data()
        np.testing.assert_array_equal([0.5, 1.5], ay.avg)
        np.testing.assert_array_almost_equal([0.4591751709536137, 1.25], ay.min)
        np.testing.assert_array_almost_equal([0.5408248290463863, 1.75], ay.max)
        np.testing.assert_array_equal([3, 2], ay.count)

        # test Sequence protocol
        x, y = hist[0]
        self.assertAlmostEqual(2.04, x)
        self.assertEqual(_StatDataItem(0.5, 0.4591751709536137, 0.5408248290463863, 3), y)
        x, y = hist[-1]
        self.assertAlmostEqual(2.36, x)
        self.assertEqual(_StatDataItem(1.5, 1.25, 1.75, 2), y)
        with self.assertRaises(IndexError):
            hist[2]

        # test reset
        hist.reset()
        ax, ay = hist.data()
        np.testing.assert_array_equal([], ax)
        np.testing.assert_array_equal([], ay.avg)
        np.testing.assert_array_equal([], ay.min)
        np.testing.assert_array_equal([], ay.max)
        np.testing.assert_array_equal([], ay.count)

        # ----------------------------
        # test when max length reached
        # ----------------------------

        overflow = 10
        for i in range(2 * MAX_LENGTH + 2 * overflow):
            # two adjacent data point will be grouped together since resolution is 0.1
            hist.append((0.1 * i, i))
        ax, ay = hist.data()
        self.assertEqual(MAX_LENGTH, len(ax))
        self.assertEqual(MAX_LENGTH, len(ay.count))
        self.assertEqual(MAX_LENGTH, len(ay.avg))
        self.assertEqual(0.2 * overflow + 0.1 * 0.5, ax[0])
        self.assertEqual(2 * overflow + 0.5, ay.avg[0])
        self.assertEqual(0.2 * (MAX_LENGTH + overflow - 1) + 0.1 * 0.5, ax[-1])
        self.assertEqual(2 * (MAX_LENGTH + overflow - 1) + 0.5, ay.avg[-1])
