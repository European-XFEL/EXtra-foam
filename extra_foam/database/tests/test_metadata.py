import unittest

from extra_foam.config import AnalysisType
from extra_foam.database.metadata import Metadata, MetaMetadata
from extra_foam.database import MetaProxy, MonProxy
from extra_foam.processes import wait_until_redis_shutdown
from extra_foam.services import start_redis_server


class TestDataProxy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        start_redis_server()

        cls._meta = MetaProxy()
        cls._mon = MonProxy()

    @classmethod
    def tearDownClass(cls):
        wait_until_redis_shutdown()

        # test 'reset' method
        cls._meta.reset()
        cls._mon.reset()

    def testAnalysisType(self):
        type1 = AnalysisType.AZIMUTHAL_INTEG
        type2 = AnalysisType.PUMP_PROBE
        type3 = AnalysisType.TR_XAS

        # register a analysis type
        self._meta.register_analysis(type1)
        self.assertTrue(self._meta.has_analysis(type1))
        self.assertFalse(self._meta.has_analysis(type3))
        self.assertTrue(self._meta.has_any_analysis([type1, type2]))
        self.assertFalse(self._meta.has_all_analysis([type1, type2]))

        # register another analysis type
        self._meta.register_analysis(type2)
        self.assertTrue(self._meta.has_all_analysis([type1, type2]))

        # register an analysis type twice
        self._meta.register_analysis(type2)
        self.assertTrue(self._meta.has_all_analysis([type1, type2]))
        self.assertEqual('2', self._meta.hget(Metadata.ANALYSIS_TYPE, type2))

        # unregister an analysis type
        self._meta.unregister_analysis(type1)
        self.assertFalse(self._meta.has_analysis(type1))

        # unregister an analysis type which has not been registered
        self._meta.unregister_analysis(type3)
        self.assertEqual('0', self._meta.hget(Metadata.ANALYSIS_TYPE, type3))

    def testMetaMetadata(self):
        class Dummy(metaclass=MetaMetadata):
            DATA_SOURCE = "meta:data_source"
            ANALYSIS_TYPE = "meta:analysis_type"
            GLOBAL_PROC = "meta:proc:global"
            IMAGE_PROC = "meta:proc:image"
            GEOMETRY_PROC = "meta:proc:geometry"

        self.assertListEqual(['global', 'image', 'geometry'], Dummy.processors)
