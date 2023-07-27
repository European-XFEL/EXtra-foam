import unittest
from unittest.mock import MagicMock, patch
import os
import tempfile

from extra_foam.config import AnalysisType, config
from extra_foam.database.metadata import Metadata, MetaMetadata
from extra_foam.database import MetaProxy, MonProxy
from extra_foam.processes import wait_until_redis_shutdown
from extra_foam.services import start_redis_server
from extra_foam.gui.misc_widgets.analysis_setup_manager import AnalysisSetupManager

LAST_SAVED = AnalysisSetupManager.LAST_SAVED
DEFAULT = AnalysisSetupManager.DEFAULT

_tmp_cfg_dir = tempfile.mkdtemp()


@patch("extra_foam.config.ROOT_PATH", _tmp_cfg_dir)
@patch.dict(config._data, {"DETECTOR": "DSSC"})
class TestDataProxy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        start_redis_server()

        cls._meta = MetaProxy()
        cls._mon = MonProxy()

        cls._meta.error = MagicMock()

    @classmethod
    def tearDownClass(cls):
        wait_until_redis_shutdown()

        # test 'reset' method
        cls._meta.reset()
        cls._mon.reset()

        os.rmdir(_tmp_cfg_dir)

    def testBaseProxyMethods(self):
        proxy = self._mon

        proxy.hset('name1', 'key1', 'value1')
        self.assertEqual('value1', proxy.hget('name1', 'key1'))

        proxy.hmset('name2', {'key1': 'value1', 'kay2': 'value2'})
        self.assertDictEqual({'key1': 'value1', 'kay2': 'value2'}, proxy.hget_all('name2'))

        ret = proxy.hget_all_multi(['name1', 'name2'])
        self.assertDictEqual({'key1': 'value1'}, ret[0])
        self.assertDictEqual({'key1': 'value1', 'kay2': 'value2'}, ret[1])

    def testAnalysisType(self):
        type1 = AnalysisType.AZIMUTHAL_INTEG
        type2 = AnalysisType.PUMP_PROBE
        type3 = AnalysisType.ROI_FOM

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
        self.assertEqual('2', self._meta.hget(Metadata.ANALYSIS_TYPE, type2.value))

        # unregister an analysis type
        self._meta.unregister_analysis(type1)
        self.assertFalse(self._meta.has_analysis(type1))

        # unregister an analysis type which has not been registered
        self._meta.unregister_analysis(type3)
        self.assertEqual('0', self._meta.hget(Metadata.ANALYSIS_TYPE, type3.value))

    def testMetaMetadata(self):
        class Dummy(metaclass=MetaMetadata):
            DATA_SOURCE = "meta:data_source"
            ANALYSIS_TYPE = "meta:analysis_type"
            GLOBAL_PROC = "meta:proc:global"
            IMAGE_PROC = "meta:proc:image"
            GEOMETRY_PROC = "meta:proc:geometry"

        self.assertListEqual(['global', 'image', 'geometry'], Dummy.processors)
        self.assertListEqual([Dummy.GLOBAL_PROC,
                              Dummy.IMAGE_PROC,
                              Dummy.GEOMETRY_PROC], Dummy.processor_keys)

    def testProcessCount(self):
        mon = self._mon
        mon.reset_process_count()

        tid, n_proc, n_drop, n_proc_p = self._mon.get_process_count()
        self.assertIsNone(tid)
        self.assertEqual('0', n_proc)
        self.assertEqual('0', n_drop)
        self.assertEqual('0', n_proc_p)

        self._mon.add_tid_with_timestamp(1234, n_pulses=20)
        tid, n_proc, n_drop, n_proc_p = self._mon.get_process_count()
        self.assertEqual('1234', tid)
        self.assertEqual('1', n_proc)
        self.assertEqual('0', n_drop)
        self.assertEqual('20', n_proc_p)

        self._mon.add_tid_with_timestamp(1235, n_pulses=10, dropped=True)
        tid, n_proc, n_drop, n_proc_p = self._mon.get_process_count()
        self.assertEqual('1235', tid)
        self.assertEqual('1', n_proc)
        self.assertEqual('1', n_drop)
        self.assertEqual('20', n_proc_p)

    def testSnapshotOperation(self):
        data = {
            Metadata.IMAGE_PROC: {"aaa": '1', "bbb": "(-1, 1)", "ccc": "sea"},
            Metadata.GLOBAL_PROC: {"a1": '10', "b1": 'True'}
        }

        def _write_to_redis(name=None):
            for k, v in data.items():
                kk = k if name is None else f"{k}:{name}"
                self._meta.hmset(kk, v)

        def _assert_data(meta, name=None):
            for k, v in data.items():
                kk = k if name is None else f"{k}:{name}"
                self.assertDictEqual(v, meta.hget_all(kk))

        def _assert_empty_data(meta, name=None):
            for k in data:
                kk = k if name is None else f"{k}:{name}"
                self.assertDictEqual({}, meta.hget_all(kk))

        # test take snapshot
        self._meta.take_snapshot(DEFAULT)
        _assert_empty_data(self._meta, DEFAULT)
        _write_to_redis()

        _assert_empty_data(self._meta, DEFAULT)
        name, timestamp, description = self._meta.take_snapshot(LAST_SAVED)
        self.assertEqual(LAST_SAVED, name)
        self.assertEqual("", description)
        _assert_data(self._meta, LAST_SAVED)

        # test copy snapshot
        self._meta.copy_snapshot(LAST_SAVED, "abc")
        _assert_data(self._meta, LAST_SAVED)
        _assert_data(self._meta, "abc")

        # test load snapshot
        self._meta.load_snapshot(DEFAULT)
        _assert_empty_data(self._meta)
        self._meta.load_snapshot("abc")
        _assert_data(self._meta, "abc")

        # test rename snapshot
        self._meta.rename_snapshot("abc", "efg")
        _assert_empty_data(self._meta, "abc")
        _assert_data(self._meta, "efg")

        # test remove snapshot
        self._meta.remove_snapshot("efg")
        _assert_empty_data(self._meta, "efg")

    def testDumpLoadConfigurations(self):
        filepath = config.setup_file

        image_proc_data = {"aaa": '1', "bbb": "(-1, 1)", "ccc": "sea"}
        self._meta.hmset(f"{Metadata.IMAGE_PROC}:Last saved", image_proc_data)
        self._meta.hmset(f"{Metadata.META_PROC}:Last saved", {
            "timestamp": "2020-03-03 03:03:03",
            "description": ""
        })

        self._meta.hmset(f"{Metadata.IMAGE_PROC}:abc", image_proc_data)
        # no 'description'
        self._meta.hmset(f"{Metadata.META_PROC}:abc", {
            "timestamp": "2020-03-04 03:03:03",
        })

        # no 'timestamp'
        self._meta.hmset(f"{Metadata.IMAGE_PROC}:efg", image_proc_data)

        try:
            self._meta.dump_all_setups([("Last saved", ''), ("abc", "abc setup"), ("efg", "efg setup")])

            self._meta.remove_snapshot("Last saved")
            self._meta.remove_snapshot("abc")
            self._meta.remove_snapshot("efg")
            self.assertDictEqual({}, self._meta.hget_all(f"{Metadata.IMAGE_PROC}:Last saved"))

            lst = self._meta.load_all_setups()

            self._meta.error.assert_called_once()
            self.assertListEqual([('Last saved', '2020-03-03 03:03:03', ''),
                                  ('abc', '2020-03-04 03:03:03', 'abc setup')], lst)

            self.assertDictEqual(image_proc_data, self._meta.hget_all(f"{Metadata.IMAGE_PROC}:Last saved"))
            self.assertDictEqual(image_proc_data, self._meta.hget_all(f"{Metadata.IMAGE_PROC}:abc"))

        finally:
            os.remove(filepath)
            self._meta.remove_snapshot("Last saved")
            self._meta.remove_snapshot("abc")
            self._meta.remove_snapshot("efg")
