import unittest

from extra_foam.pipeline.f_transformer import DataTransformer
from extra_foam.config import DataSource
from extra_foam.database import SourceItem
from extra_foam.pipeline.tests import _RawDataMixin


class TestDataTransformer(_RawDataMixin, unittest.TestCase):
    def testTransformWithoutModules(self):
        transformer = DataTransformer.transform_euxfel
        src_type = DataSource.BRIDGE

        catalog = self._create_catalog({'ABC': [('abc', 'ppt')]})

        # no data is available
        data = (dict(), dict())  # raw, meta
        raw, meta, tid = transformer(data, catalog=catalog, source_type=src_type)
        self.assertDictEqual({}, raw)
        self.assertDictEqual({}, meta)
        self.assertEqual(-1, tid)

        # test with wrong property
        data = self._gen_kb_data(1234, {'abc': [('ppt_false', 1)]})
        raw, meta, tid = transformer(data, catalog=catalog, source_type=src_type)
        self.assertDictEqual({}, raw)
        self.assertDictEqual({}, meta)
        self.assertEqual(1234, tid)

        # test with valid data
        data = self._gen_kb_data(1235, {'abc': [('ppt', 1)]})
        raw, meta, tid = transformer(data, catalog=catalog, source_type=src_type)
        self.assertDictEqual({'abc ppt': 1}, raw)
        self.assertDictEqual({'abc ppt': {'train_id': 1235, 'source_type': src_type}}, meta)
        self.assertEqual(1235, tid)

        # test with valid data (only one out of two properties are requested)
        data = self._gen_kb_data(1236, {'abc': [('ppt', 1), ('ppt2', 2)]})
        raw, meta, tid = transformer(data, catalog=catalog, source_type=src_type)
        self.assertDictEqual({'abc ppt': 1}, raw)
        self.assertDictEqual({'abc ppt': {'train_id': 1236, 'source_type': src_type}}, meta)
        self.assertEqual(1236, tid)

    def testTransformWithModules(self):
        transformer = DataTransformer.transform_euxfel
        src_type = DataSource.BRIDGE

        # test with valid data
        data = self._gen_kb_data(1234, {
            'abc': [('ppt1', 1), ('ppt2', 2)],
            'xyz_1:xtdf': [('ppt', 2)], 'xyz_2:xtdf': [('ppt', 3)]
        })
        catalog = self._create_catalog(
            {'ABC': [('abc', 'ppt2')]})
        catalog.add_item(SourceItem('XYZ', 'xyz_*:xtdf', [1, 2], 'ppt', slice(None, None), [0, 100]))
        raw, meta, tid = transformer(data, catalog=catalog, source_type=src_type)
        self.assertDictEqual({
            'abc ppt2': 2,
            'xyz_*:xtdf ppt': {'xyz_1:xtdf': {'ppt': 2}, 'xyz_2:xtdf': {'ppt': 3}}
        }, raw)
        self.assertDictEqual({
            'abc ppt2': {'train_id': 1234, 'source_type': src_type},
            'xyz_*:xtdf ppt': {'train_id': 1234, 'source_type': src_type}
        }, meta)

        # test when requested modules are different from modules in the data
        data = self._gen_kb_data(1235, {
            'abc': [('ppt1', 1), ('ppt2', 2)],
            'xyz_1:xtdf': [('ppt', 2)], 'xyz_2:xtdf': [('ppt', 3)], 'xyz_3:xtdf': [('ppt', 4)]
        })
        catalog.remove_item('xyz_*:xtdf ppt')
        catalog.add_item(SourceItem('XYZ', 'xyz_*:xtdf', [0, 1, 3, 4], 'ppt', slice(None, None), [0, 100]))
        raw, meta, tid = transformer(data, catalog=catalog, source_type=src_type)
        self.assertDictEqual({
            'abc ppt2': 2,
            'xyz_*:xtdf ppt': {'xyz_1:xtdf': {'ppt': 2}, 'xyz_3:xtdf': {'ppt': 4}}
        }, raw)
        self.assertDictEqual({
            'abc ppt2': {'train_id': 1235, 'source_type': src_type},
            'xyz_*:xtdf ppt': {'train_id': 1235, 'source_type': src_type}
        }, meta)

    def testCorrelationSingle(self):
        catalog = self._create_catalog({"ABC": [("abc", "ppt")]})

        trans = DataTransformer(catalog)

        correlated, dropped = trans.correlate(self._gen_kb_data(1001, {"abc": [("ppt", 1)]}))
        self.assertDictEqual(
            {'abc ppt': {'train_id': 1001, 'source_type': DataSource.UNKNOWN}}, correlated['meta'])
        self.assertDictEqual({'abc ppt': 1}, correlated['raw'])
        self.assertEqual(1001, correlated['processed'].tid)
        self.assertListEqual([], dropped)

        # not correlated
        correlated, dropped = trans.correlate(self._gen_kb_data(1002, {"Motor": [("b", 1)]}))
        self.assertIsNone(correlated)
        self.assertListEqual([], dropped)

        # one more not correlated
        correlated, dropped = trans.correlate(self._gen_kb_data(1003, {"Motor": [("b", 2)]}))
        self.assertIsNone(correlated)
        self.assertListEqual([], dropped)

        # correlated
        correlated, dropped = trans.correlate(self._gen_kb_data(1004, {"abc": [("ppt", 2)]}))
        self.assertDictEqual(
            {'abc ppt': {'train_id': 1004, 'source_type': DataSource.UNKNOWN}}, correlated['meta'])
        self.assertDictEqual({'abc ppt': 2}, correlated['raw'])
        self.assertEqual(1004, correlated['processed'].tid)
        self.assertListEqual([1002, 1003], dropped)

    def testCorrelationMultiple(self):
        catalog = self._create_catalog({"ABC": [("abc", "ppt")], "EFG": [("efg", "ppt")]})

        trans = DataTransformer(catalog)

        for tid in [1001, 1002, 1003]:
            correlated, dropped = trans.correlate(self._gen_kb_data(tid, {"abc": [("ppt", 1)]}))
            self.assertIsNone(correlated)
            self.assertListEqual([], dropped)

        correlated, dropped = trans.correlate(self._gen_kb_data(1002, {"efg": [("ppt", 1)]}))
        self.assertDictEqual(
            {'abc ppt': {'train_id': 1002, 'source_type': DataSource.UNKNOWN},
             'efg ppt': {'train_id': 1002, 'source_type': DataSource.UNKNOWN}}, correlated['meta'])
        self.assertDictEqual({'abc ppt': 1, 'efg ppt': 1}, correlated['raw'])
        self.assertEqual(1002, correlated['processed'].tid)
        self.assertListEqual([1001], dropped)
        self.assertListEqual([1003], list(trans._cached.keys()))

    def testCacheIsFull(self):
        catalog = self._create_catalog({"ABC": [("abc", "ppt")], "Motor": [("efg", "ppt")]})
        cache_size = 5
        trans = DataTransformer(catalog, cache_size=5)

        for i in range(cache_size + 2):
            correlated, dropped = trans.correlate(self._gen_kb_data(1000 + i, {"abc": [("ppt", 2)]}))
            self.assertIsNone(correlated)
            if i + 1 > cache_size:
                self.assertListEqual([1000 + i - cache_size], dropped)
                self.assertEqual(cache_size, len(trans._cached))
            else:
                self.assertListEqual([], dropped)
                self.assertEqual(i + 1, len(trans._cached))
