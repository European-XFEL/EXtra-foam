import unittest
from unittest.mock import patch
import copy

from extra_foam.database.data_source import (
    DataTransformer, OrderedSet, SourceCatalog, SourceItem
)
from extra_foam.config import config, DataSource


@patch.dict(config._data, {"DETECTOR": "DSSC"})
class TestSourceCatalog(unittest.TestCase):
    def testGeneral(self):
        catalog = SourceCatalog()
        item = SourceItem('DSSC', 'dssc_device_id', [], 'image.data', None, None)

        src = f"{item.name} {item.property}"
        catalog.add_item(item)
        self.assertEqual(src, catalog.main_detector)
        self.assertEqual(1, len(catalog))
        catalog.remove_item(src)
        self.assertEqual('', catalog.main_detector)
        self.assertEqual(0, len(catalog))

        item1 = SourceItem('Motor', 'motor_device1', [], 'actualPosition', None, (-1, 1))
        src1 = f"{item1.name} {item1.property}"
        item2 = SourceItem('XGM', 'xgm_device', [], 'intensityTD', slice(1, 10, 1), (0, 100))
        src2 = f"{item2.name} {item2.property}"
        catalog.add_item(item1)
        catalog.add_item(item2)
        self.assertEqual(2, len(catalog))
        self.assertEqual('Motor', catalog.get_category(src1))
        self.assertEqual(None, catalog.get_slicer(src1))
        self.assertEqual(slice(1, 10, 1), catalog.get_slicer(src2))
        self.assertEqual((0, 100), catalog.get_vrange(src2))

        item3 = SourceItem('Motor', 'motor_device2', [], 'actualPosition', None, (-1, 1))
        src3 = f"{item3.name} {item3.property}"
        catalog.add_item(item3)
        self.assertEqual(OrderedSet([src1, src3]), catalog.from_category("Motor"))

        catalog.clear()
        self.assertEqual(0, len(catalog))
        self.assertEqual('', catalog.main_detector)

    def testCopy(self):
        catalog = SourceCatalog()
        catalog.add_item(SourceItem('DSSC', 'dssc_device_id', [], 'image.data', None, None))
        catalog.add_item(SourceItem('Motor', 'motor_device1', [], 'actualPosition', None, (-1, 1)))
        catalog.add_item(SourceItem('XGM', 'xgm_device', [], 'intensityTD', slice(1, 10, 1), (0, 100)))

        catalog_cp = copy.copy(catalog)
        self.assertDictEqual(catalog._items, catalog_cp._items)
        self.assertIsNot(catalog._items, catalog_cp._items)
        self.assertDictEqual(catalog._categories, catalog_cp._categories)
        self.assertIsNot(catalog._categories, catalog_cp._categories)
        self.assertEqual(catalog._main_detector_category, catalog_cp._main_detector_category)
        self.assertEqual(catalog._main_detector, catalog_cp._main_detector)


class TestDataTransformer(unittest.TestCase):
    def testExtractData(self):
        transformer = DataTransformer.transform_euxfel

        src_type = DataSource.BRIDGE
        det_ctg = 'ABC'
        det_id = 'abc'
        det_ppt = 'p_abc'
        det_src = f"{det_id} {det_ppt}"

        # no data is available
        raw = dict()
        meta = dict()
        catalog = SourceCatalog()
        catalog.add_item(SourceItem(det_ctg, det_id, [], det_ppt, None, None))
        new_raw, new_meta, not_found = transformer(
            raw, meta, catalog=catalog, source_type=src_type)
        self.assertDictEqual({}, new_raw)
        self.assertDictEqual({}, new_meta)
        self.assertEqual(1, len(catalog))
        self.assertEqual(1, len(not_found))

        # requested, source names are in both 'raw' and 'meta', however,
        # property is not in 'raw'.
        raw = {det_id: dict()}
        meta = {det_id: {'timestamp.tid': 1234}}
        catalog = SourceCatalog()
        catalog.add_item(SourceItem(det_ctg, det_id, [], det_ppt, None, None))
        new_raw, new_meta, not_found = transformer(
            raw, meta, catalog=catalog, source_type=src_type)
        self.assertDictEqual({}, new_raw)
        self.assertDictEqual({}, new_meta)
        self.assertEqual(1, len(catalog))
        self.assertEqual(1, len(not_found))

        # with valid non-modular data
        raw = {det_id: {det_ppt: 1}}
        meta = {det_id: {'timestamp.tid': 1234}}
        catalog = SourceCatalog()
        catalog.add_item(SourceItem(det_ctg, det_id, [], det_ppt, None, None))
        new_raw, new_meta, not_found = transformer(
            raw, meta, catalog=catalog, source_type=src_type)
        self.assertDictEqual({det_src: 1}, new_raw)
        self.assertDictEqual({
            det_src: {'tid': 1234, 'source_type': src_type},
        }, new_meta)
        self.assertIn(det_src, catalog)

        # with valid non-modular data (two properties of one source are requested)
        det_ppt2 = 'p2_abc'
        raw = {det_id: {det_ppt: 1, det_ppt2: 2}}
        meta = {det_id: {'timestamp.tid': 1234}}
        catalog = SourceCatalog()
        catalog.add_item(SourceItem(det_ctg, det_id, [], det_ppt, None, None))
        new_raw, new_meta, not_found = transformer(
            raw, meta, catalog=catalog, source_type=src_type)
        self.assertDictEqual({det_src: 1}, new_raw)
        self.assertDictEqual({
            det_src: {'tid': 1234, 'source_type': src_type},
        }, new_meta)
        self.assertIn(det_src, catalog)
        self.assertEqual(1, len(catalog))
        self.assertEqual(0, len(not_found))

        # with modules
        raw = {
            det_id: {det_ppt: 1, det_ppt2: 2},
            'xyz_1:xtdf': {'p_xyz': 2}, 'xyz_2:xtdf': {'p_xyz': 3}
        }
        meta = {
            det_id: {'timestamp.tid': 1234},
            'xyz_1:xtdf': {'timestamp.tid': 1234}, 'xyz_2:xtdf': {'timestamp.tid': 1234}
        }
        catalog = SourceCatalog()
        catalog.add_item(SourceItem(det_ctg, det_id, [], det_ppt, None, None))
        catalog.add_item(SourceItem(det_ctg, det_id, [], det_ppt2, None, None))
        catalog.add_item(SourceItem('XYZ', 'xyz_*:xtdf', [1, 2], 'p_xyz', slice(None, None), [0, 100]))
        new_raw, new_meta, not_found = transformer(
            raw, meta, catalog=catalog, source_type=src_type)
        self.assertDictEqual({
            'abc p_abc': 1,
            'abc p2_abc': 2,
            'xyz_*:xtdf p_xyz': {'xyz_1:xtdf': {'p_xyz': 2}, 'xyz_2:xtdf': {'p_xyz': 3}}
        }, new_raw)
        self.assertDictEqual({
            'abc p_abc': {'tid': 1234, 'source_type': src_type},
            'abc p2_abc': {'tid': 1234, 'source_type': src_type},
            'xyz_*:xtdf p_xyz': {'tid': 1234, 'source_type': src_type}
        }, new_meta)
        self.assertEqual(3, len(catalog))
        self.assertIn('abc p_abc', catalog)
        self.assertIn('abc p2_abc', catalog)
        self.assertIn('xyz_*:xtdf p_xyz', catalog)

        # requested modules are different from modules in the data
        raw = {
            det_id: {det_ppt: 1, det_ppt2: 2},
            'xyz_1:xtdf': {'p_xyz': 2},
            'xyz_2:xtdf': {'p_xyz': 3},
            'xyz_3:xtdf': {'p_xyz': 4}
        }
        meta = {
            det_id: {'timestamp.tid': 1234},
            'xyz_1:xtdf': {'timestamp.tid': 1234},
            'xyz_2:xtdf': {'timestamp.tid': 1234},
            'xyz_3:xtdf': {'timestamp.tid': 1234}
        }
        catalog = SourceCatalog()
        catalog.add_item(SourceItem(det_ctg, det_id, [], det_ppt, None, None))
        catalog.add_item(SourceItem(det_ctg, det_id, [], det_ppt2, None, None))
        catalog.add_item(SourceItem('XYZ', 'xyz_*:xtdf', [0, 1, 3, 4], 'p_xyz',
                                    slice(None, None), [0, 100]))
        new_raw, new_meta, not_found = transformer(
            raw, meta, catalog=catalog, source_type=src_type)
        self.assertDictEqual({
            'abc p_abc': 1,
            'abc p2_abc': 2,
            'xyz_*:xtdf p_xyz': {'xyz_1:xtdf': {'p_xyz': 2}, 'xyz_3:xtdf': {'p_xyz': 4}}
        }, new_raw)
        self.assertDictEqual({
            'abc p_abc': {'tid': 1234, 'source_type': src_type},
            'abc p2_abc': {'tid': 1234, 'source_type': src_type},
            'xyz_*:xtdf p_xyz': {'tid': 1234, 'source_type': src_type}
        }, new_meta)
        self.assertEqual(3, len(catalog))
        self.assertIn('abc p_abc', catalog)
        self.assertIn('abc p2_abc', catalog)
        self.assertIn('xyz_*:xtdf p_xyz', catalog)
