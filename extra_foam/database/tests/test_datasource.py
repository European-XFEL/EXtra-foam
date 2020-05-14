import unittest
from unittest.mock import patch
import copy

from extra_foam.database.data_source import (
    OrderedSet, SourceCatalog, SourceItem
)
from extra_foam.config import config


@patch.dict(config._data, {"DETECTOR": "DSSC"})
class TestSourceCatalog(unittest.TestCase):
    def testGeneral(self):
        catalog = SourceCatalog()
        item = SourceItem('DSSC', 'dssc_device_id', [], 'image.data', None, None, 1)

        self.assertIn("META timestamp.tid", catalog)

        src = f"{item.name} {item.property}"
        catalog.add_item(item)
        self.assertEqual(src, catalog.main_detector)
        self.assertEqual(1, len(catalog))
        catalog.remove_item(src)
        self.assertEqual('', catalog.main_detector)
        self.assertEqual(0, len(catalog))

        src1 = f"motor_device1 actualPosition"
        src2 = f"xgm_device intensityTD"
        catalog.add_item(
            'Motor', 'motor_device1', [], 'actualPosition', None, (-1, 1), 0)
        catalog.add_item(
            'XGM', 'xgm_device', [], 'intensityTD', vrange=(0, 100), slicer=slice(1, 10, 1), ktype=1)
        self.assertEqual(2, len(catalog))
        self.assertEqual('Motor', catalog.get_category(src1))
        self.assertEqual(None, catalog.get_slicer(src1))
        self.assertEqual(slice(1, 10, 1), catalog.get_slicer(src2))
        self.assertEqual((0, 100), catalog.get_vrange(src2))
        self.assertEqual(0, catalog.get_type(src1))
        self.assertEqual(1, catalog.get_type(src2))

        item3 = SourceItem('Motor', 'motor_device2', [], 'actualPosition', None, (-1, 1), 0)
        src3 = f"{item3.name} {item3.property}"
        catalog.add_item(item3)
        self.assertEqual(OrderedSet([src1, src3]), catalog.from_category("Motor"))

        catalog.clear()
        self.assertEqual(0, len(catalog))
        self.assertEqual('', catalog.main_detector)

    def testCopy(self):
        catalog = SourceCatalog()
        catalog.add_item(SourceItem(
            'DSSC', 'dssc_device_id', [], 'image.data', None, None, 1))
        catalog.add_item(SourceItem(
            'Motor', 'motor_device1', [], 'actualPosition', None, (-1, 1), 0))
        catalog.add_item(SourceItem(
            'XGM', 'xgm_device', [], 'intensityTD', slice(1, 10, 1), (0, 100), 1))

        catalog_cp = copy.copy(catalog)
        self.assertDictEqual(catalog._items, catalog_cp._items)
        self.assertIsNot(catalog._items, catalog_cp._items)
        self.assertDictEqual(catalog._categories, catalog_cp._categories)
        self.assertIsNot(catalog._categories, catalog_cp._categories)
        self.assertEqual(catalog._main_detector_category, catalog_cp._main_detector_category)
        self.assertEqual(catalog._main_detector, catalog_cp._main_detector)
