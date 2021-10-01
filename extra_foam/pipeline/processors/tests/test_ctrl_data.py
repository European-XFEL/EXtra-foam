"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import unittest

from extra_foam.pipeline.tests import _TestDataMixin
from extra_foam.pipeline.processors.control_data import CtrlDataProcessor
from extra_foam.database import SourceItem
from extra_foam.pipeline.exceptions import SkipTrainError
from extra_foam.config import config, KaraboType


class TestCtrlData(_TestDataMixin, unittest.TestCase):
    def testGeneral(self):
        proc = CtrlDataProcessor()

        data, processed = self.simple_data(1234, (2, 2))
        meta = data['meta']
        raw = data['raw']
        catalog = data['catalog']

        for ctg in ['Motor', 'Monochromator', 'Magnet', config["SOURCE_USER_DEFINED_CATEGORY"]]:
            item = SourceItem(ctg, 'device1', [], 'property1', None, (-1, 1), KaraboType.CONTROL_DATA)
            catalog.add_item(item)
            src = f"{item.name} {item.property}"
            meta[src] = {'train_id': 12346}
            raw[src] = 0
            proc.process(data)
            raw[src] = 2
            with self.assertRaises(SkipTrainError):
                proc.process(data)

        catalog.clear()
        for ctg in ['XGM', 'DSSC', 'JungFrau']:
            item = SourceItem(ctg, 'device1', [], 'property1', None, (-1, 1), KaraboType.PIPELINE_DATA)
            catalog.add_item(item)
            src = f"{item.name} {item.property}"
            meta[src] = {'train_id': 12346}
            raw[src] = 2
            # it will not raise for other sources
            proc.process(data)
