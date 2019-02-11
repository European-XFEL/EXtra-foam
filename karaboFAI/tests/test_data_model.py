import unittest

from karaboFAI.data_processing.data_model import (
    AbstractData, ProcessedData, TrainData
)


class TestDataModel(unittest.TestCase):
    def test_TrainData(self):
        class Dummy(AbstractData):
            values = TrainData()

        dm = Dummy()

        dm.values = (1, 'a')
        dm.values = (2, 'b')
        tids, values, _ = dm.values
        self.assertListEqual([1, 2], tids)
        self.assertListEqual(['a', 'b'], values)

        dm.values = (3, 'c')
        tids, values, _ = dm.values
        self.assertListEqual([1, 2, 3], tids)
        self.assertListEqual(['a', 'b', 'c'], values)

        del dm.values
        tids, values, _ = dm.values
        self.assertListEqual([2, 3], tids)
        self.assertListEqual(['b', 'c'], values)

        Dummy.clear()
        tids, values, _ = dm.values
        self.assertListEqual([], tids)
        self.assertListEqual([], values)

    def test_ProcessedData(self):
        data = ProcessedData(1234)
        self.assertEqual(1234, data.tid)

        data.roi.values1 = (1234, None)
        tids, values, _ = data.roi.values1
        self.assertListEqual([1234], tids)
        self.assertListEqual([None], values)

        data.roi.values1 = (1235, 2.0)
        tids, values, _ = data.roi.values1
        self.assertListEqual([1234, 1235], tids)
        self.assertListEqual([None, 2.0], values)

    def test_CorrelationData(self):
        data = ProcessedData(-1)

        data.add_correlator(0, "device1", "property1")
        data.correlation.param0 = (10, 20)
        data.correlation.param0 = (11, 22)
        fom, corr, info = data.correlation.param0
        self.assertListEqual([10, 11], fom)
        self.assertListEqual([20, 22], corr)
        self.assertEqual("device1", info["device_id"])
        self.assertEqual("property1", info["property"])

        data.add_correlator(1, "device2", "property2")
        data.correlation.param1 = (100, 200)
        data.correlation.param1 = (110, 220)
        fom, corr, info = data.correlation.param1
        self.assertListEqual([100, 110], fom)
        self.assertListEqual([200, 220], corr)
        self.assertEqual("device2", info["device_id"])
        self.assertEqual("property2", info["property"])
        # check that param0 remains unchanged
        fom, corr, info = data.correlation.param0
        self.assertListEqual([10, 11], fom)
        self.assertListEqual([20, 22], corr)
        self.assertEqual("device1", info["device_id"])
        self.assertEqual("property1", info["property"])

        # test clear history
        ProcessedData.clear_correlation_hist()
        fom, corr, info = data.correlation.param0
        self.assertListEqual([], fom)
        self.assertListEqual([], corr)
        self.assertEqual("device1", info["device_id"])
        self.assertEqual("property1", info["property"])
        fom, corr, info = data.correlation.param1
        self.assertListEqual([], fom)
        self.assertListEqual([], corr)
        self.assertEqual("device2", info["device_id"])
        self.assertEqual("property2", info["property"])

        # when device_id or property is empty, the corresponding 'param'
        # will be removed
        data.add_correlator(0, "", "property2")
        with self.assertRaises(AttributeError):
            data.correlation.param0

        data.add_correlator(1, "device2", "")
        with self.assertRaises(AttributeError):
            data.correlation.param1
