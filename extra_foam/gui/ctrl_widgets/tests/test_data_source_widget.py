import unittest
from unittest.mock import patch, PropertyMock
import time

from PyQt5 import QtTest
from PyQt5.QtWidgets import QWidget
from PyQt5.QtCore import Qt

from extra_foam.logger import logger
from extra_foam.gui import mkQApp
from extra_foam.config import config, ConfigWrapper
from extra_foam.gui.ctrl_widgets.data_source_widget import DataSourceWidget
from extra_foam.services import start_redis_server
from extra_foam.processes import ProcessInfoList, wait_until_redis_shutdown
from extra_foam.database import Metadata, MetaProxy

from extra_foam.pipeline.tests import _RawDataMixin

app = mkQApp()

logger.setLevel("CRITICAL")


class TestDataSourceWidget(unittest.TestCase, _RawDataMixin):
    @classmethod
    def setUpClass(cls):
        class DummyParent(QWidget):
            def createCtrlWidget(self, widget):
                pass

        cls._dummy = DummyParent()

        start_redis_server()

    @classmethod
    def tearDownClass(cls):
        wait_until_redis_shutdown()

    @patch.dict(config._data, {"DETECTOR": "DSSC", "TOPIC": "SCS", "SOURCE_EXPIRATION_TIMER": 0.01})
    def testUpdateAvailableSources(self):
        widget = DataSourceWidget(parent=self._dummy)
        list_model = widget._avail_src_model
        tree_model = widget._src_tree_model
        proxy = widget._mon

        # test default
        widget.updateAvailableSources()
        self.assertListEqual([], list_model._sources)

        # test new sources
        _, meta = self._gen_kb_data(1234, {"abc": [('ppt', 1)], "efg": [('ppt', 2)]})
        proxy.set_available_sources(meta, ["efg ppt"])
        widget.updateAvailableSources()
        self.assertListEqual(["abc", "efg"], list_model._sources)
        self.assertListEqual(["efg ppt"], tree_model._matched_srcs)

        # test old sources do not exist when new sources are set
        _, meta = self._gen_kb_data(1234, {"cba": [('ppt', 1)], "gfe": [('ppt', 2)]})
        proxy.set_available_sources(meta, ["gfe ppt"])
        widget.updateAvailableSources()
        self.assertListEqual(["abc", "cba", "efg", "gfe"], list_model._sources)
        self.assertListEqual(["efg ppt", "gfe ppt"], tree_model._matched_srcs)

        # test expiration
        time.sleep(0.05)
        widget.updateAvailableSources()
        self.assertListEqual([], list_model._sources)
        self.assertListEqual([], tree_model._matched_srcs)

    @patch.dict(config._data, {"DETECTOR": "DSSC", "TOPIC": "SCS"})
    @patch("extra_foam.gui.ctrl_widgets.data_source_widget.list_foam_processes")
    def testProcessMonitorMV(self, query):
        widget = DataSourceWidget(parent=self._dummy)
        view = widget._process_mon_view

        query.return_value = [ProcessInfoList(
            name='ZeroMQ',
            foam_name='foam name',
            foam_type='foam type',
            pid=1234,
            status='zombie'
        )]
        widget.updateProcessInfo()
        self.assertEqual("ZeroMQ", view.model().index(0, 0).data())
        self.assertEqual(1234, view.model().index(0, 3).data())
        self.assertEqual("zombie", view.model().index(0, 4).data())
        # test old text will be removed
        query.return_value = [ProcessInfoList(
            name='kafka',
            foam_name='foam name',
            foam_type='foam type',
            pid=1234,
            status='sleeping'
        )]
        widget.updateProcessInfo()
        self.assertEqual("kafka", view.model().index(0, 0).data())
        self.assertEqual(1234, view.model().index(0, 3).data())
        self.assertEqual("sleeping", view.model().index(0, 4).data())
        self.assertIsNone(view.model().index(1, 4).data())

    @patch("extra_foam.config.ConfigWrapper.appendix_streamers", new_callable=PropertyMock)
    def testConnectView(self, streamers):
        from extra_foam.config import _Config
        StreamerEndpointItem = _Config.StreamerEndpointItem

        streamers.return_value = [
            StreamerEndpointItem("s1", 1, "127.0.0.1", 12345),
            StreamerEndpointItem("s2", 1, "127.0.0.2", 12346),
        ]
        widget = DataSourceWidget(parent=self._dummy)
        model = widget._con_model

        self.assertEqual([False, 's1', model._getSourceTypeString(1), '127.0.0.1', '12345'],
                         model._connections[1])
        self.assertEqual([False, 's2', model._getSourceTypeString(1), '127.0.0.2', '12346'],
                         model._connections[2])

        # TODO: test with QtTest
        model._connections[1][0] = True
        model._connections[2][0] = True

        # Test duplicated endpoints
        con2_backup = model._connections[2]
        model._connections[2] = model._connections[1]
        with patch("extra_foam.gui.ctrl_widgets.data_source_widget.logger.error") as mocked_error:
            self.assertFalse(widget.updateMetaData())
            mocked_error.assert_called_once()
        model._connections[2] = con2_backup

        # Test different source types
        model._connections[2][2] = model._getSourceTypeString(0)
        with patch("extra_foam.gui.ctrl_widgets.data_source_widget.logger.error") as mocked_error:
            self.assertFalse(widget.updateMetaData())
            mocked_error.assert_called_once()

        # Test the connections are set in Redis
        model._connections[2][2] = model._getSourceTypeString(1)
        self.assertTrue(widget.updateMetaData())
        cons = MetaProxy().hget_all(Metadata.CONNECTION)
        # FIXME: the port '4501' depends on the source type
        for addr, tp in zip(['tcp://127.0.0.1:4501', 'tcp://127.0.0.1:12345', 'tcp://127.0.0.2:12346'],
                            ['1', '1', '1']):
            self.assertIn(addr, cons)
            self.assertEqual(tp, cons[addr])

        # Modify connections
        model._connections[1][0] = False
        model._connections[2][2:] = [model._getSourceTypeString(1), '127.0.0.3', '12356']
        self.assertTrue(widget.updateMetaData())
        cons = MetaProxy().hget_all(Metadata.CONNECTION)
        # FIXME: the port '4501' depends on the source type
        for addr, tp in zip(['tcp://127.0.0.1:4501', 'tcp://127.0.0.3:12356'], ['1', '1']):
            self.assertIn(addr, cons)
            self.assertEqual(tp, cons[addr])

    @patch.dict(config._data, {"PULSE_RESOLVED": True})
    @patch.object(ConfigWrapper, "pipeline_sources", new_callable=PropertyMock)
    @patch.object(ConfigWrapper, "control_sources", new_callable=PropertyMock)
    def testDataSourceTreeModelPs(self, control_sources, pipeline_sources):
        from PyQt5.QtCore import QModelIndex

        pipeline_sources.return_value = {
            'DSSC': {"A": ['a'], 'B': ['b']},
            'XGM': {"XA": ['intensity']},
        }

        control_sources.return_value = {
            'XGM': {"XA": ['flux', 'xpos']},
            'MONOCHROMATOR': {"MONO": ['e']},
            'MOTOR': {"MA": ['p'], "MB": ['p'], "MC": ['p']}
        }

        widget = DataSourceWidget(parent=self._dummy)
        model = widget._src_tree_model

        # ---------
        # test root
        # ---------
        self.assertEqual('DSSC', model.index(0, 2, QModelIndex()).data())
        self.assertEqual('XGM', model.index(1, 2, QModelIndex()).data())
        self.assertEqual('MONOCHROMATOR', model.index(2, 2, QModelIndex()).data())
        self.assertEqual('MOTOR', model.index(3, 2, QModelIndex()).data())
        self.assertEqual(config["SOURCE_USER_DEFINED_CATEGORY"],
                         model.index(4, 2, QModelIndex()).data())
        self.assertFalse(model.index(5, 2, QModelIndex()).isValid())
        self.assertEqual(5, model.rowCount(QModelIndex()))
        self.assertEqual(6, model.columnCount(QModelIndex()))

        # -----------------------
        # test exclusive category
        # -----------------------

        dssc_ctg = model.index(0, 0, QModelIndex())
        self.assertEqual(False, model.index(0, 0, dssc_ctg).data())
        self.assertEqual(1, model.index(0, 1, dssc_ctg).data())
        self.assertEqual('A', model.index(0, 2, dssc_ctg).data())
        self.assertEqual('a', model.index(0, 3, dssc_ctg).data())
        self.assertEqual(':', model.index(0, 4, dssc_ctg).data())
        self.assertEqual('', model.index(0, 5, dssc_ctg).data())

        spy = QtTest.QSignalSpy(model.source_item_toggled_sgn)
        # first check since it is not allowed to modify in unchecked state
        model._matched_srcs = ['A a']
        # FIXME: 'setData' does not care about which the column index with CheckStateRole.
        model.setData(model.index(0, 0, dssc_ctg), True, Qt.CheckStateRole)
        self.assertEqual(1, len(spy))
        self.assertTrue(spy[0][0])
        self.assertTupleEqual(('DSSC', 'A', '[]', 'a', '[None, None]', '', 1), spy[0][1])
        # check availability
        self.assertTrue(model.data(model.index(0, 0, dssc_ctg), Qt.DisplayRole))

        spy = QtTest.QSignalSpy(model.source_item_toggled_sgn)
        # change device ID
        model.setData(model.index(0, 2, dssc_ctg), 'A+', Qt.EditRole)
        self.assertEqual(2, len(spy))
        # check signal for deleting old source
        self.assertFalse(spy[0][0])
        self.assertEqual('A a', spy[0][1])
        # check signal for adding new source
        self.assertTrue(spy[1][0])
        self.assertTupleEqual(('DSSC', 'A+', '[]', 'a', '[None, None]', '', 1), spy[1][1])

        spy = QtTest.QSignalSpy(model.source_item_toggled_sgn)
        # change property
        model.setData(model.index(0, 3, dssc_ctg), 'a-', Qt.EditRole)
        self.assertEqual(2, len(spy))
        # check signal for deleting old source
        self.assertFalse(spy[0][0])
        self.assertEqual('A+ a', spy[0][1])
        # check signal for adding new source
        self.assertTrue(spy[1][0])
        self.assertTupleEqual(('DSSC', 'A+', '[]', 'a-', '[None, None]', '', 1), spy[1][1])

        spy = QtTest.QSignalSpy(model.source_item_toggled_sgn)
        # change slicer
        model.setData(model.index(0, 4, dssc_ctg), '::2', Qt.EditRole)
        self.assertEqual(2, len(spy))
        # check signal for deleting old source
        self.assertFalse(spy[0][0])
        # deleting does not check slicer
        # check signal for adding new source
        self.assertTrue(spy[1][0])
        self.assertTupleEqual(('DSSC', 'A+', '[]', 'a-', '[None, None, 2]', '', 1), spy[1][1])

        spy = QtTest.QSignalSpy(model.source_item_toggled_sgn)
        # change a DSSC source
        model._matched_srcs = ['B b']
        # FIXME: 'setData' does not care about which the column index with CheckStateRole.
        model.setData(model.index(1, 0, dssc_ctg), True, Qt.CheckStateRole)
        self.assertEqual(2, len(spy))
        # check signal for deleting old source ('DSSC' is an exclusive category)
        self.assertFalse(spy[0][0])
        self.assertEqual('A+ a-', spy[0][1])
        # check signal for adding new source
        self.assertTrue(spy[1][0])
        self.assertTupleEqual(('DSSC', 'B', '[]', 'b', '[None, None]', '', 1), spy[1][1])
        # check availability
        self.assertFalse(model.data(model.index(0, 0, dssc_ctg), Qt.DisplayRole))
        self.assertTrue(model.data(model.index(1, 0, dssc_ctg), Qt.DisplayRole))

        spy = QtTest.QSignalSpy(model.source_item_toggled_sgn)
        # uncheck a DSSC source
        # FIXME: 'setData' does not care about which the column index with CheckStateRole.
        model.setData(model.index(1, 0, dssc_ctg), False, Qt.CheckStateRole)
        self.assertEqual(1, len(spy))
        # check signal for deleting old source
        self.assertFalse(spy[0][0])
        self.assertEqual('B b', spy[0][1])
        # check availability
        self.assertFalse(model.data(model.index(1, 0, dssc_ctg), Qt.DisplayRole))

        # ---------------------
        # test mixed category
        # ---------------------
        xgm_ctg = model.index(1, 0, QModelIndex())
        self.assertEqual(False, model.index(0, 0, xgm_ctg).data())
        self.assertEqual(1, model.index(0, 1, xgm_ctg).data())
        self.assertEqual('XA', model.index(0, 2, xgm_ctg).data())
        self.assertEqual('intensity', model.index(0, 3, xgm_ctg).data())
        self.assertEqual(':', model.index(0, 4, xgm_ctg).data())
        self.assertEqual('-inf, inf', model.index(0, 5, xgm_ctg).data())
        self.assertEqual(False, model.index(0, 0, xgm_ctg).data())
        self.assertEqual(0, model.index(1, 1, xgm_ctg).data())
        self.assertEqual('XA', model.index(1, 2, xgm_ctg).data())
        self.assertEqual('flux', model.index(1, 3, xgm_ctg).data())
        self.assertEqual('', model.index(1, 4, xgm_ctg).data())
        self.assertEqual('-inf, inf', model.index(1, 5, xgm_ctg).data())
        self.assertEqual(False, model.index(0, 0, xgm_ctg).data())
        self.assertEqual(0, model.index(2, 1, xgm_ctg).data())
        self.assertEqual('XA', model.index(2, 2, xgm_ctg).data())
        self.assertEqual('xpos', model.index(2, 3, xgm_ctg).data())
        self.assertEqual('', model.index(2, 4, xgm_ctg).data())
        self.assertEqual('-inf, inf', model.index(2, 5, xgm_ctg).data())

        spy = QtTest.QSignalSpy(model.source_item_toggled_sgn)
        # FIXME: 'setData' does not care about which the column index with CheckStateRole.
        model.setData(model.index(2, 0, xgm_ctg), True, Qt.CheckStateRole)
        self.assertEqual(1, len(spy))
        self.assertTupleEqual(('XGM', 'XA', '[]', 'xpos', '', '(-inf, inf)', 0), spy[0][1])

        spy = QtTest.QSignalSpy(model.source_item_toggled_sgn)
        # FIXME: 'setData' does not care about which the column index with CheckStateRole.
        model.setData(model.index(1, 0, xgm_ctg), True, Qt.CheckStateRole)
        self.assertEqual(1, len(spy))
        self.assertTupleEqual(('XGM', 'XA', '[]', 'flux', '', '(-inf, inf)', 0), spy[0][1])

        spy = QtTest.QSignalSpy(model.source_item_toggled_sgn)
        # FIXME: 'setData' does not care about which the column index with CheckStateRole.
        model.setData(model.index(0, 0, xgm_ctg), True, Qt.CheckStateRole)
        self.assertEqual(1, len(spy))
        self.assertTupleEqual(('XGM', 'XA', '[]', 'intensity', '[None, None]', '(-inf, inf)', 1), spy[0][1])

        spy = QtTest.QSignalSpy(model.source_item_toggled_sgn)
        # FIXME: 'setData' does not care about which the column index with CheckStateRole.
        model.setData(model.index(2, 0, xgm_ctg), False, Qt.CheckStateRole)
        self.assertEqual(1, len(spy))
        self.assertEqual('XA xpos', spy[0][1])

        spy = QtTest.QSignalSpy(model.source_item_toggled_sgn)
        # FIXME: 'setData' does not care about which the column index with CheckStateRole.
        model.setData(model.index(0, 0, xgm_ctg), False, Qt.CheckStateRole)
        self.assertEqual(1, len(spy))
        self.assertEqual('XA intensity', spy[0][1])

        spy = QtTest.QSignalSpy(model.source_item_toggled_sgn)
        # change slicer
        model.setData(model.index(1, 5, xgm_ctg), '-1, 1', Qt.EditRole)
        self.assertEqual(2, len(spy))
        # delete old source
        self.assertFalse(spy[0][0])
        # deleting does not check range
        # add new source
        self.assertTrue(spy[1][0])
        self.assertTupleEqual(('XGM', 'XA', '[]', 'flux', '', '(-1.0, 1.0)', 0), spy[1][1])

    @patch.dict(config._data, {"PULSE_RESOLVED": False})
    @patch.object(ConfigWrapper, "pipeline_sources", new_callable=PropertyMock)
    @patch.object(ConfigWrapper, "control_sources", new_callable=PropertyMock)
    def testDataSourceTreeModelTs(self, control_sources, pipeline_sources):
        from PyQt5.QtCore import QModelIndex

        pipeline_sources.return_value = {
            'JungFrau': {"A": ['a'], 'B': ['b']},
        }

        control_sources.return_value = {
            'MOTOR': {"MA": ['p'], "MB": ['p'], "MC": ['p']}
        }

        widget = DataSourceWidget(parent=self._dummy)
        model = widget._src_tree_model

        jf_ctg = model.index(0, 0, QModelIndex())
        self.assertEqual(False, model.index(0, 0, jf_ctg).data())
        self.assertEqual(1, model.index(0, 1, jf_ctg).data())
        self.assertEqual('A', model.index(0, 2, jf_ctg).data())
        self.assertEqual('a', model.index(0, 3, jf_ctg).data())
        # no slicer for train-resolved detectors
        self.assertEqual('', model.index(0, 4, jf_ctg).data())
        self.assertEqual('', model.index(0, 5, jf_ctg).data())
