import unittest
from unittest.mock import patch, PropertyMock
import time

from PyQt5 import QtTest
from PyQt5.QtWidgets import QWidget

from extra_foam.logger import logger
from extra_foam.gui import mkQApp
from extra_foam.config import config
from extra_foam.gui.ctrl_widgets.data_source_widget import DataSourceWidget
from extra_foam.services import start_redis_server
from extra_foam.processes import ProcessInfoList, wait_until_redis_shutdown
from extra_foam.database import Metadata, MetaProxy

app = mkQApp()

logger.setLevel("CRITICAL")


class TestDataSourceWidget(unittest.TestCase):
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

    @patch.dict(config._data, {"DETECTOR": "DSSC", "TOPIC": "SCS", "SOURCE_EXPIRATION_TIMER": 10})
    def testDataSourceListMV(self):
        widget = DataSourceWidget(parent=self._dummy)
        model = widget._avail_src_model
        proxy = widget._mon

        # test default
        widget.updateSourceList()
        self.assertListEqual([], model._sources)

        # test new sources
        proxy.set_available_sources({"abc": "1234567", "efg": "234567"})
        widget.updateSourceList()
        self.assertListEqual(["abc", "efg"], model._sources)

        # test old sources do not exist when new sources are set
        proxy.set_available_sources({"cba": "1234567", "gfe": "234567"})
        widget.updateSourceList()
        self.assertListEqual(["cba", "gfe"], model._sources)

        # test expiration
        time.sleep(0.020)
        widget.updateSourceList()
        self.assertListEqual([], model._sources)

    @patch.dict(config._data, {"DETECTOR": "DSSC", "TOPIC": "SCS"})
    @patch("extra_foam.gui.ctrl_widgets.data_source_widget.list_foam_processes")
    def testProcessMonitorMV(self, query):
        widget = DataSourceWidget(parent=self._dummy)
        view = widget._process_mon_view
        model = widget._process_mon_model

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
        view = widget._con_view
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
        for addr, tp in zip(['tcp://127.0.0.1:45454', 'tcp://127.0.0.1:12345', 'tcp://127.0.0.2:12346'],
                            ['1', '1', '1']):
            self.assertIn(addr, cons)
            self.assertEqual(tp, cons[addr])

        # Modify connections
        model._connections[1][0] = False
        model._connections[2][2:] = [model._getSourceTypeString(1), '127.0.0.3', '12356']
        self.assertTrue(widget.updateMetaData())
        cons = MetaProxy().hget_all(Metadata.CONNECTION)
        for addr, tp in zip(['tcp://127.0.0.1:45454', 'tcp://127.0.0.3:12356'], ['1', '1']):
            self.assertIn(addr, cons)
            self.assertEqual(tp, cons[addr])

    def testDataSourceTreeItem(self):
        pass

    def testDataSourceTreeModel(self):
        pass
