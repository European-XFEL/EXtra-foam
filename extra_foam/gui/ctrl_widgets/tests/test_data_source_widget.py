import unittest
from unittest.mock import patch
import time

from PyQt5.QtWidgets import QWidget

from extra_foam.logger import logger
from extra_foam.gui import mkQApp
from extra_foam.config import config
from extra_foam.gui.ctrl_widgets.data_source_widget import DataSourceWidget
from extra_foam.services import start_redis_server
from extra_foam.processes import ProcessInfoList, wait_until_redis_shutdown

app = mkQApp()

logger.setLevel("CRITICAL")


class TestDataSourceWidget(unittest.TestCase):
    class DummyParent(QWidget):
        def createCtrlWidget(self, widget):
            pass

    @classmethod
    def setUpClass(cls):
        start_redis_server()

    @classmethod
    def tearDownClass(cls):
        wait_until_redis_shutdown()

    @patch.dict(config._data, {"DETECTOR": "DSSC", "TOPIC": "SCS", "SOURCES_EXPIRATION_TIME": 10})
    def testDataSourceListMV(self):
        parent = self.DummyParent()
        widget = DataSourceWidget(parent)
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
        parent = self.DummyParent()
        widget = DataSourceWidget(parent)
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

    def testDataSourceTreeItem(self):
        pass

    def testDataSourceTreeModel(self):
        pass
