import unittest

from PyQt5.QtTest import QSignalSpy, QTest
from PyQt5.QtCore import Qt

from karaboFAI.gui.ctrl_widgets import DataSourceWidget
from karaboFAI.gui.ctrl_widgets.data_source_widget import (
    DataSourceTreeItem, DataSourceTreeModel, DataSourceListModel
)

from karaboFAI.logger import logger

logger.setLevel("CRITICAL")


class TestDataSourceWidget(unittest.TestCase):
    def testDataSourceTreeItem(self):
        pass

    def testDataSourceTreeModel(self):
        pass
