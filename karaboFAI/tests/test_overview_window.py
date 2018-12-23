import unittest

from karaboFAI.main_gui import MainGUI
from karaboFAI.widgets.pyqtgraph import mkQApp, QtGui, QtCore
from karaboFAI.windows import OverviewWindow

mkQApp()


class Dummy(QtGui.QMainWindow):

    diff_integration_range_sgn = QtCore.pyqtSignal(float, float)
    normalization_range_sgn = QtCore.pyqtSignal(float, float)
    mask_range_sgn = QtCore.pyqtSignal(float, float)

    def __init__(self):
        super().__init__()

    def registerPlotWindow(self, instance):
        pass

    def updateSharedParameters(self):
        pass


class TestOverviewWindow(unittest.TestCase):

    def setUp(self):
        self._data = []

        self._parent = Dummy()

    def testOpenOverviewWindow(self):
        win = OverviewWindow(MainGUI.Data4Visualization(), parent=self._parent)
