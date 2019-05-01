import unittest

from karaboFAI.services import FaiServer
from karaboFAI.gui.windows.base_window import AbstractWindow, SingletonWindow


class TestSingletonWindow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        FaiServer()
        SingletonWindow._instances.clear()

    @SingletonWindow
    class FooWindow(AbstractWindow):
        pass

    def test_singleton(self):
        win1 = self.FooWindow()
        win2 = self.FooWindow()
        self.assertEqual(win1, win2)

        self.assertEqual(1, len(SingletonWindow._instances))
        key = list(SingletonWindow._instances.keys())[0]
        self.assertEqual('FooWindow', key.__name__)
