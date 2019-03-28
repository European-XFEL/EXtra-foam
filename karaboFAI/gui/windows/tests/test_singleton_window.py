import unittest

from karaboFAI.gui.windows.base_window import AbstractWindow, SingletonWindow


class TestSingletonWindow(unittest.TestCase):

    @SingletonWindow
    class FooWindow(AbstractWindow):
        pass

    def test_singleton(self):
        win1 = self.FooWindow()
        win2 = self.FooWindow()
        self.assertEqual(win1, win2)
