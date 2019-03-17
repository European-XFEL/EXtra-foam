import unittest

from karaboFAI.gui.main_gui import Mediator


class TestMediator(unittest.TestCase):

    def testSingleton(self):
        m1 = Mediator()
        m2 = Mediator()
        self.assertEqual(m1, m2)
