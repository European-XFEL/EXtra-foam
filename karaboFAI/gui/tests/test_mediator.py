import unittest

from karaboFAI.gui.mediator import Mediator


class TestMediator(unittest.TestCase):

    def testSingleton(self):
        m1 = Mediator()
        m2 = Mediator()
        self.assertEqual(m1, m2)
