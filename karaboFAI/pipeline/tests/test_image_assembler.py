import unittest

from karaboFAI.pipeline.image_assembler import ImageAssemblerFactory


class TestAgipdAssembler(unittest.TestCase):
    def setUp(self):
        self._assembler = ImageAssemblerFactory.create("AGIPD")


class TestLpdAssembler(unittest.TestCase):
    def setUp(self):
        self._assembler = ImageAssemblerFactory.create("LPD")


class TestJungfrauAssembler(unittest.TestCase):
    def setUp(self):
        self._assembler = ImageAssemblerFactory.create("JungFrau")


class TestFastccdAssembler(unittest.TestCase):
    def setUp(self):
        self._assembler = ImageAssemblerFactory.create("FastCCD")
