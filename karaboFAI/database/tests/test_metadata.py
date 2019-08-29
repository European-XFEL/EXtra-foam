import unittest

from karaboFAI.database.metadata import MetaMetadata


class TestMetadata(unittest.TestCase):
    def testMetaMetadata(self):
        class Dummy(metaclass=MetaMetadata):
            SESSION = "meta:session"
            DATA_SOURCE = "meta:source"
            ANALYSIS_TYPE = "meta:analysis_type"
            GLOBAL_PROC = "meta:proc:global"
            IMAGE_PROC = "meta:proc:image"
            GEOMETRY_PROC = "meta:proc:geometry"

        self.assertListEqual(['global', 'image', 'geometry'], Dummy.processors)
