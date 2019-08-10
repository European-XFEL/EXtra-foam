"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

Unittest for file IO.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import unittest
from unittest.mock import patch
import os
import tempfile

import numpy as np

from karaboFAI.config import config
from karaboFAI.logger import logger
from karaboFAI.file_io import read_image, write_image

logger.setLevel("CRITICAL")


class TestFileIO(unittest.TestCase):
    def testReadImage(self):

        # test empty input
        with self.assertRaisesRegex(ValueError, 'Please specify the reference'):
            read_image('')

        # test wrong shape
        with patch('imageio.imread', return_value=np.ones((2, 2))):
            with self.assertRaisesRegex(ValueError, 'Shape of'):
                read_image('abc', expected_shape=(3, 2))

        # test dtype
        with patch('imageio.imread', return_value=np.ones((3, 2))):
            img = read_image('abc')
            self.assertEqual(img.dtype, config['IMAGE_DTYPE'])
            self.assertEqual((3, 2), img.shape)

    def testWriteImage(self):
        img = np.ones((2, 2))
        fp, filepath = tempfile.mkstemp(suffix='.tiff')
        write_image(img, filepath)
        os.close(fp)
        os.remove(filepath)
