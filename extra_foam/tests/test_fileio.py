"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import unittest
from unittest.mock import patch
import os
import tempfile

import numpy as np

from extra_foam.config import config
from extra_foam.logger import logger
from extra_foam.file_io import read_numpy_array, read_image, write_image

logger.setLevel("CRITICAL")


class TestFileIO(unittest.TestCase):
    def testReadImage(self):

        # test read empty input
        with self.assertRaisesRegex(ValueError, 'Please specify'):
            read_image('')

        # test wrong shape
        with patch('imageio.imread', return_value=np.ones((2, 2))):
            with self.assertRaisesRegex(ValueError, 'Shape of'):
                read_image('abc', expected_shape=(3, 2))

        # test wrong dimension
        with patch('imageio.imread', return_value=np.ones((2, 2, 2))):
            with self.assertRaisesRegex(ValueError, '2D array'):
                read_image('abc')

        # test read invalid file format
        with tempfile.NamedTemporaryFile(suffix='.txt') as fp:
            with self.assertRaisesRegex(ValueError, 'Could not find a format'):
                read_image(fp.name)

    def testWriteImage(self):
        # test write empty input
        with self.assertRaisesRegex(ValueError, 'Please specify'):
            read_image('')

        # test write invalid file format
        with tempfile.NamedTemporaryFile(suffix='.txt') as fp:
            with self.assertRaisesRegex(ValueError, 'Could not find a format'):
                read_image(fp.name)

        # test read and write valid file formats
        self._assert_write_read('.tif')
        self._assert_write_read('.npy')

        # test read and write .png file
        self._assert_write_read('.png', scale=255)

    def _assert_write_read(self, file_type, *, scale=1):
        img = np.ones((2, 3), dtype=np.float32)
        with tempfile.NamedTemporaryFile(suffix=file_type) as fp:
            write_image(img, fp.name)
            ref = read_image(fp.name)
            np.testing.assert_array_equal(scale * img, ref)

    def testReadNumpyArray(self):
        # test read empty input
        with self.assertRaisesRegex(ValueError, 'Please specify'):
            read_image('')

        # test wrong dimension
        with patch('numpy.load', return_value=np.ones((2, 2, 2, 2))):
            with self.assertRaisesRegex(ValueError, 'Expect array with dimensions (.*?): actual 4'):
                read_numpy_array('abc', dimensions=(2, 3))
        with patch('numpy.load', return_value=np.ones(2)):
            with self.assertRaisesRegex(ValueError, 'Expect array with dimensions (.*?): actual 1'):
                read_numpy_array('abc', dimensions=(2,))

        # test valid data
        with patch('numpy.load', return_value=np.ones((2, 2, 2, 2))):
            read_numpy_array('abc')

        for arr in [np.ones([2, 2]), np.ones([4, 2, 2], dtype=np.float32)]:
            fp = tempfile.NamedTemporaryFile(suffix='.npy')
            np.save(fp.name, arr)
            ret = read_numpy_array(fp, dimensions=(2, 3))
            np.testing.assert_array_equal(arr, ret)

        # file does not have suffix '.npy'
        with self.assertRaises(ValueError):
            for arr in [np.ones([2, 2]), np.ones([4, 2, 2], dtype=np.float32)]:
                fp = tempfile.NamedTemporaryFile()
                np.save(fp.name, arr)
                read_numpy_array(fp)
