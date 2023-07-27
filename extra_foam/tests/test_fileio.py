"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import os.path as osp
import unittest
from unittest.mock import patch
import tempfile

import numpy as np

from extra_foam.logger import logger
from extra_foam.file_io import read_image, read_numpy_array, write_image, write_numpy_array

logger.setLevel("CRITICAL")


class TestFileIO(unittest.TestCase):
    def testReadImage(self):

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
            with self.assertRaisesRegex(ValueError, 'Could not find a backend'):
                read_image(fp.name)

    def testWriteImage(self):
        img = np.ones((2, 3), dtype=np.float32)

        # test write empty input
        with self.assertRaisesRegex(ValueError, 'Empty filepath!'):
            write_image('', img)

        # test write invalid file format
        with tempfile.NamedTemporaryFile(suffix='.txt') as fp:
            with self.assertRaisesRegex(ValueError, 'Could not find a backend'):
                write_image(fp.name, img)

        # test read and write valid file formats
        self._assert_write_read(img, '.tif')
        self._assert_write_read(img, '.npy')

        # Test writing .png files. We don't test reading because Pillow can't
        # write floating-point PNG images.
        self._assert_write_read(img, '.png', scale=255, check=False)

    def _assert_write_read(self, img, file_type, *, scale=1, check=True):
        with tempfile.NamedTemporaryFile(suffix=file_type) as fp:
            write_image(fp.name, img)

            ref = read_image(fp.name)
            if check:
                np.testing.assert_array_equal(scale * img, ref)

    def testReadNumpyArray(self):
        # test wrong file suffix
        with patch('numpy.load', return_value=np.ones((2, 2, 2, 2))):
            with self.assertRaisesRegex(ValueError, 'Input must be a .npy file'):
                read_numpy_array('abc')

        # test wrong dimension
        with patch('numpy.load', return_value=np.ones((2, 2, 2, 2))):
            with self.assertRaisesRegex(ValueError, 'Expect array with dimensions (.*?): actual 4'):
                read_numpy_array('abc.npy', dimensions=(2, 3))
        with patch('numpy.load', return_value=np.ones(2)):
            with self.assertRaisesRegex(ValueError, 'Expect array with dimensions (.*?): actual 1'):
                read_numpy_array('abc.npy', dimensions=(2,))

        # test valid data
        with patch('numpy.load', return_value=np.ones((2, 2, 2, 2))):
            read_numpy_array('abc.npy')

        for arr in [np.ones([2, 2]), np.ones([4, 2, 2], dtype=np.float32)]:
            fp = tempfile.NamedTemporaryFile(suffix='.npy')
            np.save(fp.name, arr)
            ret = read_numpy_array(fp.name, dimensions=(2, 3))
            np.testing.assert_array_equal(arr, ret)

        # file does not have suffix '.npy'
        with self.assertRaises(ValueError):
            for arr in [np.ones([2, 2]), np.ones([4, 2, 2], dtype=np.float32)]:
                fp = tempfile.NamedTemporaryFile()
                np.save(fp.name, arr)
                read_numpy_array(fp.name)

    def testWriteNumpyArray(self):
        arr = np.ones((2, 3, 4), dtype=np.float32)

        # test write empty input
        with self.assertRaisesRegex(ValueError, 'Empty filepath!'):
            write_numpy_array('', arr)

        # test read and write valid file formats
        with tempfile.NamedTemporaryFile(suffix=".npy") as fp:
            write_numpy_array(osp.splitext(fp.name)[0], arr)
            # check ".npy" is appended to the saved file
            ref = read_numpy_array(fp.name)
            np.testing.assert_array_equal(arr, ref)
