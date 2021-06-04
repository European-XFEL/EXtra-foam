"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import unittest

import numpy as np
from extra_foam.serialization import (
    serialize_image, deserialize_image, serialize_images, deserialize_images
)


class TestSerialization(unittest.TestCase):
    def testRaises(self):
        with self.assertRaises(TypeError):
            serialize_image([1, 2, 3])
        with self.assertRaises(ValueError):
            serialize_image(np.array([1, 2, 3]))

        with self.assertRaises(TypeError):
            serialize_images([[1, 2, 3], [4, 5, 6]])
        with self.assertRaises(ValueError):
            serialize_images(np.ones((2, 2)))

    def testGeneral(self):
        # test serializing and deserializing a single image
        orig_img = np.array([[1, 2, 3],
                             [4, 5, 6]], dtype=np.float32)

        img_bytes = serialize_image(orig_img)
        img = deserialize_image(img_bytes)

        np.testing.assert_array_equal(orig_img, img)
        self.assertEqual(orig_img.shape, img.shape)
        self.assertEqual(np.float32, img.dtype)

        # test serializing and deserializing a mask
        orig_mask = np.array([[1, 1, 0], [0, 1, 1]], dtype=bool)

        mask_bytes = serialize_image(orig_mask, is_mask=True)
        mask = deserialize_image(mask_bytes, is_mask=True)

        np.testing.assert_array_equal(orig_mask, mask)
        self.assertEqual(orig_mask.shape, mask.shape)
        self.assertEqual(bool, mask.dtype)

        # test casting array into a mask
        orig_mask = np.array([[1., 1., 0], [0, -1., 1.]], dtype=np.float32)

        mask_bytes = serialize_image(orig_mask, is_mask=True)
        mask = deserialize_image(mask_bytes, is_mask=True)

        np.testing.assert_array_equal(orig_mask.astype(bool), mask)
        self.assertEqual(orig_mask.shape, mask.shape)
        self.assertEqual(bool, mask.dtype)

        # test serializing and deserializing a group of images
        orig_imgs = np.array([[[1, 2], [3, 4]],
                              [[1, 2], [3, 4]]], dtype=np.int32)

        imgs_bytes = serialize_images(orig_imgs)
        imgs = deserialize_images(imgs_bytes, dtype=np.int32)

        np.testing.assert_array_equal(orig_imgs, imgs)
        self.assertEqual(orig_imgs.shape, imgs.shape)
        self.assertEqual(np.float32, img.dtype)
