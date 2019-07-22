"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

Unittest for serialization.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import unittest

import numpy as np
from karaboFAI.serialization import (
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

        # test serializing and deserializing an image mask
        orig_mask = np.array([[1, 1, 0],
                              [0, 1, 1]], dtype=np.bool)

        mask_bytes = serialize_image(orig_mask, is_mask=True)
        mask = deserialize_image(mask_bytes, is_mask=True)

        np.testing.assert_array_equal(orig_mask, mask)
        self.assertEqual(orig_mask.shape, mask.shape)
        self.assertEqual(np.bool, mask.dtype)

        # test serializing and deserializing a group of images
        orig_imgs = np.array([[[1, 2], [3, 4]],
                              [[1, 2], [3, 4]]], dtype=np.int32)

        imgs_bytes = serialize_images(orig_imgs)
        imgs = deserialize_images(imgs_bytes, dtype=np.int32)

        np.testing.assert_array_equal(orig_imgs, imgs)
        self.assertEqual(orig_imgs.shape, imgs.shape)
        self.assertEqual(np.float32, img.dtype)
