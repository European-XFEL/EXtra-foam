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
from karaboFAI.pipeline.serialization import (
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
        shape = (100, 120)
        orig_img = np.ones(shape, dtype=np.float32)

        img_bytes = serialize_image(orig_img)

        img = deserialize_image(img_bytes)
        self.assertEqual(shape, img.shape)
        self.assertEqual(np.float32, img.dtype)

        # test serializing and deserializing a group of images
        shape = (4, 100, 120)
        orig_imgs = np.ones(shape, dtype=np.float32)

        imgs_bytes = serialize_images(orig_imgs)

        imgs = deserialize_images(imgs_bytes)
        self.assertEqual(shape, imgs.shape)
        self.assertEqual(np.float32, img.dtype)
