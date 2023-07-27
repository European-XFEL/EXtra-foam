"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Ebad Kamil <ebad.kamil@xfel.eu>, Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import warnings
from unittest.mock import patch

import numpy as np

from extra_foam.pipeline.processors.image_transform import ImageTransformProcessor
from extra_foam.pipeline.tests import _TestDataMixin
from extra_foam.config import ImageTransformType

warnings.filterwarnings("ignore", category=RuntimeWarning)


class TestImageTransform(_TestDataMixin):
    def setup_method(self):
        self._proc = ImageTransformProcessor()

    def teardown_method(self):
        pass

    def testUndefined(self):
        self._proc._transform_type = ImageTransformType.UNDEFINED

        for i in range(2):
            data, processed = self.data_with_assembled(1001, (4, 10, 10))
            image = processed.image
            self._proc.process(data)
            assert image.transformed is None
            assert image.transform_type == ImageTransformType.UNDEFINED

    def testConcentricRings(self):
        self._proc._transform_type = ImageTransformType.CONCENTRIC_RINGS

        for i in range(2):
            data, processed = self.data_with_assembled(1000 + i, (4, 10, 10))
            image = processed.image
            self._proc.process(data)
            assert image.transformed is None
            assert image.transform_type == ImageTransformType.CONCENTRIC_RINGS

    @patch("extra_foam.pipeline.processors.image_transform.fourier_transform_2d")
    def testFourierTransform(self, mocked_f):
        self._proc._transform_type = ImageTransformType.FOURIER_TRANSFORM

        fft = self._proc._fft

        for i in range(2):
            data, processed = self.data_with_assembled(1001, (4, 10, 10))
            image = processed.image
            self._proc.process(data)
            mocked_f.assert_called_with(image.masked_mean, logrithmic=fft.logrithmic)
            assert image.transform_type == ImageTransformType.FOURIER_TRANSFORM

    @patch("extra_foam.pipeline.processors.image_transform.edge_detect")
    def testEdgeDetection(self, mocked_f):
        self._proc._transform_type = ImageTransformType.EDGE_DETECTION

        ed = self._proc._ed
        ed.kernel_size = 3
        ed.sigma = 1.0
        ed.threshold = (50, 100)

        for i in range(2):
            data, processed = self.data_with_assembled(1001, (4, 10, 10))
            image = processed.image
            self._proc.process(data)
            mocked_f.assert_called_with(image.masked_mean,
                                        kernel_size=ed.kernel_size,
                                        sigma=ed.sigma,
                                        threshold=ed.threshold)
            assert image.transform_type == ImageTransformType.EDGE_DETECTION
