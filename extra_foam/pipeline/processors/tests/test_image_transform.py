"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Ebad Kamil <ebad.kamil@xfel.eu>, Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from unittest.mock import patch
import pytest

import numpy as np

from extra_foam.pipeline.processors.image_transform import ImageTransformProcessor
from extra_foam.pipeline.tests import _TestDataMixin
from extra_foam.config import ImageTransformType

np.warnings.filterwarnings("ignore", category=RuntimeWarning)


class TestImageTransform(_TestDataMixin):
    @pytest.fixture(autouse=True)
    def setUp(self):
        self._proc = ImageTransformProcessor()

    def testFourierTransform(self):
        self._proc._transform_type = ImageTransformType.FOURIER_TRANSFORM
        # self._proc.process({})

    def testEdgeDetection(self):
        self._proc._transform_type = ImageTransformType.EDGE_DETECTION
        # self._proc.process({})
