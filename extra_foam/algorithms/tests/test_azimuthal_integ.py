import unittest
import pytest

import numpy as np

from extra_foam.algorithms.azimuthal_integ import (
    AzimuthalIntegrator, ConcentricRingFinder, energy2wavelength, compute_q
)


class TestAzimuthalIntegrationMisc(unittest.TestCase):
    def testEnergyToWavelength(self):
        # 12.4 keV -> 1 A
        self.assertAlmostEqual(1, 1e10 * energy2wavelength(12.3984e3), places=5)

    def testComputeQ(self):
        # any catchy numbers?
        pass


class TestAzimuthalIntegrator:
    @pytest.mark.parametrize("dtype", [np.float32, np.uint16, np.int16])
    def testIntegrate1D(self, dtype):
        img = np.arange(1024).reshape((16, 64)).astype(dtype)

        distance = 0.2
        pixel1, pixel2 = 1e-4, 2e-4
        poni1, poni2 = -6 * pixel1, 130 * pixel2
        wavelength = 1e-10

        integrator = AzimuthalIntegrator(
            dist=distance, poni1=poni1, poni2=poni2, pixel1=pixel1, pixel2=pixel2,
            wavelength=wavelength)

        q0, s0 = integrator.integrate1d(img, npt=0)
        q1, s1 = integrator.integrate1d(img, npt=1)
        assert q0 == q1
        assert s0 == s1

        q10, s10 = integrator.integrate1d(img, npt=10, min_count=img.size)
        assert not np.any(s10)

        q100, s100 = integrator.integrate1d(img, npt=999)

        # TODO: test correctness. For example, compare with pyFAI.


class TestConcentricRingsFinder:
    @pytest.mark.parametrize("dtype", [np.float32, np.uint16, np.int16])
    def testRingDetection(self, dtype):
        img = np.arange(1024).reshape((16, 64)).astype(dtype)

        pixel_x, pixel_y = 2e-4, 1e-4
        cx, cy = 10, 20
        min_count = 5

        finder = ConcentricRingFinder(pixel_x, pixel_y)
        finder.search(img, cx, cy, min_count)
        finder.integrate(img, cx, cy, min_count)

        # TODO: test correctness
