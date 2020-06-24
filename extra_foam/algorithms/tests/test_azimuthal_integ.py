import unittest

import numpy as np

from pyFAI.azimuthalIntegrator import AzimuthalIntegrator as PyfaiAzimuthalIntegrator

from extra_foam.algorithms.azimuthal_integ import (
    AzimuthalIntegrator, energy2wavelength, compute_q
)


class TestAzimuthalIntegrationMisc(unittest.TestCase):
    def testEnergyToWavelength(self):
        # 12.4 keV -> 1 A
        self.assertAlmostEqual(1, 1e10 * energy2wavelength(12.3984e3), places=5)

    def testComputeQ(self):
        # any catchy numbers?
        pass


class TestAzimuthalIntegrator(unittest.TestCase):
    def testGeneral(self):
        img = np.ones((10, 10), dtype=float)

        distance = 1.
        pixel1, pixel2 = 1e-4, 2e-4
        poni1, poni2 = 2 * pixel1, 4 * pixel2
        wavelength = 1e-10

        npt = 10

        pyfai_integrator = PyfaiAzimuthalIntegrator(
            dist=distance, poni1=poni1, poni2=poni2, pixel1=pixel1, pixel2=pixel2,
            wavelength=wavelength)
        q_gt, s_gt = pyfai_integrator.integrate1d(img, npt=npt, correctSolidAngle=True)

        integrator = AzimuthalIntegrator(
            dist=distance, poni1=poni1, poni2=poni2, pixel1=pixel1, pixel2=pixel2,
            wavelength=wavelength)
        q, s = integrator.integrate1d(img, npt=npt)
