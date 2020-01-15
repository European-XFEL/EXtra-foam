import unittest

from extra_foam.algorithms.azimuthal_integ import (
    energy2wavelength, compute_q
)


class TestAzimuthalIntegration(unittest.TestCase):
    def testEnergyToWavelength(self):
        # 12.4 keV -> 1 A
        self.assertAlmostEqual(1, 1e10 * energy2wavelength(12.3984e3), places=5)

    def testComputeQ(self):
        # any catchy numbers?
        pass
