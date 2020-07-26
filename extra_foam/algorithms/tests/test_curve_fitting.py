import pytest

import numpy as np

from extra_foam.algorithms.curve_fitting import CurveFitting, FittingType


class TestCurveFitting:

    @pytest.mark.parametrize("fitting_type", [item for item in FittingType.__members__.values()
                                              if item != FittingType.UNDEFINED])
    def testFormat(self, fitting_type):
        algo = CurveFitting.create(fitting_type)

        x = np.arange(20, dtype=np.float32)
        y = np.arange(20, dtype=np.float32)
        x[1] = np.nan
        y[2] = np.nan

        with pytest.raises(ValueError, match="infs or NaNs"):
            algo.fit(x, y)

        algo.format(*np.arange(algo.n))

    def testLinear(self):
        algo = CurveFitting.create(FittingType.LINEAR)
        x = np.arange(20)
        y = np.array([102.22005873, 102.66409566, 102.72394046, 103.41332855, 113.29819636,
                      113.33326799, 118.79746557, 120.16134836, 117.62479302, 134.35383486,
                      126.69920673, 126.40798345, 146.41444414, 139.07981823, 142.83380609,
                      151.35626020, 142.75946603, 159.29039204, 146.57267763, 148.43920169])
        popt = algo.fit(x, y, p0=[1, 1])
        assert abs(popt[0] - 100) < 1
        assert abs(popt[1] - 3) < 0.5

    def testCubic(self):
        algo = CurveFitting.create(FittingType.CUBIC)
        x = np.arange(20)
        y = np.array([ 167.78993412,   52.04183763,   77.46176015,   78.47154770,
                       113.37726877,  138.80041972,  119.22930993,   91.48231176,
                        75.16525518,   19.26532408,   -7.09183177,  -82.52070709,
                       -85.38445264, -215.05069450, -326.57202156, -474.94496689,
                      -577.63196986, -888.04779165, -984.03381051, -1214.4252093])
        popt = algo.fit(x, y, p0=[1, 1, 1, 1])
        assert abs(popt[0] - 108) < 1
        assert abs(popt[3] + 0.3) < 0.03

    def testGaussian(self):
        algo = CurveFitting.create(FittingType.GAUSSIAN)

        x = np.array([-3.66430075, -3.26993987, -2.87557899, -2.48121811, -2.08685722, -1.69249634,
                      -1.29813546, -0.90377458, -0.50941370, -0.11505282,  0.27930806,  0.67366894,
                       1.06802982,  1.46239070,  1.85675158,  2.25111246,  2.64547334,  3.03983422,
                       3.43419510,  3.82855598])
        y = np.array([4, 7, 23, 86, 174, 400, 662, 974, 1364, 1566, 1508, 1325, 895, 528, 281, 138,
                      45, 14, 5, 1])
        y += 10
        x += 100

        popt = algo.fit(x, y, p0=[1, 100, 0.9, 10])
        assert 1500 < popt[0] < 1600
        assert abs(popt[1] - 100) < 10
        assert abs(abs(popt[2]) - 1) < 0.1
        assert 10 < popt[3] < 20

    def testLorentzian(self):
        algo = CurveFitting.create(FittingType.LORENTZIAN)

        x = np.arange(20) + 90
        y = np.array([ 34.94049126, 40.02754777, 39.76174261, 35.29033723, 34.90574794,
                       34.15116848, 43.13161911, 44.36247681, 50.21500427, 76.86329198,
                      105.22065798, 78.23873672, 52.12398640, 40.89729233, 40.60966561,
                       35.44761773, 40.06147386, 39.26495213, 36.69235285, 32.29345426])

        popt = algo.fit(x, y, p0=[100, 1, 100, 1])
        assert abs(popt[0] - 100) < 10
        assert abs(popt[1] - 1) < 0.2
        assert abs(popt[2] - 100) < 1
        assert abs(popt[3] - 30) < 5

    def testERF(self):
        algo = CurveFitting.create(FittingType.ERF)

        x = np.arange(20) - 5
        y = np.array([4.20715134, 4.70349781, 4.29314648, 4.25909993, 4.46186085, 4.61299807,
                      4.07488064, 4.01316896, 4.09148713, 4.39301747, 4.26477643, 4.19947183,
                      4.94517831, 5.69089186, 6.35998697, 6.59577612, 6.74455303, 6.69911633,
                      6.19510895, 6.31749930])
        popt = algo.fit(x, y, p0=[1, 1, 1, 1])

        assert abs(popt[0] + 1) < 0.2
        assert abs(popt[2] - 8) < 1
        assert abs(popt[3] - 5) < 1
