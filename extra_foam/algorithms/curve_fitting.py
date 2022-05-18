"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>, Ebad Kamil <ebad.kamil@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from abc import abstractmethod
from enum import Enum

import numpy as np
from scipy.optimize import curve_fit
from scipy import special


class FittingType(Enum):
    UNDEFINED = "None"
    LINEAR = "Linear"
    CUBIC = "Cubic"
    GAUSSIAN = "Gaussian"
    LORENTZIAN = "Lorentzian"
    ERF = "erf"


class CurveFitting:
    @abstractmethod
    class _BaseCurveFitting:
        def __init__(self):
            self.n = 0

        def fit(self, x, y, *, p0=None):
            popt, _ = curve_fit(self, x, y, p0=p0, check_finite=True)
            return popt

        def parameters(self):
            """
            Return a list of parameters by name. Intended for display in GUIs etc.
            """
            raise NotImplementedError()

        def guess_p0(self, x, y):
            """
            Return initial guesses for all required parameters.
            """
            raise NotImplementedError()

        @staticmethod
        def format(*args):
            raise NotImplementedError()

    class Linear(_BaseCurveFitting):
        def __init__(self):
            super().__init__()
            self.n = 2

        def __call__(self, x, a, b):
            return a + b * x

        def parameters(self):
            return ["Offset", "Slope"]

        def guess_p0(self, x, y):
            offset = np.nanmin(y)
            slope = (y[-1] - y[0]) / (x[-1] - x[0])

            return offset, slope

        @staticmethod
        def format(a, b):
            """Override."""
            return f"y = a + b * x, \na = {a:.4E}, b = {b:.4E}"

    class Cubic(_BaseCurveFitting):
        def __init__(self):
            super().__init__()
            self.n = 4

        def __call__(self, x, a, b, c, d):
            return a + b * x + c * x**2 + d * x**3

        @staticmethod
        def format(a, b, c, d):
            return f"y = a + b * x + c * x^2 + d * x^3, \n" \
                   f"a = {a:.4E}, b = {b:.4E}, c = {c:.4E}, d = {d:.4E}"

    class Gaussian(_BaseCurveFitting):
        def __init__(self):
            super().__init__()
            self.n = 4

        def __call__(self, x, a, b, c, d):
            return a * np.exp(-(x - b)**2 / (2 * c**2)) + d

        def parameters(self):
            return ["Amplitude", "Mean", "Std dev.", "Offset"]

        def guess_p0(self, x, y):
            max_idx = np.nanargmax(y)
            mean = x[max_idx]
            amplitude = max(1, y[max_idx])
            stdev = len(y) / 4
            offset = np.nanmin(y)

            return [amplitude, mean, stdev, offset]

        @staticmethod
        def format(a, b, c, d):
            return f"y = a * exp(-(x - b)^2 / (2 * c^2)) + d, \n" \
                   f"a = {a:.4E}, b = {b:.4E}, c = {c:.4E}, d = {d:.4E}"

    class Lorentzian(_BaseCurveFitting):
        def __init__(self):
            super().__init__()
            self.n = 4

        def __call__(self, x, a, b, c, d):
            return a / ((x - b)**2 + c**2) + d

        @staticmethod
        def format(a, b, c, d):
            return f"y = a / (4 * (x - b)^2 + c ^ 2) + d, \n" \
                   f"a = {a:.4E}, b = {b:.4E}, c = {c:.4E}, d = {d:.4E}"

    class Erf(_BaseCurveFitting):
        def __init__(self):
            super().__init__()
            self.n = 4

        def __call__(self, x, a, b, c, d):
            return a * special.erf(b * (x - c)) + d

        @staticmethod
        def format(a, b, c, d):
            return f"y = a * erf(b * (x - c)) + d, \n" \
                   f"a = {a:.4E}, b = {b:.4E}, c = {c:.4E}, d = {d:.4E}"

    @classmethod
    def create(cls, fitting_type):
        if fitting_type == FittingType.LINEAR:
            return cls.Linear()

        if fitting_type == FittingType.CUBIC:
            return cls.Cubic()

        if fitting_type == FittingType.GAUSSIAN:
            return cls.Gaussian()

        if fitting_type == FittingType.LORENTZIAN:
            return cls.Lorentzian()

        if fitting_type == FittingType.ERF:
            return cls.Erf()
