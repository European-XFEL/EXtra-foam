"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from ..pyqtgraph import ColorMap, intColor, mkPen, mkBrush
from ..pyqtgraph.graphicsItems.GradientEditorItem import Gradients


class QualitativeColors:

    def __init__(self, alpha=255):
        self.n = (251, 154, 153, alpha)  # pink
        self.r = (227, 26, 28, alpha)  # red
        self.o = (255, 127, 0, alpha)  # orange
        self.y = (255, 255, 153, alpha)  # yellow
        self.c = (166, 206, 227, alpha)  # cyan
        self.b = (31, 120, 180, alpha)  # blue
        self.s = (178, 223, 138, alpha)  # grass green
        self.g = (51, 160, 44, alpha)  # green
        self.p = (106, 61, 154, alpha)  # purple
        self.d = (202, 178, 214, alpha)  # orchid
        self.w = (177, 89, 40, alpha)  # brown


Colors = QualitativeColors


class SequentialColors:

    def __init__(self, alpha=255):
        # red
        self.r = [
            (153, 0, 13, alpha),
            (203, 24, 29, alpha),
            (239, 59, 44, alpha),
            (251, 106, 74, alpha),
            (252, 146, 114, alpha)
        ]
        # blue
        self.b = [
            (158, 202, 225, alpha),
            (107, 174, 214, alpha),
            (66, 146, 198, alpha),
            (33, 113, 181, alpha),
            (8, 69, 148, alpha),
        ]
        # magenta
        self.m = [
            (74, 20, 134, alpha),
            (106, 81, 163, alpha),
            (128, 125, 186, alpha),
            (158, 154, 200, alpha),
            (188, 189, 220, alpha)
        ]
        # green
        self.g = [
            (161, 217, 155, alpha),
            (116, 196, 118, alpha),
            (65, 171, 93, alpha),
            (35, 139, 69, alpha),
            (0, 90, 50, alpha),
        ]

    def s1(self, n):
        """Return a list with n sequential colors.

        The colors will be repeated if n is larger than the number of
        pre-defined colors.
        """
        c = self.r + self.b + self.m + self.g

        if n > len(c):
            c = c * (int(n/len(c)) + 1)
        return c[:n]


def make_pen(color, *, width=1, alpha=255, **kwargs):
    """Convenient function for making QPen.

    :param int width: width of QPen.

    Note: due to a bug in Qt, setting width greater than 1 will significantly
          degrade the performance.
          https://github.com/pyqtgraph/pyqtgraph/issues/533
    """
    if color is None:
        return mkPen(None)

    if isinstance(color, int):
        return mkPen(intColor(color, **kwargs), width=width)

    if isinstance(color, tuple) and len(color) == 4:
        # (r, g, b, alpha)
        return mkPen(color)

    return mkPen(getattr(Colors(alpha=alpha), color[0]), width=width, **kwargs)


def make_brush(color, alpha=255):
    return mkBrush(getattr(Colors(alpha=alpha), color[0]))


# Valid keys: thermal, flame, yellowy, bipolar, spectrum, cyclic, greyclip, grey
colorMapFactory = \
    {name: ColorMap(*zip(*Gradients[name]["ticks"]))
     for name in Gradients.keys()}


lookupTableFactory = {name: cmap.getLookupTable()
                      for name, cmap in colorMapFactory.items()}
